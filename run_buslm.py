import numpy as np
import torch
from tqdm.auto import tqdm
import torch.optim as optim
from pathlib import Path
import utils
import os
from dataloader_bus import DataLoaderTrain, DataLoaderTest
from infer_embedding import news_feature, infer_news_embedding

from preprocess import read_news_bert, get_doc_input_bert
from train_preprocess2 import read_news_buslm
from BusLm import ModelBert
from parameters import parse_args

from tnlrv3.bus_modeling import BusLM_rec
from tnlrv3.modeling import TuringNLRv3ForSequenceClassification
from tnlrv3.configuration_tnlrv3 import TuringNLRv3Config
from tnlrv3.tokenization_tnlrv3 import TuringNLRv3Tokenizer

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import time
import logging
import math
import random

MODEL_CLASSES = {
    'tnlrv3': (TuringNLRv3Config, TuringNLRv3ForSequenceClassification, TuringNLRv3Tokenizer),
    'buslm': (TuringNLRv3Config, BusLM_rec, TuringNLRv3Tokenizer)
}


def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size,)
    torch.cuda.set_device(rank)

    # Explicitly setting seed
    torch.manual_seed(42)

def cleanup():
    dist.destroy_process_group()

def load_bert(args):
    assert args.bert_model in ("tnlrv3","buslm")
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.bert_model]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        output_hidden_states = True)

    config.bus_num = args.bus_num

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)
    bert_model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config)
    return bert_model,tokenizer

def warmup_linear(args,step):
    if step <= args.warmup_step:
        return step/args.warmup_step
    return max(1e-4,(args.schedule_step-step)/(args.schedule_step-args.warmup_step))



def train(local_rank,args,cache,news_idx_incache,prefetch_step):
    '''
    shared memory:
    cache: (array) global_cache
    news_idx_incache: (dict) {news_id:index in cache}
    prefetch_step: (list) the step of data generation, for sync of dataloader
    '''
    utils.setuplogging()
    os.environ["RANK"] = str(local_rank)
    setup(local_rank, args.world_size)
    device = torch.device("cuda", local_rank)

    cache = cache[0]

    logging.info('loading model: {}'.format(args.bert_model))
    bert_model,tokenizer = load_bert(args)

    if args.freeze_bert:
        logging.info('!!! Freeze the parameters of {}'.format(args.bert_model))
        for param in bert_model.parameters():
            param.requires_grad = False

        # choose which block trainabel
        for index, layer in enumerate(bert_model.bert.encoder.layer):
            if index in args.finetune_blocks:
                # logging.info(f"finetune {index} block")
                for param in layer.parameters():
                    param.requires_grad = True
    else:
        logging.info('!!!Not freeze the parameters of {}'.format(args.bert_model))

    news_feature, category_dict, subcategory_dict,domain_dict= read_news_buslm(
        os.path.join(args.root_data_dir,f'docs_body.tsv'),
        args, tokenizer,'train')
    logging.info('-----------news_num:{}-----------'.format(len(news_feature)))

    #init the news_idx_incache;  sync
    assert args.cache_num >= len(news_feature)
    if local_rank == 0:
        idx = 0
        for news in news_feature.keys():
            news_idx_incache[news] = [idx,-args.max_step_in_cache]
            idx += 1
    dist.barrier()


    model = ModelBert(args, bert_model, len(category_dict), len(domain_dict), len(subcategory_dict))
    model = model.to(device)
    if args.world_size > 1:
        ddp_model = DDP(model,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)
    else:
        ddp_model = model
    lr_scaler = args.world_size
    if args.warmup_lr:
        rest_param = filter(lambda x: id(x) not in list(map(id, bert_model.parameters())), ddp_model.parameters())
        optimizer = optim.Adam([
            {'params': bert_model.parameters(), 'lr': args.pretrain_lr*warmup_linear(args,1)},
            {'params': rest_param, 'lr': args.lr*warmup_linear(args,1)}])
    else:
        rest_param = filter(lambda x: id(x) not in list(map(id, bert_model.parameters())), ddp_model.parameters())
        optimizer = optim.Adam([
            {'params': bert_model.parameters(), 'lr': args.pretrain_lr * lr_scaler},
            {'params': rest_param, 'lr': args.lr * lr_scaler}])

    dataloader = DataLoaderTrain(
        data_dir=os.path.join(args.root_data_dir,
                              f'autoregressive'),
        args=args,
        local_rank=local_rank,
        cache=cache,
        news_idx_incache=news_idx_incache,
        prefetch_step=prefetch_step,
        world_size=args.world_size,
        worker_rank=local_rank,
        cuda_device_idx=local_rank,
        news_combined=news_feature,
        enable_prefetch=True,
        enable_shuffle=True,
        enable_stream_queue=True,
        enable_gpu=args.enable_gpu,
    )

    logging.info('Training...')

    for ep in range(args.epochs):
        hit_ratio = 0
        ht_num = 0
        hit_num = 0
        all_num = 0
        loss = 0.0
        start_time = time.time()
        usernum = 0
        for cnt, batch in tqdm(enumerate(dataloader)):
            if cnt > args.max_steps_per_epoch:
                break

            address_cache, update_cache, batch = batch
            usernum += batch[3].shape[0]

            if args.enable_gpu:
                segments = [
                    x.cuda(non_blocking=True) for x in batch[0]]
                token_masks = [
                    x.cuda(non_blocking=True) for x in batch[1]]
                seg_masks,elements, batch_hist, batch_mask, batch_negs = [
                    x.cuda(non_blocking=True) for x in batch[2:]]
            else:
                segments,token_masks, seg_masks, elements, batch_hist, batch_mask, batch_negs = batch

            if address_cache is not None:
                cache_vec = torch.FloatTensor(cache[address_cache]).cuda(non_blocking=True)
                hit_ratio += cache_vec.size(0) / (seg_masks.size(0) + cache_vec.size(0))
                ht_num += 1
                hit_num += cache_vec.size(0)
                all_num += (seg_masks.size(0) + cache_vec.size(0))
            else:
                cache_vec = None


            bz_loss,encode_vecs = ddp_model(segments,token_masks, seg_masks, elements, cache_vec, batch_hist, batch_mask, batch_negs)
            loss += bz_loss.data.float()
            optimizer.zero_grad()
            bz_loss.backward()
            optimizer.step()

            if args.drop_encoder_ratio > 0:
                encode_vecs = encode_vecs.detach().cpu().numpy()
                cache[update_cache] = encode_vecs

            if args.warmup_lr:
                optimizer.param_groups[0]['lr'] = args.pretrain_lr*warmup_linear(args,cnt+1)  #* lr_scaler
                optimizer.param_groups[1]['lr'] = args.lr*warmup_linear(args,cnt+1)   #* lr_scaler
                if cnt % 500 == 0:
                    logging.info(
                        'learning_rate:{},{}'.format(args.pretrain_lr*warmup_linear(args,cnt+1), args.lr*warmup_linear(args,cnt+1)))
            else:
                if cnt == 25000:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = 1e-5 * lr_scaler
                logging.info(f"change lr rate {1e-5 * lr_scaler}")

            if cnt % 100 == 0:
                logging.info(
                    '[{}] cost_time:{} step:{},  usernum: {}, train_loss: {:.5f}'.format(
                        local_rank, time.time()-start_time, cnt, usernum, loss.data / (cnt+1)))
                if hit_num > 0:
                    logging.info(
                        '[{}] step:{}, avarage hit ratio:{}'.format(
                            local_rank, cnt, hit_ratio / ht_num))
                    logging.info(
                        '[{}] step:{}, all hit ratio:{}'.format(
                            local_rank, cnt, hit_num / all_num))

            # save model minibatch
            if local_rank == 0 and (cnt+1) % args.save_steps == 0:
                ckpt_path = os.path.join(args.model_dir, f'{args.savename}-epoch-{ep + 1}-{cnt}.pt')
                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'category_dict': category_dict,
                        'domain_dict': domain_dict,
                        'subcategory_dict': subcategory_dict
                    }, ckpt_path)
                logging.info(f"Model saved to {ckpt_path}")

            dist.barrier()

        loss /= (cnt+1)
        logging.info('epoch:{}, loss:{},usernum:{}, time:{}'.format(ep+1, loss, usernum,time.time()-start_time))

        # save model last of epoch
        if local_rank == 0:
            ckpt_path = os.path.join(args.model_dir, '{}-epoch-{}.pt'.format(args.savename,ep+1))
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'category_dict': category_dict,
                    'domain_dict': domain_dict,
                    'subcategory_dict': subcategory_dict
                }, ckpt_path)
            logging.info(f"Model saved to {ckpt_path}")
        logging.info("time:{}".format(time.time()-start_time))
    dataloader.join()

    cleanup()




if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    args = parse_args()
    args.world_size = 4
    # args.root_data_dir= '/data/t-shxiao/rec/data/'
    args.config_name= '../Turing/buslm-base-uncased-config.json'
    # args.tokenizer_name='/data/t-shxiao/adalm16/data/bert-pretrained-cache/vocab.txt'
    # args.model_name_or_path='/data/t-shxiao/adalm16/data/finetune_model/model_3200.bin'
    args.batch_size = 20000
    args.cache_num = 910000
    args.drop_encoder_ratio = 1

    args.news_attributes = ['title', 'abstract', 'body']  # , 'abstract', 'category', 'domain', 'subcategory']
    args.bus_connection = True
    args.body_seg_num = 2
    args.min_title_length = 20
    args.block_num=2
    args.bert_model = 'buslm'

    seg_num = 0
    for name in args.news_attributes:
        if name == 'title':
            seg_num += 1
        elif name == 'abstract':
            seg_num += 1
        elif name == 'body':
            seg_num += args.body_seg_num
    args.seg_num = seg_num
    if seg_num>1 and args.bus_connection:
        args.bus_num = seg_num
    else:
        args.bus_num = 0


    if 'train' in args.mode:
        print('-----------trian------------')
        if args.world_size > 1:
        # synchronizer = Barrier(args.world_size)
        # serializer = Lock()
            cache = np.zeros((args.cache_num,args.news_dim))
            global_cache = mp.Manager().list([cache])
            news_idx_incache  = mp.Manager().dict()
            global_prefetch_step = mp.Manager().list([0]*args.world_size)
            mp.spawn(train,
                     args = (args,global_cache,news_idx_incache,global_prefetch_step),
                         nprocs=args.world_size,
                         join=True)
        else:
            cache = [np.zeros((args.cache_num, args.news_dim))]#[torch.zeros(args.cache_num, args.news_dim, requires_grad=False) for x in range(args.world_size)]
            news_idx_incache = {}
            prefetch_step = [0]
            train(0,args,cache,news_idx_incache,prefetch_step)

    if 'test' in args.mode:
        print('-----------test------------')
        if args.world_size > 1:
            mp.spawn(test,
                     args = (args,),
                         nprocs=args.world_size,
                         join=True)
        else:
            test(0,args)



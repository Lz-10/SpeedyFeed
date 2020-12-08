import numpy as np
import torch
from tqdm.auto import tqdm
import torch.optim as optim
from pathlib import Path
import utils
import os
from dataloader_cache_allitem import DataLoaderTrain, DataLoaderTest
from infer_embedding import news_feature, infer_news_embedding

from preprocess import read_news_bert, get_doc_input_bert
from train_preprocess import read_news_bert_nopadding
from ddp_model import ModelBert
from parameters import parse_args

from tnlrv3.modeling import TuringNLRv3ForSequenceClassification
from tnlrv3.configuration_tnlrv3 import TuringNLRv3Config
from tnlrv3.tokenization_tnlrv3 import TuringNLRv3Tokenizer

from torch.multiprocessing import Barrier,Lock
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import time
import logging
import math
import random

MODEL_CLASSES = {
    'tnlrv3': (TuringNLRv3Config, TuringNLRv3ForSequenceClassification, TuringNLRv3Tokenizer),
    'bert': (None,None,None)
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
    assert args.bert_model in ("tnlrv3","bert")
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.bert_model]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        output_hidden_states = True)
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

    news_combined, news_length, category_dict, domain_dict, subcategory_dict = read_news_bert_nopadding(
        os.path.join(args.root_data_dir,f'docs.tsv'),
        args, tokenizer )
    logging.info('-----------news_num:{}-----------'.format(len(news_combined)))


    assert args.cache_num >= len(news_length)
    if local_rank == 0:
        idx = 0
        for news in news_length.keys():
            news_idx_incache[news] = [idx,-args.max_step_in_cache]
            idx += 1
    dist.barrier()


    model = ModelBert(args, bert_model, device, len(category_dict), len(domain_dict), len(subcategory_dict))
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
        news_combined=news_combined,
        news_length=news_length,
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

            title_length, address_cache, update_cache, batch = batch
            usernum += batch[3].shape[0]

            if args.enable_gpu:
                batch_news_feature, batch_hist, batch_mask, batch_negs = [
                    x.cuda(non_blocking=True) for x in batch]
            else:
                batch_news_feature, batch_hist, batch_mask, batch_negs = batch

            if address_cache is not None:
                cache_vec = torch.FloatTensor(cache[address_cache]).cuda(non_blocking=True)
                hit_ratio += cache_vec.size(0) / (batch_news_feature.size(0) + cache_vec.size(0))
                ht_num += 1
                hit_num += cache_vec.size(0)
                all_num += (batch_news_feature.size(0) + cache_vec.size(0))
            else:
                cache_vec = None


            bz_loss,encode_vecs = ddp_model(batch_news_feature, cache_vec, batch_hist, batch_mask, batch_negs,title_length)
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



def test(local_rank,args):
    setup(local_rank, args.world_size)

    if args.load_ckpt_name is not None:
        ckpt_path = utils.get_checkpoint(args.model_dir, args.load_ckpt_name)
    else:
        ckpt_path = utils.latest_checkpoint(args.model_dir)
    assert ckpt_path is not None, 'No ckpt found'
    checkpoint = torch.load(ckpt_path)
    subcategory_dict = checkpoint['subcategory_dict']
    category_dict = checkpoint['category_dict']
    domain_dict = checkpoint['domain_dict']

    device = torch.device("cuda", local_rank)

    # load model

    bert_model,tokenizer = load_bert(args)
    model = ModelBert(args, bert_model, device, len(category_dict), len(domain_dict), len(subcategory_dict))
    model = model.to(device)
    if args.world_size > 1:
        ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    else:
        ddp_model = model

    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    ddp_model.load_state_dict(checkpoint['model_state_dict'],map_location=map_location)
    logging.info(f"Model loaded from {ckpt_path}")

    model.eval()
    torch.set_grad_enabled(False)

    news, news_index = read_news_bert(
        os.path.join(args.root_data_dir,
                     f'test_files/docs.tsv'),
        args,
        tokenizer,
        mode='test'
    )


    news_combined = news_feature(args, news, news_index, category_dict, domain_dict, subcategory_dict)
    news_scoring = infer_news_embedding(args,news_combined,model)

    logging.info("news scoring num: {}".format(news_scoring.shape[0]))


    dataloader = DataLoaderTest(
        news_index=news_index,
        news_scoring=news_scoring,
        news_bias_scoring=None,
        data_dir=os.path.join(args.root_data_dir,
                              f'test_files'),
        filename_pat=args.filename_pat,
        args=args,
        world_size= args.world_size,
        worker_rank=local_rank,
        cuda_device_idx=local_rank,
        enable_prefetch=True,
        enable_shuffle=True,
        enable_gpu=args.enable_gpu,
    )
    from metrics import roc_auc_score, ndcg_score, mrr_score, ctr_score

    AUC = [[], []]
    MRR = [[], []]
    nDCG5 = [[], []]
    nDCG10 = [[], []]
    CTR1 = [[], []]
    CTR3 = [[], []]
    CTR5 = [[], []]
    CTR10 = [[], []]


    def print_metrics(hvd_local_rank, cnt, x):
        logging.info("[{}] Ed: {}: {}".format(hvd_local_rank, cnt, \
                                              '\t'.join(["{:0.2f}".format(i * 100) for i in x])))

    def get_mean(arr):
        return [np.array(i).mean() for i in arr]

    # for cnt, (log_vecs, log_mask, news_vecs, news_bias, labels) in enumerate(dataloader):

    for cnt, (log_vecs, log_mask, news_vecs, labels) in tqdm(enumerate(dataloader)):
        his_lens = torch.sum(log_mask, dim=-1).to(torch.device("cpu")).detach().numpy()

        if args.enable_gpu:
            log_vecs = log_vecs.cuda(non_blocking=True)
            log_mask = log_mask.cuda(non_blocking=True)


        user_vecs = model.user_encoder.infer_user_vec(log_vecs, log_mask).to(torch.device("cpu")).detach().numpy()

        for index, user_vec, news_vec, label, his_len in zip(
                range(len(labels)), user_vecs, news_vecs, labels, his_lens):

            if label.mean() == 0 or label.mean() == 1:
                continue

            score = np.dot(
                news_vec, user_vec
            )

            auc = roc_auc_score(label, score)
            mrr = mrr_score(label, score)
            ndcg5 = ndcg_score(label, score, k=5)
            ndcg10 = ndcg_score(label, score, k=10)
            ctr1 = ctr_score(label, score, k=1)
            ctr3 = ctr_score(label, score, k=3)
            ctr5 = ctr_score(label, score, k=5)
            ctr10 = ctr_score(label, score, k=10)

            AUC[0].append(auc)
            MRR[0].append(mrr)
            nDCG5[0].append(ndcg5)
            nDCG10[0].append(ndcg10)
            CTR1[0].append(ctr1)
            CTR3[0].append(ctr3)
            CTR5[0].append(ctr5)
            CTR10[0].append(ctr10)


            if his_len <= 5:
                AUC[1].append(auc)
                MRR[1].append(mrr)
                nDCG5[1].append(ndcg5)
                nDCG10[1].append(ndcg10)
                CTR1[1].append(ctr1)
                CTR3[1].append(ctr3)
                CTR5[1].append(ctr5)
                CTR10[1].append(ctr10)


        if cnt == 0:
            for i in range(2):
                print_metrics(hvd_rank, 0,
                              get_mean([AUC[i], MRR[i], nDCG5[i], nDCG10[i], CTR1[i], CTR3[i], CTR5[i], CTR10[i]]))
        if (cnt + 1) % args.log_steps == 0:
            for i in range(2):
                print_metrics(hvd_rank, (cnt + 1) * args.batch_size, get_mean([AUC[i], MRR[i], nDCG5[i], \
                                                                               nDCG10[i], CTR1[i], CTR3[i], CTR5[i],
                                                                               CTR10[i]]))

    dataloader.join()

    for i in range(2):
        print_metrics(hvd_rank, (cnt + 1) * args.batch_size, get_mean([AUC[i], MRR[i], nDCG5[i], \
                                                                       nDCG10[i], CTR1[i], CTR3[i], CTR5[i],
                                                                       CTR10[i]]))




if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    args = parse_args()
    args.world_size = 4
    # args.root_data_dir= '/data/t-shxiao/rec/data/'
    # args.config_name= '/data/t-shxiao/adalm16/data/model/unilm2-base-uncased-config.json'
    # args.tokenizer_name='/data/t-shxiao/adalm16/data/bert-pretrained-cache/vocab.txt'
    # args.model_name_or_path='/data/t-shxiao/adalm16/data/finetune_model/model_3200.bin'
    # args.batch_size = 10000
    # args.cache_num = 910000
    # args.drop_encoder_ratio = 1

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



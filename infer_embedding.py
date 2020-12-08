import numpy as np
import torch
from tqdm.auto import tqdm
from pathlib import Path
import utils
import os
import sys
import logging
from torch.utils.data import Dataset, DataLoader
from auto_reg_model import ModelBert
# from auto_reg_model_mean import ModelBert
from preprocess import infer_read_news_bert,get_doc_input_bert
from parameters import parse_args

from tnlrv3.modeling import TuringNLRv3ForSequenceClassification
from tnlrv3.configuration_tnlrv3 import TuringNLRv3Config
from tnlrv3.tokenization_tnlrv3 import TuringNLRv3Tokenizer
# import horovod.torch as hvd
import math
import time
from recall_test import get_day_item,CreatIndex,get_similarity,get_result

MODEL_CLASSES = {
    'tnlrv3': (TuringNLRv3Config, TuringNLRv3ForSequenceClassification, TuringNLRv3Tokenizer),
    'bert': (None,None,None)
}

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



def news_feature(args,news, news_index, category_dict, domain_dict, subcategory_dict):
    news_title, news_title_attmask, \
    news_abstract, news_abstract_attmask, \
    news_body, news_body_attmask, \
    news_category, news_domain, news_subcategory = get_doc_input_bert(
        news, news_index, category_dict, domain_dict, subcategory_dict, args)

    news_combined = np.concatenate([
        x for x in
        [news_title, news_title_attmask, \
         news_abstract, news_abstract_attmask, \
         news_body, news_body_attmask, \
         news_category, news_domain, news_subcategory]
        if x is not None], axis=1)
    return news_combined


class NewsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]


def news_collate_fn(arr):
    arr = torch.LongTensor(arr)
    return arr


def infer_news_embedding(args,news_combined,model):
    news_dataset = NewsDataset(news_combined)
    news_dataloader = DataLoader(news_dataset,
                                 batch_size=args.batch_size,    #args.batch_size * 4,
                                 num_workers=args.num_workers,
                                 collate_fn=news_collate_fn)

    news_scoring = []
    with torch.no_grad():
        for input_ids in tqdm(news_dataloader):
            input_ids = input_ids.cuda()
            news_vec = model.news_encoder(input_ids,title_length=args.num_words_title)
            news_vec = news_vec.to(torch.device("cpu")).detach().numpy()
            news_scoring.extend(news_vec)

    news_scoring = np.array(news_scoring)

    return news_scoring



def get_embed(args,ckpt_path,data_dir):
    # hvd_size, hvd_rank, hvd_local_rank = utils.init_hvd_cuda(
    #     args.enable_hvd, args.enable_gpu)


    assert ckpt_path is not None, 'No ckpt found'
    checkpoint = torch.load(ckpt_path)
    subcategory_dict = checkpoint['subcategory_dict']
    category_dict = checkpoint['category_dict']
    # word_dict = checkpoint['word_dict']
    domain_dict = checkpoint['domain_dict']

    # load model

    bert_model, tokenizer = load_bert(args)
    model = ModelBert(args, bert_model, len(category_dict), len(domain_dict), len(subcategory_dict))

    if args.enable_gpu:
        model.cuda()

    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Model loaded from {ckpt_path}")

    # if args.enable_hvd:
    #     hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    model.eval()
    torch.set_grad_enabled(False)

    news, news_index = infer_read_news_bert(os.path.join('/data/t-shxiao/octopus/Data/index_docid_title.tsv'), args, tokenizer, mode='test')
    news_combined = news_feature(args, news, news_index, category_dict, domain_dict, subcategory_dict)

    news_scoring = infer_news_embedding(args, news_combined, model)

    logging.info("news scoring num: {}".format(news_scoring.shape[0]))

    return model,  news_scoring



def hist_pos(date,data_dir):
    if date < 10:
        # filename = os.path.join(data_dir,"history_pos/history_pos-10-0{}.tsv".format(date))
        filename = "/Feeds-nfs/data/t-wangli/projects/recall_test/ProcessedData/history-pos/history-pos-08-0{}.tsv".format(date)
    else:
        # filename = os.path.join(data_dir,"history_pos/history_pos-10-{}.tsv".format(date))
        filename = "/Feeds-nfs/data/t-wangli/projects/recall_test/ProcessedData/history-pos/history-pos-08-{}.tsv".format(date)
    history = []
    positems = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            linesplit = line.strip().split('\t')
            uid, hist, pos = linesplit
            history.append([int(x) for x in hist.split(';')])
            positems.append([int(x) for x in pos.split(';')])
    return history, positems

def generate_user_data(history, all_item_embedding, user_batch_size):
    # history = np.array(history)
    step = math.ceil(len(history) / user_batch_size)
    for i in range(step):
        start = user_batch_size * i
        end = min(user_batch_size * (i + 1), len(history))
        index = history[start:end]
        # print(index)
        index = np.array(index)
        mask = [[1]*len(index[0])]
        yield all_item_embedding[index], mask


def CreatUserEmbed(history, all_item_embedding, user_batch_size, model):
    user_embedding = []
    with torch.no_grad():
        user_progress = tqdm(enumerate(generate_user_data(history, all_item_embedding, user_batch_size)),
                             dynamic_ncols=True,
                             total=(math.ceil(len(history) / user_batch_size)))
        for step, batch in user_progress:
            log_vecs,log_mask = batch
            log_vecs = torch.tensor(log_vecs).cuda(non_blocking=True)
            log_mask = torch.tensor(log_mask).float().cuda(non_blocking=True)

            user_vecs = model.user_encoder.infer_user_vec(log_vecs, log_mask).to(torch.device("cpu")).detach().numpy()

            user_embedding.extend(user_vecs)

    user_embedding = np.array(user_embedding)
    return user_embedding

if __name__ == '__main__':
    utils.setuplogging()
    args = parse_args()
    hnswlib_batch_size = 5000
    mode = 'ip' #cosine
    modelname = args.savename  #'auto_neg4'
    data_dir = args.root_data_dir
    ckpt_path = os.path.join(args.model_dir, args.load_ckpt_name)  #'/Feeds-nfs/data/t-shxiao/model/neg4_auto-epoch-1.pt'

    # if not os.path.exists('/Feeds-nfs/data/t-shxiao/embedding/{}'.format(modelname)):
    #     os.mkdir('/Feeds-nfs/data/t-shxiao/embedding/{}'.format(modelname))
    #     os.mkdir('/Feeds-nfs/data/t-shxiao/embedding/{}/user_embedding'.format(modelname))

    # new_embed = np.load('/Feeds-nfs/data/t-shxiao/embedding/fix_batch/item_embedding.npy')
    model,all_item_embedding = get_embed(args,ckpt_path,data_dir)
    # model = get_embed(args)
    # np.save('/Feeds-nfs/data/t-shxiao/embedding/{}/item_embedding'.format(modelname),new_embed)

    user_batch_size = 1

    res = np.array([0.0] * 5)
    print(all_item_embedding.shape)
    get_similarity(all_item_embedding)

    for date in range(1, 32):
        day_item = get_day_item(date, data_dir)
        item_num = len(day_item)

        item_embedding = all_item_embedding[day_item]
        #
        p = CreatIndex(5000, item_embedding, day_item, mode)
        # p.save_index("embedding/unium_yuting/index/8_{}_query_{}.bin".format(date, mode))
        # p = hnswlib.Index(space=mode, dim=np.shape(item_embedding)[-1])
        # p.load_index("embedding/unium_yuting/index/8_{}_query_{}.bin".format(date, mode), max_elements=len(item_embedding))
        p.set_ef(1500)

        history, positems = hist_pos(date, data_dir)
        history,positems = history[:5000],positems[:5000]
        user_embedding = CreatUserEmbed(history, all_item_embedding, user_batch_size, model)
        user_num = len(positems)
        print(user_num, item_num)

        day_res = get_result(user_embedding, positems, p, '', hnswlib_batch_size)
        res += day_res
        print(day_res)
        # fres.write(info)
        # fres.write('\n')
        # fres.flush()
        # print(res)

    res = res / 31
    # fres.write(str(list(res)))
    print(res)

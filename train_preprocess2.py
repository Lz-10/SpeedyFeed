from six.moves.urllib.parse import urlparse
from collections import Counter
import tensorflow as tf
from tqdm import tqdm
import pickle
import os
from nltk.tokenize import sent_tokenize
import gc


def get_domain(url):
    domain = urlparse(url).netloc
    return domain

def save_data(args,news_feature, categories_id, domains_id, subcategories_id,mode):
    process_news = {}
    process_news['news_feature'] = news_feature
    process_news['categories_id'] = categories_id
    process_news['subcategories_id'] = subcategories_id
    process_news['domains_id'] = domains_id


    if not os.path.exists(os.path.join(args.root_data_dir,'news_pkl')):
        os.mkdir(os.path.join(args.root_data_dir,'news_pkl'))
    with open(os.path.join(args.root_data_dir,'news_pkl/news_data_body_{}.pkl'.format(mode)), 'wb') as f:
        pickle.dump(process_news, f)

def pad_to_fix_len(x, fix_length, padding_value=0):
    pad_x = x[-fix_length:] + [padding_value] * (fix_length - len(x))
    mask = [1] * min(fix_length, len(x)) + [0] * (fix_length - len(x))
    return pad_x, mask


def split_token(text,tokenizer,max_l=32):
    if text == '':
        smask = 0
    else:
        smask = 1
    text = text.lower()
    text = tokenizer(text, max_length=max_l, truncation=True)
    return smask,text['input_ids']

def read_news_buslm(news_path, args, tokenizer,mode):
    data_path = os.path.join(args.root_data_dir,'news_pkl/news_data_body_{}.pkl'.format(mode))
    if os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            process_news = pickle.load(f)
        return process_news['news_feature'],\
               process_news['categories_id'],process_news['subcategories_id'],\
               process_news['domains_id']

    categories_id = {}
    subcategories_id = {}
    domains_id = {}
    news_feature = {}
    with tf.io.gfile.GFile(news_path, "r") as f:
        for line in tqdm(f):
            splited = line.strip('\n').split('\t')
            doc_id, category, subcategory, title, abstract, body = splited
            if doc_id not in news_feature:
                tokens = []
                seg_mask = []
                element = []

                if 'title' in args.news_attributes:
                    smask,text = split_token(title,tokenizer,args.num_words_title)
                    tokens.append(text)
                    seg_mask.append(smask)


                if 'abstract' in args.news_attributes:
                    smask, text = split_token(abstract,tokenizer, args.num_words_title)
                    tokens.append(text)
                    seg_mask.append(smask)


                if 'body' in args.news_attributes:
                    sent_tokenize_list = sent_tokenize(body)[:args.body_seg_num]
                    sent_tokenize_list = sent_tokenize_list + ['']*(args.body_seg_num-len(sent_tokenize_list))
                    for text in sent_tokenize_list:
                        smask, text = split_token(text,tokenizer, args.num_words_title)
                        tokens.append(text)
                        seg_mask.append(smask)

                if 'category' in args.news_attributes:
                    if category not in categories_id:
                        categories_id[category] = len(categories_id)
                    element.append(categories_id[category])


                if 'subcategory' in args.news_attributes:
                    if subcategory not in subcategories_id:
                        subcategories_id[subcategory] = len(subcategories_id)
                    element.append(subcategories_id[subcategory])


                if 'domain' in args.news_attributes:
                    domain = get_domain(url)
                    if domain not in domains_id:
                        domains_id[domain] = len(domains_id)
                    element.append(domains_id[domain])

                news_feature[doc_id] = (tokens,seg_mask,element)


                if len(news_feature)%5000==0:
                    print('loading news:',len(news_feature))
                    # del category, subcategory, splited,text,body,title,abstract
                    # gc.collect()

    save_data(args,news_feature, categories_id, domains_id, subcategories_id,mode)
    return news_feature, categories_id, subcategories_id,domains_id,




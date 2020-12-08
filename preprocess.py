from six.moves.urllib.parse import urlparse
from collections import Counter
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import re
from utils import word_tokenize
import os
import pickle

def get_domain(url):
    domain = urlparse(url).netloc
    return domain

def save_data(args,news, news_index, category_dict, domain_dict, subcategory_dict,mode):
    process_news = {}
    process_news['news'] = news
    process_news['news_index'] = news_index
    if category_dict is not None:
        process_news['category_dict'] = category_dict
        process_news['domain_dict'] = domain_dict
        process_news['subcategory_dict'] = subcategory_dict
    with open(os.path.join(args.root_data_dir,'old_{}_news_data.pkl'.format(mode)), 'wb') as f:
        pickle.dump(process_news, f)


def read_news_bert(news_path, args, tokenizer, mode='train'):
    if os.path.exists(os.path.join(args.root_data_dir,'old_{}_news_data.pkl'.format(mode))):
        with open(os.path.join(args.root_data_dir,'old_{}_news_data.pkl'.format(mode)), 'rb') as f:
            process_news = pickle.load(f)
        if mode == 'train':
            return process_news['news'],process_news['news_index'],\
                   process_news['category_dict'],process_news['domain_dict'],\
                   process_news['subcategory_dict']
        elif mode == 'test':
            return process_news['news'], process_news['news_index']

    news = {}
    categories = []
    subcategories = []
    domains = []
    news_index = {}
    index = 1

    # title_length = []

    with tf.io.gfile.GFile(news_path, "r") as f:
        for line in tqdm(f):
            splited = line.strip('\n').split('\t')
            doc_id, category, subcategory, title, abstract = splited
            if doc_id not in news_index:
                news_index[doc_id] = index
                index += 1

                if 'title' in args.news_attributes:
                    title = title.lower()
                    title = tokenizer(title, max_length=args.num_words_title, \
                    pad_to_max_length=True, truncation=True)
                    # title_length.append(sum(title['attention_mask']))
                    # print(sum(title['attention_mask']))
                    # args.num_words_title
                else:
                    title = []

                if 'abstract' in args.news_attributes:
                    abstract = abstract.lower()
                    abstract = tokenizer(abstract, max_length=args.num_words_abstract, \
                    pad_to_max_length=True, truncation=True)
                else:
                    abstract = []

                if 'body' in args.news_attributes:
                    body = body.lower()
                    body = tokenizer(body, max_length=args.num_words_body, \
                    pad_to_max_length=True, truncation=True)
                else:
                    body = []

                if 'category' in args.news_attributes:
                    categories.append(category)
                else:
                    category = None

                if 'subcategory' in args.news_attributes:
                    subcategories.append(subcategory)
                else:
                    subcategory = None

                if 'domain' in args.news_attributes:
                    domain = get_domain(url)
                    domains.append(domain)
                else:
                    domain = None

                news[doc_id] = [title, abstract, body, category, domain, subcategory]


    if mode == 'train':
        categories = list(set(categories))
        category_dict = {}
        index = 1
        for x in categories:
            category_dict[x] = index
            index += 1

        subcategories = list(set(subcategories))
        subcategory_dict = {}
        index = 1
        for x in subcategories:
            subcategory_dict[x] = index
            index += 1

        domains = list(set(domains))
        domain_dict = {}
        index = 1
        for x in domains:
            domain_dict[x] = index
            index += 1
        save_data(args, news, news_index, category_dict, domain_dict, subcategory_dict, mode)
        return news, news_index, category_dict, domain_dict, subcategory_dict
    elif mode == 'test':
        save_data(args, news, news_index, None, None, None, mode)
        return news, news_index
    else:
        assert False, 'Wrong mode!'



def infer_read_news_bert(news_path, args, tokenizer, mode='train'):
    news = {}
    categories = []
    subcategories = []
    domains = []
    news_index = {}

    with tf.io.gfile.GFile(news_path, "r") as f:
        for line in tqdm(f):
            splited = line.strip('\n').split('\t')
            index, doc_id, title = splited
            if doc_id not in news_index:
                news_index[doc_id] = int(index)

                if 'title' in args.news_attributes:
                    title = title.lower()
                    title = tokenizer(title, max_length=args.num_words_title, \
                                      pad_to_max_length=True, truncation=True)
                else:
                    title = []

                if 'abstract' in args.news_attributes:
                    abstract = abstract.lower()
                    abstract = tokenizer(abstract, max_length=args.num_words_abstract, \
                                         pad_to_max_length=True, truncation=True)
                else:
                    abstract = []

                if 'body' in args.news_attributes:
                    body = body.lower()
                    body = tokenizer(body, max_length=args.num_words_body, \
                                     pad_to_max_length=True, truncation=True)
                else:
                    body = []

                if 'category' in args.news_attributes:
                    categories.append(category)
                else:
                    category = None

                if 'subcategory' in args.news_attributes:
                    subcategories.append(subcategory)
                else:
                    subcategory = None

                if 'domain' in args.news_attributes:
                    domain = get_domain(url)
                    domains.append(domain)
                else:
                    domain = None

                news[doc_id] = [title, abstract, body, category, domain, subcategory]

            # if len(news) == 1000:
            #     break

    if mode == 'train':
        categories = list(set(categories))
        category_dict = {}
        index = 1
        for x in categories:
            category_dict[x] = index
            index += 1

        subcategories = list(set(subcategories))
        subcategory_dict = {}
        index = 1
        for x in subcategories:
            subcategory_dict[x] = index
            index += 1

        domains = list(set(domains))
        domain_dict = {}
        index = 1
        for x in domains:
            domain_dict[x] = index
            index += 1

        return news, news_index, category_dict, domain_dict, subcategory_dict
    elif mode == 'test':
        return news, news_index
    else:
        assert False, 'Wrong mode!'


def get_doc_input_bert(news, news_index, category_dict, domain_dict, subcategory_dict, args):
    news_num = len(news) + 1
    if 'title' in args.news_attributes:
        news_title = np.zeros((news_num, args.num_words_title), dtype='int32')
        news_title_type = np.zeros((news_num, args.num_words_title), dtype='int32')
        news_title_attmask = np.zeros((news_num, args.num_words_title), dtype='int32')
    else:
        news_title = None
        news_title_type = None
        news_title_attmask = None

    if 'abstract' in args.news_attributes:
        news_abstract = np.zeros((news_num, args.num_words_abstract), dtype='int32')
        news_abstract_type = np.zeros((news_num, args.num_words_abstract), dtype='int32')
        news_abstract_attmask = np.zeros((news_num, args.num_words_abstract), dtype='int32')
    else:
        news_abstract = None
        news_abstract_type = None
        news_abstract_attmask = None

    if 'body' in args.news_attributes:
        news_body = np.zeros((news_num, args.num_words_body), dtype='int32')
        news_body_type = np.zeros((news_num, args.num_words_body), dtype='int32')
        news_body_attmask = np.zeros((news_num, args.num_words_body), dtype='int32')
    else:
        news_body = None
        news_body_type = None
        news_body_attmask = None

    if 'category' in args.news_attributes:
        news_category = np.zeros((news_num, 1), dtype='int32')
    else:
        news_category = None

    if 'domain' in args.news_attributes:
        news_domain = np.zeros((news_num, 1), dtype='int32')
    else:
        news_domain = None

    if 'subcategory' in args.news_attributes:
        news_subcategory = np.zeros((news_num, 1), dtype='int32')
    else:
        news_subcategory = None

    for key in tqdm(news):
        title, abstract, body, category, domain, subcategory = news[key]
        doc_index = news_index[key]

        if 'title' in args.news_attributes:
            news_title[doc_index] = title['input_ids']
            news_title_type[doc_index] = title['token_type_ids']
            news_title_attmask[doc_index] = title['attention_mask']

        if 'abstract' in args.news_attributes:
            news_abstract[doc_index] = abstract['input_ids']
            news_abstract_type[doc_index] = abstract['token_type_ids']
            news_abstract_attmask[doc_index] = abstract['attention_mask']

        if 'body' in args.news_attributes:
            news_body[doc_index] = body['input_ids']
            news_body_type[doc_index] = body['token_type_ids']
            news_body_attmask[doc_index] = body['attention_mask']

        if 'category' in args.news_attributes:
            news_category[doc_index, 0] = category_dict[category] if category in category_dict else 0

        if 'subcategory' in args.news_attributes:
            news_subcategory[doc_index, 0] = subcategory_dict[subcategory] if subcategory in subcategory_dict else 0

        if 'domain' in args.news_attributes:
            news_domain[doc_index, 0] = domain_dict[domain] if domain in domain_dict else 0

    return news_title, news_title_attmask, \
           news_abstract, news_abstract_attmask, \
           news_body, news_body_attmask, \
           news_category, news_domain, news_subcategory


if __name__ == "__main__":
    from parameters import parse_args

    args = parse_args()
    args.news_attributes = ['title', 'body', 'category', 'subcategory', 'domain']
    news, news_index, category_dict, word_dict, domain_dict, subcategory_dict = read_news_bert(
        "/Feeds-nfs/data/v-jinyi/MSNPipeline/MSNLatency1/en-us/2020-07-31/docs.tsv",
        args)
    news_title, news_abstract, news_body, news_category, news_domain, news_subcategory = get_doc_input_bert(
        news, news_index, category_dict, word_dict, domain_dict, subcategory_dict, args)

    print(category_dict)
    print(news_category)
from six.moves.urllib.parse import urlparse
from collections import Counter
import tensorflow as tf
from tqdm import tqdm
import pickle
import os

def get_domain(url):
    domain = urlparse(url).netloc
    return domain

def save_data(args,news_combine, news_length, categories_id, domains_id, subcategories_id):
    process_news = {}
    process_news['news_combine'] = news_combine
    process_news['news_length'] = news_length
    process_news['categories_id'] = categories_id
    process_news['domains_id'] = domains_id
    process_news['subcategories_id'] = subcategories_id

    with open(os.path.join(args.root_data_dir,'news_data.pkl'), 'wb') as f:
        pickle.dump(process_news, f)


def read_news_bert_nopadding(news_path, args, tokenizer):
    if os.path.exists(os.path.join(args.root_data_dir,'news_data.pkl')):
        with open(os.path.join(args.root_data_dir,'news_data.pkl'), 'rb') as f:
            process_news = pickle.load(f)
        return process_news['news_combine'],process_news['news_length'],\
               process_news['categories_id'],process_news['domains_id'],\
               process_news['subcategories_id']

    categories_id = {}
    subcategories_id = {}
    domains_id = {}
    news_combine = {}
    news_length = {}


    with tf.io.gfile.GFile(news_path, "r") as f:
        for line in tqdm(f):
            splited = line.strip('\n').split('\t')
            doc_id, category, subcategory, title, abstract = splited
            if doc_id not in news_combine:
                temp_feature = []

                if 'title' in args.news_attributes:
                    title = title.lower()
                    title = tokenizer(title, max_length=args.num_words_title, truncation=True)
                    temp_feature.extend(title['input_ids'])


                if 'abstract' in args.news_attributes:
                    abstract = abstract.lower()
                    abstract = tokenizer(abstract, max_length=args.num_words_abstract, truncation=True)
                    temp_feature.extend(abstract['input_ids'])


                if 'body' in args.news_attributes:
                    body = body.lower()
                    body = tokenizer(body, max_length=args.num_words_body, truncation=True)
                    temp_feature.extend(body['input_ids'])


                if 'category' in args.news_attributes:
                    if category not in categories_id:
                        categories_id[category] = len(categories_id)
                    temp_feature.append(categories_id[category])

                if 'subcategory' in args.news_attributes:
                    if subcategory not in subcategories_id:
                        subcategories_id[subcategory] = len(subcategories_id)
                    temp_feature.append(subcategories_id[subcategory])

                if 'domain' in args.news_attributes:
                    domain = get_domain(url)
                    if domain not in domains_id:
                        domains_id[domain] = len(domains_id)
                    temp_feature.append(domains_id[domain])


                news_combine[doc_id] = temp_feature
                news_length[doc_id] = len(temp_feature)

    save_data(args,news_combine, news_length, categories_id, domains_id, subcategories_id)
    return news_combine, news_length, categories_id, domains_id, subcategories_id



if __name__ == "__main__":
    from parameters import parse_args

    args = parse_args()
    args.news_attributes = ['title', 'body', 'category', 'subcategory', 'domain']
    news, news_index, category_dict, word_dict, domain_dict, subcategory_dict = read_news_bert_nopadding(
        "/Feeds-nfs/data/v-jinyi/MSNPipeline/MSNLatency1/en-us/2020-07-31/docs.tsv",
        args)
    news_title, news_abstract, news_body, news_category, news_domain, news_subcategory = get_doc_input_bert(
        news, news_index, category_dict, word_dict, domain_dict, subcategory_dict, args)

    print(category_dict)
    print(news_category)
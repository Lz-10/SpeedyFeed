import sys
import traceback
import logging
import time
import random
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch.utils.data import IterableDataset
from streaming_queue import StreamSampler
from streaming import StreamSamplerTest
import utils
import math



class DataLoaderTrain(IterableDataset):
    def __init__(self,
                 data_dir,
                 args,
                 local_rank,
                 cache,
                 news_idx_incache,
                 prefetch_step,
                 world_size,
                 worker_rank,
                 cuda_device_idx,
                 news_combined,
                 news_length,
                 enable_prefetch=True,
                 enable_shuffle=False,
                 enable_stream_queue=False,
                 enable_gpu=True):
        self.data_dir = data_dir
        self.filename_pat = args.filename_pat

        self.npratio = args.npratio
        self.user_log_length = args.user_log_length
        self.batch_size = args.batch_size

        self.worker_rank = worker_rank
        self.world_size = world_size
        self.cuda_device_idx = cuda_device_idx
        # data loader only cares about the config after tokenization.
        self.sampler = None

        self.enable_prefetch = enable_prefetch
        self.enable_shuffle = enable_shuffle
        self.enable_gpu = enable_gpu
        self.epoch = -1

        self.news_combined = news_combined
        self.news_length = news_length

        self.args = args
        self.feature_max_len = self.args.num_words_title #+self.args.num_words_abstract+...
        self.enable_stream_queue = enable_stream_queue

        self.global_step = 0
        self.local_rank = local_rank
        self.cache=cache
        self.news_idx_incache=news_idx_incache
        self.prefetch_step=prefetch_step

    def drop_encoder_prob(self,prob, step):
        return prob - prob*math.exp(-100*step/self.args.schedule_step)

    def _produce(self):
        # need to reset cuda device in produce thread.
        blocks = [[]for x in range(self.args.block_num)]   #hitory_id, neg_id
        block_sets = [set() for x in range(self.args.block_num)]
        block_cache_sets = [set() for x in range(self.args.block_num)]
        block_max_length = [0 for x in range(self.args.block_num) ]

        block_space = [(self.feature_max_len // self.args.block_num)*i for i in range(self.args.block_num)]

        self.use_cache = False

        if self.enable_gpu:
            torch.cuda.set_device(self.cuda_device_idx)
        try:
            self.epoch += 1
            self.sampler = StreamSampler(
                data_dir=self.data_dir,
                filename_pat=self.filename_pat,
                batch_size=self.batch_size,
                user_log_length = self.user_log_length,
                worker_rank=self.worker_rank,
                world_size=self.world_size,
                enable_shuffle=self.enable_shuffle,
                shuffle_seed=self.epoch,  # epoch id as shuffle random seed
            )
            # t0 = time.time()
            if self.enable_stream_queue:
                self.sampler_batch = self.sampler
            else:
                self.sampler_batch = self.sampler._generate_batch(self.batch_size)
            for one_user in self.sampler_batch:
                if self.stopped:
                    break

                news_set, history, negs = self._process(one_user)
                cache_set, encode_set = self.split_news_set(news_set, self.use_cache)
                # print('split------------------------',len(news_set),len(cache_set),len(encode_set))
                if len(encode_set) > 0:
                    max_len = max([self.news_length[nid] if nid in self.news_length else 0 for nid in encode_set])
                else:
                    max_len = 0


                for i in range(self.args.block_num-1,-1,-1):
                    if max_len > block_space[i]:
                        if max(max_len,block_max_length[i])*len(block_sets[i] | encode_set) > self.batch_size:
                            address_cache,update_cache,batch = self.gen_batch(block_sets[i],block_cache_sets[i],blocks[i],block_max_length[i],self.global_step)
                            self.outputs.put((block_max_length[i],address_cache,update_cache,batch))
                            self.aval_count += 1
                            self.global_step += 1
                            self.prefetch_step[self.local_rank] += 1

                            block_sets[i] = set();block_cache_sets[i] = set();blocks[i]=[];block_max_length[i]=0

                            self.synchronization()
                            self.update_use_cache()

                        block_max_length[i] = max(max_len, block_max_length[i])
                        block_sets[i] = block_sets[i] | encode_set
                        block_cache_sets[i] = block_cache_sets[i] | cache_set
                        blocks[i].append((history,negs))
                        break

        except:
            traceback.print_exc(file=sys.stdout)
            self.pool.shutdown(wait=False)
            raise

    def synchronization(self):
        while sum(self.prefetch_step) != self.prefetch_step[self.local_rank] * self.world_size:
            pass

    def update_use_cache(self):
        if random.random() < self.drop_encoder_prob(self.args.drop_encoder_ratio, self.global_step):
            self.use_cache = True
        else:
            self.use_cache = False

    def split_news_set(self,news_set,use_cache):
        if use_cache:
            cache_set = set()
            encode_set = set()
            for n in news_set:
                if n == 'MISS':
                    continue
                if self.global_step - self.news_idx_incache[n][1] < self.args.max_step_in_cache:   #should be <=
                    cache_set.add(n)
                else:
                    encode_set.add(n)

            return cache_set,encode_set
        else:
            news_set.discard('MISS')
            return set(), news_set


    def gen_batch(self,encode_set,cache_set,data,max_len,global_step):
        batch_news_feature = []
        news_index = {'MISS': 0}
        idx = 1

        #get embedding from cache
        if len(cache_set)==0:
            address_cache = None
        else:
            address_cache = []
            for n in cache_set:
                address_cache.append(self.news_idx_incache[n][0])
                news_index[n] = idx
                idx += 1
            address_cache = np.array(address_cache)

        update_cache = [] #list(range(100))
        for n in encode_set:
            news_index[n] = idx
            idx += 1
            n_feature = self.news_combined[n] + [0] * (max_len - len(self.news_combined[n]))
            mask = [1] * len(self.news_combined[n]) + [0] * (max_len - len(self.news_combined[n]))
            n_feature.extend(mask)
            batch_news_feature.append(n_feature)

            # update cache
            update_cache.append(self.news_idx_incache[n][0])
            self.news_idx_incache[n] = [self.news_idx_incache[n][0],global_step]

        batch_news_feature = np.array(batch_news_feature)

        batch_hist = []
        batch_negs = []
        batch_mask = []
        max_hist_len = max([len(x[0]) for x in data])
        for history,negs in data:
            history = self.trans_to_nindex(history,news_index)
            history,mask = self.pad_to_fix_len(history,max_hist_len)
            batch_hist.append(history)
            batch_mask.append(mask)

            negs = [self.trans_to_nindex(n,news_index) for n in negs]
            negs = self.pad_to_fix_len_neg(negs,max_hist_len-1)
            batch_negs.append(negs)

        if self.enable_gpu:
            batch_news_feature = torch.LongTensor(batch_news_feature).cuda()
            batch_hist = torch.LongTensor(batch_hist).cuda()
            batch_negs = torch.LongTensor(batch_negs).cuda()
            batch_mask = torch.FloatTensor(batch_mask).cuda()
        else:
            batch_news_feature = torch.LongTensor(batch_news_feature)
            batch_hist = torch.LongTensor(batch_hist)
            batch_negs = torch.LongTensor(batch_negs)
            batch_mask = torch.FloatTensor(batch_mask)

        return address_cache,np.array(update_cache),(batch_news_feature,batch_hist,batch_mask,batch_negs)


    def trans_to_nindex(self, nids,news_index):
        return [news_index[i] for i in nids]

    def pad_to_fix_len(self, x, fix_length, padding_value=0):
        pad_x = x[-fix_length:] + [padding_value] * (fix_length - len(x))
        mask = [1] * min(fix_length, len(x)) + [0] * (fix_length - len(x))
        return pad_x, mask

    def pad_to_fix_len_neg(self, x, fix_length, padding_value=0):
        pad_x = x[-fix_length:] + [[padding_value] * self.npratio] * (fix_length - len(x))
        return pad_x

    def _process(self, line):
        clicked = []
        negnews = []
        u_set = []

        # print(len(u.decode(encoding='utf-8').strip().split('\t')))
        uid, click_length, sessions = line.strip().split('\t')
        for sess in sessions.split('|'):
            # print(sess)
            pos, neg = sess.split('&')
            pos = [p if p in self.news_length else 'MISS' for p in pos.split(';')]
            clicked.extend(pos)

            neg = neg.split(';')
            for p in pos:
                if len(neg) < self.npratio:
                    neg = neg*(int(self.npratio/len(neg))+1)
                sample_neg = [n if n in self.news_length else 'MISS' for n in random.sample(neg, self.npratio)]
                negnews.append(sample_neg)

        clicked = clicked[-self.user_log_length:]
        negnews = negnews[-(self.user_log_length-1):]

        for p in clicked:
            u_set.append(p)
        for ns in negnews:
            u_set.extend(ns)

        u_set = set(u_set)
        return u_set,clicked,negnews

    def start(self):
        self.epoch += 1
        self.sampler = StreamSampler(
            data_dir=self.data_dir,  # !!! the train data is not in the root_data_dir path
            filename_pat=self.filename_pat,
            batch_size=self.batch_size,
            user_log_length = self.user_log_length,
            worker_rank=self.worker_rank,
            world_size=self.world_size,
            enable_shuffle=self.enable_shuffle,
            shuffle_seed=self.epoch,  # epoch id as shuffle random seed
        )
        self.sampler.__iter__()

    def start_async(self):
        self.aval_count = 0
        self.stopped = False
        self.outputs = Queue(10)
        self.pool = ThreadPoolExecutor(1)
        self.pool.submit(self._produce)

    def __iter__(self):
        """Implement IterableDataset method to provide data iterator."""
        logging.info("DataLoader __iter__()")
        if self.enable_prefetch:
            self.join()
            self.start_async()
        else:
            self.start()
        return self

    def __next__(self):
        if self.enable_prefetch:
            if self.sampler and self.aval_count == 0 and self.sampler.end == True:
                raise StopIteration
            next_batch = self.outputs.get()
            self.outputs.task_done()
            self.aval_count -= 1
        else:
            next_batch = self._process(self.sampler.__next__())
        return next_batch

    def join(self):
        self.stopped = True
        if self.sampler:
            if self.enable_prefetch:
                while self.outputs.qsize() > 0:
                    self.outputs.get()
                    self.outputs.task_done()
                self.outputs.join()
                self.pool.shutdown(wait=True)
                logging.info("shut down pool.")
            self.sampler = None


class DataLoaderTest():
    def __init__(self,
                 data_dir,
                 filename_pat,
                 args,
                 world_size,
                 worker_rank,
                 cuda_device_idx,
                 news_index,
                 news_scoring,
                 news_bias_scoring=None,
                 enable_prefetch=True,
                 enable_shuffle=False,
                 enable_gpu=True):
        self.data_dir = data_dir
        self.filename_pat = filename_pat

        self.npratio = args.npratio
        self.user_log_length = args.user_log_length
        self.batch_size = args.batch_size

        self.worker_rank = worker_rank
        self.world_size = world_size
        self.cuda_device_idx = cuda_device_idx
        # data loader only cares about the config after tokenization.
        self.sampler = None

        self.enable_prefetch = enable_prefetch
        self.enable_shuffle = enable_shuffle
        self.enable_gpu = enable_gpu
        self.epoch = -1

        self.news_scoring = news_scoring
        self.news_bias_scoring = news_bias_scoring
        self.news_index = news_index

    def start(self):
        self.epoch += 1
        self.sampler = StreamSamplerTest(
            data_dir=self.data_dir,
            filename_pat=self.filename_pat,
            batch_size=self.batch_size,
            worker_rank=self.worker_rank,
            world_size=self.world_size,
            enable_shuffle=self.enable_shuffle,
            shuffle_seed=self.epoch,  # epoch id as shuffle random seed
        )
        self.sampler.__iter__()

    def start_async(self):
        self.aval_count = 0
        self.stopped = False
        self.outputs = Queue(10)
        self.pool = ThreadPoolExecutor(1)
        self.pool.submit(self._produce)

    def trans_to_nindex(self, nids):
        return [self.news_index[i] if i in self.news_index else 0 for i in nids]

    def pad_to_fix_len(self, x, fix_length, padding_value=0):
        pad_x = x[-fix_length:] + [padding_value] * (fix_length - len(x))
        mask = [1] * min(fix_length, len(x)) + [0] * (fix_length - len(x))
        return pad_x, mask

    def pad_to_fix_len_neg(self, x, fix_length, padding_value=0):
        pad_x = x[-fix_length:] + [[padding_value] * self.npratio] * (fix_length - len(x))
        return pad_x

    def _produce(self):
        # need to reset cuda device in produce thread.
        if self.enable_gpu:
            torch.cuda.set_device(self.cuda_device_idx)
        try:
            self.epoch += 1
            self.sampler = StreamSamplerTest(
                data_dir=self.data_dir,
                filename_pat=self.filename_pat,
                batch_size=self.batch_size,
                worker_rank=self.worker_rank,
                world_size=self.world_size,
                enable_shuffle=self.enable_shuffle,
                shuffle_seed=self.epoch,  # epoch id as shuffle random seed
            )
            # t0 = time.time()
            for batch in self.sampler:
                if self.stopped:
                    break
                context = self._process(batch)
                self.outputs.put(context)
                self.aval_count += 1
                # logging.info(f"_produce cost:{time.time()-t0}")
                # t0 = time.time()
        except:
            traceback.print_exc(file=sys.stdout)
            self.pool.shutdown(wait=False)
            raise

    def _process(self, batch):
        batch_size = len(batch)
        # batch = [x.decode(encoding="utf-8").split("\t") for x in batch]
        user_feature_batch, log_mask_batch, news_feature_batch, label_batch = [], [], [], []


        for line in batch:
            uid, click_docs, pos, neg = line.decode(encoding="utf-8").split('\t')

            click_docs = click_docs.split(';')
            click_docs, log_mask = self.pad_to_fix_len(self.trans_to_nindex(click_docs),
                                                       self.user_log_length)
            user_feature = self.news_scoring[click_docs]

            sess_pos = pos.split(';')
            sess_neg = neg.split(';')
            sess_pos = self.trans_to_nindex(sess_pos)
            sess_neg = self.trans_to_nindex(sess_neg)

            sample_news = sess_pos + sess_neg
            labels = [1] * len(sess_pos) + [0] * len(sess_neg)

            news_feature = self.news_scoring[sample_news]

            user_feature_batch.append(user_feature)
            log_mask_batch.append(log_mask)
            news_feature_batch.append(news_feature)
            label_batch.append(np.array(labels))

        if self.enable_gpu:
            user_feature_batch = torch.FloatTensor(user_feature_batch).cuda()
            log_mask_batch = torch.FloatTensor(log_mask_batch).cuda()

        else:
            user_feature_batch = torch.FloatTensor(user_feature_batch)
            log_mask_batch = torch.FloatTensor(log_mask_batch)

        return user_feature_batch, log_mask_batch, news_feature_batch, label_batch

    def __iter__(self):
        """Implement IterableDataset method to provide data iterator."""
        logging.info("DataLoader __iter__()")
        if self.enable_prefetch:
            self.join()
            self.start_async()
        else:
            self.start()
        return self

    def __next__(self):
        if self.sampler and self.sampler.reach_end() and self.aval_count == 0:
            raise StopIteration
        if self.enable_prefetch:
            next_batch = self.outputs.get()
            self.outputs.task_done()
            self.aval_count -= 1
        else:
            next_batch = self._process(self.sampler.__next__())
        return next_batch

    def join(self):
        self.stopped = True
        if self.sampler:
            if self.enable_prefetch:
                while self.outputs.qsize() > 0:
                    self.outputs.get()
                    self.outputs.task_done()
                self.outputs.join()
                self.pool.shutdown(wait=True)
                logging.info("shut down pool.")
            self.sampler = None


if __name__ == "__main__":
    import logging
    import utils
    utils.setuplogging()

    from train_preprocess import read_news_bert_nopadding
    from parameters import parse_args
    from tnlrv3.modeling import TuringNLRv3ForSequenceClassification
    from tnlrv3.configuration_tnlrv3 import TuringNLRv3Config
    from tnlrv3.tokenization_tnlrv3 import TuringNLRv3Tokenizer
    import os

    MODEL_CLASSES = {
        'tnlrv3': (TuringNLRv3Config, TuringNLRv3ForSequenceClassification, TuringNLRv3Tokenizer),
    }
    args = parse_args()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)
    bert_model = model_class.from_pretrained(args.model_name_or_path,
                                             from_tf=bool('.ckpt' in args.model_name_or_path),
                                             config=config)

    news_combined, news_length, category_dict, domain_dict, subcategory_dict = read_news_bert_nopadding(
        os.path.join(args.root_data_dir, f'docs.tsv'),
        args, tokenizer)

    a = max([v for v in news_length.values() ])
    # hvd_size, hvd_rank, hvd_local_rank = utils.init_hvd_cuda(True)

    args.batch_size = 27000
    args.block_num = 3
    dataloader = DataLoaderTrain(
        news_combined=news_combined,
        news_length = news_length,
        data_dir='/data/t-shxiao/test/cosmos-speedup-turing/rec_bert/data/data/autoregressive',
        filename_pat='ProtoBuf_59.tsv',   #args.filename_pat,
        args=args,
        world_size=1 ,#hvd_size,
        worker_rank=0, #hvd_rank,
        cuda_device_idx=0,#hvd_local_rank,
        enable_prefetch=True,
        enable_shuffle=True,
        enable_gpu=args.enable_gpu
    )

    # print(os.listdir('/data/t-shxiao/test/cosmos-speedup-turing/rec_bert/data/data/autoregressive'))
    usernum = 0
    st = time.time()
    count = 0
    for b in dataloader:
        for b in dataloader:
            # print(b[0].shape[0])
            # try:
            #     print(dataloader.outputs.qsize(),dataloader.sampler.outputs.qsize())
            # except:
            #     print(dataloader.outputs.qsize())
            # print(dataloader.hit)
            usernum += b[1][3].shape[0]
            count += 1
            if count % 100 == 0:
                # print(usernum, count,dataloader.overlap,dataloader.hit,dataloader.new_num,dataloader.hit/dataloader.new_num)
                print(count, usernum, time.time() - st)
        print(usernum, count, usernum, time.time() - st)
        # usernum += b[0].shape[0]
        # print(time.time()-st,b[0].shape,usernum,dataloader.outputs.qsize())

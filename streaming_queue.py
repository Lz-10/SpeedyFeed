import os
import sys
import traceback
import logging
import fnmatch
import random
import numpy as np
import tensorflow as tf
from queue import Queue
import utils
from concurrent.futures import ThreadPoolExecutor
from streaming import get_files,get_worker_files


class StreamSampler:
    def __init__(
        self,
        data_dir,
        filename_pat,
        batch_size,
        user_log_length,
        worker_rank,
        world_size,
        enable_shuffle=False,
        shuffle_seed=0,
    ):
        self.batch_size = batch_size
        self.user_log_length = user_log_length
        self.data_paths = get_worker_files(
            data_dir,
            worker_rank,
            world_size,
            filename_pat,
            shuffle=enable_shuffle,
            seed=shuffle_seed,
        )
        self.end = False
        self.sampler = None


    def start_async(self):
        self.aval_count = 0
        self.end = False
        self.outputs = Queue(1000)
        self.pool = ThreadPoolExecutor(1)
        self.pool.submit(self._produce)

    def _produce(self):
        try:
            self.sampler = self._generate_batch(self.batch_size)
            for batch in self.sampler:
                if self.end:
                    break
                self.outputs.put(batch)
                self.aval_count += 1
        except:
            traceback.print_exc(file=sys.stdout)
            self.pool.shutdown(wait=False)
            raise

    def _generate_batch(self,batch_size):
        for path in self.data_paths:
            fdata = open(path, 'r', encoding='utf-8')
            line = fdata.readline()
            while line:
                yield line
                line = fdata.readline()
        self.end = True

    def __iter__(self):
        self.join()
        self.start_async()
        return self

    def __next__(self):
        if self.sampler and  self.aval_count == 0 and self.end == True:
            raise StopIteration
        next_batch = self.outputs.get()
        self.outputs.task_done()
        self.aval_count -= 1
        return next_batch

    def join(self):
        self.end = True
        if self.sampler:
            while self.outputs.qsize() > 0:
                self.outputs.get()
                self.outputs.task_done()
            self.outputs.join()
            self.pool.shutdown(wait=True)
            logging.info("shut down pool.")
        self.sampler = None


if __name__ == "__main__":
    print("start")
    # sampler = StreamSamplerTest(
    #     "/Feeds-nfs/data/v-jinyi/MSNPipeline/MSNLatency1/en-us/2020-07-31/",
    #     "ProtoBuf_000*.tsv", 32, 0, 1)

    sampler = StreamSampler(
        data_dir='/data/t-shxiao/test/cosmos-speedup-turing/rec_bert/data/data/autoregressive',
        # !!! the train data is not in the root_data_dir path
        filename_pat="ProtoBuf_*.tsv",
        batch_size=10000,
        worker_rank=0,
        world_size=1,
        user_log_length=100,
        enable_shuffle=True,
        shuffle_seed=0,  # epoch id as shuffle random seed
    )

    count = 0
    for batch in sampler:
        count+=1
        print(batch)

        # if count == 5:
        #     break

    # f = open('/data/t-shxiao/test/cosmos-speedup-turing/rec_bert/data/2020-07-31/ProtoBuf_0002.tsv', 'r',
    #          encoding='utf-8')
    # for line in f.readlines():
    #     print(line)
    #     count += 1
    #     if count == 10:
    #         break


    #
    # f = open("/Feeds-nfs/data/v-jinyi/MSNPipeline/MSNLatency1/en-us/2020-07-31/ProtoBuf_0001.tsv",'r',encoding='utf-8')
    # for line in f.readlines():
    #     print('---------------------------------------------')
    #     line = line.split('\t')
    #     # print(line[6])
    #     for i in range(len(line)):
    #         print(i,line[i])
        # break


    # import time
    # for i in sampler:
    #     logging.info("sampler")
    #     logging.info(i)
    #     time.sleep(5)

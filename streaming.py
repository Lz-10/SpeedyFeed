import os
import logging
import fnmatch
import random
import numpy as np
import tensorflow as tf

import utils


def get_files(dirname, filename_pat="*", recursive=False):
    if not tf.io.gfile.exists(dirname):
        return None
    files = []
    for x in tf.io.gfile.listdir(dirname):
        path = os.path.join(dirname, x)
        if tf.io.gfile.isdir(path):
            if recursive:
                files.extend(get_files(path, filename_pat))
        elif fnmatch.fnmatch(x, filename_pat):
            files.append(path)
    return files


def get_worker_files(dirname,
                     worker_rank,
                     world_size,
                     filename_pat="*",
                     shuffle=False,
                     seed=0):
    """Get file paths belong to one worker."""
    all_files = get_files(dirname, filename_pat)
    all_files.sort()
    if shuffle:
        # g_mutex = threading.Lock()
        # g_mutex.acquire()
        random.seed(seed)
        random.shuffle(all_files)
        # g_mutex.release()
    files = []
    for i in range(worker_rank, len(all_files), world_size):
        files.append(all_files[i])
    # logging.info(
    #     f"worker_rank:{worker_rank}, world_size:{world_size}, shuffle:{shuffle}, seed:{seed}, directory:{dirname}, files:{files}"
    # )
    return files


class StreamReader:
    def __init__(self, data_paths, batch_size):
        tf.config.experimental.set_visible_devices([], device_type="GPU")
        # logging.info(f"visible_devices:{tf.config.experimental.get_visible_devices()}")
        path_len = len(data_paths)
        # logging.info(f"[StreamReader] path_len:{path_len}, paths: {data_paths}")
        dataset = tf.data.Dataset.list_files(data_paths).interleave(
            lambda x: tf.data.TextLineDataset(x),
            cycle_length=path_len,
            block_length=batch_size,
            num_parallel_calls=min(path_len, 64),
        )

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)
        self.next_batch = dataset.make_one_shot_iterator().get_next()
        self.session = None

    def reset(self):
        # print(f"StreamReader reset(), {self.session}, pid:{threading.currentThread()}")
        if self.session:
            self.session.close()
        self.session = tf.Session()
        self.endofstream = False

    def get_next(self):
        try:
            ret = self.session.run(self.next_batch)
        except tf.errors.OutOfRangeError:
            self.endofstream = True
            return None
        return ret

    def reach_end(self):
        # print(f"StreamReader reach_end(), {self.endofstream}")
        return self.endofstream


class StreamSampler:
    def __init__(
        self,
        data_dir,
        filename_pat,
        batch_size,
        worker_rank,
        world_size,
        enable_shuffle=False,
        shuffle_seed=0,
    ):
        data_paths = get_worker_files(
            data_dir,
            worker_rank,
            world_size,
            filename_pat,
            shuffle=enable_shuffle,
            seed=shuffle_seed,
        )
        self.stream_reader = StreamReader(data_paths, batch_size)

    def __iter__(self):
        self.stream_reader.reset()
        return self

    def __next__(self):
        """Implement iterator interface."""
        # logging.info(f"[StreamSampler] __next__")
        next_batch = self.stream_reader.get_next()
        if not isinstance(next_batch, np.ndarray) and not isinstance(
                next_batch, tuple):
            raise StopIteration
        # print(next_batch.shape)
        return next_batch

    def reach_end(self):
        return self.stream_reader.reach_end()


class StreamReaderTest(StreamReader):
    def __init__(self, data_paths, batch_size):
        tf.config.experimental.set_visible_devices([], device_type="GPU")
        # logging.info(f"visible_devices:{tf.config.experimental.get_visible_devices()}")
        path_len = len(data_paths)
        # logging.info(f"[StreamReader] path_len:{path_len}, paths: {data_paths}")
        dataset = tf.data.Dataset.list_files(data_paths).interleave(
            lambda x: tf.data.TextLineDataset(x),
            cycle_length=path_len,
            block_length=128,
            num_parallel_calls=min(path_len, 64),
        )
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)
        self.next_batch = dataset.make_one_shot_iterator().get_next()
        self.session = None


class StreamSamplerTest(StreamSampler):
    def __init__(
        self,
        data_dir,
        filename_pat,
        batch_size,
        worker_rank,
        world_size,
        enable_shuffle=False,
        shuffle_seed=0,
    ):
        data_paths = get_worker_files(
            data_dir,
            worker_rank,
            world_size,
            filename_pat,
            shuffle=enable_shuffle,
            seed=shuffle_seed,
        )
        self.stream_reader = StreamReaderTest(data_paths, batch_size)


if __name__ == "__main__":
    utils.setuplogger()
    print("start")
    # sampler = StreamSamplerTest(
    #     "/Feeds-nfs/data/v-jinyi/MSNPipeline/MSNLatency1/en-us/2020-07-31/",
    #     "ProtoBuf_000*.tsv", 32, 0, 1)

    sampler = StreamSampler(
        data_dir='/data/t-shxiao/test/cosmos-speedup-turing/rec_bert/data/2020-07-31',
        # !!! the train data is not in the root_data_dir path
        filename_pat="ProtoBuf_000*.tsv",
        batch_size=3,
        worker_rank=0,
        world_size=1,
        enable_shuffle=True,
        shuffle_seed=0,  # epoch id as shuffle random seed
    )
    count = 0
    for batch in sampler:
        print(batch)
        count+=1
        if count == 5:
            break

    f = open('/data/t-shxiao/test/cosmos-speedup-turing/rec_bert/data/2020-07-31/ProtoBuf_0002.tsv', 'r',
             encoding='utf-8')
    for line in f.readlines():
        print(line)
        count += 1
        if count == 10:
            break







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

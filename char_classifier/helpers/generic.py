# coding=utf-8
import torch
import threading
import time
try:
    import queue
except ImportError:
    import Queue as queue


def batch_generator(x, y, batch_size, enable_cuda=False):
    sample_count = len(y)
    this_many_batches = sample_count // batch_size
    if sample_count % batch_size > 0:
        this_many_batches += 1

    while True:
        for batch_num in range(this_many_batches):
            # grab the chunk of samples in the current bucket
            batch_start = batch_num * batch_size
            batch_end = batch_start + batch_size
            if batch_start >= sample_count:
                continue
            batch_end = min(batch_end, sample_count)
            batch_x = x[batch_start: batch_end]
            batch_y = y[batch_start: batch_end]
            batch_x = batch_x.astype('float32') / 255.0
            batch_x = batch_x.reshape((batch_x.shape[0], 1) + batch_x.shape[1:])

            if enable_cuda:
                batch_x = torch.autograd.Variable(torch.from_numpy(batch_x).type(torch.FloatTensor).cuda())
                batch_y = torch.autograd.Variable(torch.from_numpy(batch_y).type(torch.LongTensor).cuda())
            else:
                batch_x = torch.autograd.Variable(torch.from_numpy(batch_x).type(torch.FloatTensor))
                batch_y = torch.autograd.Variable(torch.from_numpy(batch_y).type(torch.LongTensor))
            yield batch_x, batch_y


def generator_queue(generator, max_q_size=10, wait_time=0.05, nb_worker=1):
    '''Builds a threading queue out of a data generator.
    Used in `fit_generator`, `evaluate_generator`.
    '''
    q = queue.Queue()
    _stop = threading.Event()

    def data_generator_task():
        while not _stop.is_set():
            try:
                if q.qsize() < max_q_size:
                    try:
                        generator_output = next(generator)
                    except ValueError:
                        continue
                    q.put(generator_output)
                else:
                    time.sleep(wait_time)
            except Exception:
                _stop.set()
                raise

    generator_threads = [threading.Thread(target=data_generator_task)
                         for _ in range(nb_worker)]

    for thread in generator_threads:
        thread.daemon = True
        thread.start()
    return q, _stop

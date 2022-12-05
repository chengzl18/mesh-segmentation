import multiprocessing
import numpy as np
import functools
import time


def timed(func):
    @functools.wraps(func)
    def timed_wrapper(*args, **kwargs):
        print(f'{func.__name__}', end=' ')
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f'{end_time - start_time:.2f}s')
        return result

    return timed_wrapper


def equal_split(all_items, all_ids, n, split_manner):
    """将all_items, all_ids均分成块，每块的形式都是[(id,item),...]"""
    total = len(all_items)
    if all_ids is None:
        all_ids = list(range(total))
    assert split_manner in ['chunk', 'turn']
    if split_manner == 'chunk':
        indices = np.array_split(range(total), n)
    else:
        indices = [list(range(total)[i::n]) for i in range(n)]
    items = [[all_items[i] for i in inds] for inds in indices]  # (id, doc)
    return items


def single_run(func, docs):
    """用func逐一处理docs"""
    docs, pos = docs
    res = func(np.array(docs))
    return res


def parallel_run(func, all_docs, num_proc, split_manner='chunk'):  # all_doc是一个list, func对一个doc做处理
    """并行处理

    Args:
        func: 对一个doc做处理的函数
        all_docs: 所有需要被处理的doc
        num_proc: 进程数量
        split_manner: chunk/turn 分块分配/轮流分配

    Return:
        当func不是模型批量处理时，返回结果等价于 [func(doc) for doc in all_docs]
        当func是模型批量处理时，返回结果类似，只不过func是批量处理的
    """
    num_proc = min(num_proc, len(all_docs))
    split_docs = equal_split(all_docs, None, num_proc, split_manner)
    results = []
    with multiprocessing.Pool(num_proc) as p:
        pids = list(range(num_proc))
        assert len(pids) == len(split_docs)
        for single_res in p.imap(functools.partial(single_run, func), list(zip(split_docs, pids))):
            results.append(single_res)
    return np.concatenate(results, axis=0)

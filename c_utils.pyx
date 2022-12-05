# distutils: language = c++
import numpy
cimport numpy
ctypedef numpy.float64_t dtype_t
ctypedef numpy.int_t int_t
from libcpp.queue cimport priority_queue
from libcpp.utility cimport pair
from libcpp.unordered_set cimport unordered_set
cimport cython
import heapq

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
def dijkstra(int[:,:] f_nbrs_id, double[:,:] f_nbrs_dis, int[:] starts):
    # https://blog.paperspace.com/faster-numpy-array-processing-ndarray-cython/
    # 为什么numpy慢 https://stackoverflow.com/questions/42349587/cython-slow-numpy-arrays
    cdef int offset = starts[0]
    cdef int pos
    cdef double[:,:] f_dis = numpy.full((len(starts), len(f_nbrs_id)), -numpy.inf)  # f_dis[pos][i]是offset+pos到i的距离
    cdef si
    cdef int start
    cdef priority_queue[pair[double, int]] min_heap
    cdef unordered_set[int] visited  # min_heap: [(距离, fid)]
    cdef pair[double, int] cur_node
    cdef int i
    cdef int cur
    cdef int nid
    cdef double cur_dis
    cdef double n_dis
    cdef double dis
    for start in starts:
        pos =start-offset
        f_dis[pos,start] = 0
        min_heap = priority_queue[pair[double, int]]()
        min_heap.push(pair[double, int](0, start))
        visited = unordered_set[int]()
        while not min_heap.empty():
            cur_node = min_heap.top() # 当前的node
            min_heap.pop()
            cur_dis = cur_node.first
            cur = cur_node.second
            if visited.find(cur) != visited.end():
                continue
            visited.insert(cur)
            for i in range(3):
                nid = f_nbrs_id[cur,i]
                n_dis = f_nbrs_dis[cur,i]
                if visited.find(nid) != visited.end():
                    continue
                dis = cur_dis - n_dis  # + -> -
                if dis > f_dis[pos,nid]:  # < -> >
                    f_dis[pos,nid] = dis
                    min_heap.push(pair[double, int](dis, nid))
    return -numpy.array(f_dis)

def dijkstra_(f_nbrs_id, f_nbrs_dis, start):
    f_dis = numpy.full(len(f_nbrs_id), numpy.inf)  # f_dis[i]是start到i的距离
    f_dis[start] = 0
    min_heap, visited = [(0, start)], set()  # min_heap: [(距离, fid)]
    while min_heap:
        cur_dis, cur = heapq.heappop(min_heap)
        if cur in visited:
            continue
        visited.add(cur)

        for i in range(len(f_nbrs_id[cur])):
            nid = f_nbrs_id[cur][i]
            n_dis = f_nbrs_dis[cur][i]
            if nid in visited:
                continue
            dis = cur_dis + n_dis
            if dis < f_dis[nid]:
                f_dis[nid] = dis
                heapq.heappush(min_heap, (dis, nid))

    assert len(visited) == len(f_dis), f'{len(visited)} {len(f_dis)}'
    return f_dis

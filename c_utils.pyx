import numpy as np
import heapq

def dijkstra(f_nbrs_id, f_nbrs_dis, start):
    f_dis = np.full(len(f_nbrs_id), np.inf)  # f_dis[i]是start到i的距离
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

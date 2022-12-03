from typing import List
import heapq
from queue import Queue
import numpy as np
from numpy.linalg import norm


def read_ply():
    vertices, faces = [], []
    with open('dino.ply', 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    for i, line in enumerate(lines):
        if line.startswith('element vertex'):
            v_num = int(line.split(' ')[-1])
        if line.startswith('element face'):
            f_num = int(line.split(' ')[-1])
        if line == 'endheader':
            break
    for line in lines[-(v_num + f_num):-f_num]:
        x, y, z = line.split(' ')
        vertices.append([float(x), float(y), float(z)])
    for line in lines[-f_num:]:
        v1, v2, v3 = line.split(' ')[1:4]
        faces.append([int(v1), int(v2), int(v3)])
    return np.array(vertices), np.array(faces)


def write_ply(model):
    with open('output.ply', 'w') as f:
        f.write(f"ply\nformat ascii 1.0\n"
                f"element vertex {len(model.vs)}\nproperty float x\nproperty float y\nproperty float z\n"
                f"element face {len(model.fs)}\nproperty list uchar int vertex_indices\n"
                f"property uint8 red\nproperty uint8 green\nproperty uint8 blue\n"
                f"end_header\n")
        for v in model.vs:
            f.write(f'{v[0]}{v[1]}{v[2]}\n')
        for face in model.fs:
            f.write(f'3 {face.vids[0]} {face.vids[1]} {face.vids[2]} ')
            label = face.label
            f.write(f'{60 * (label % 4 + 1)} {80 * ((label + 1) % 3 + 1)} {50 * ((label + 2) % 5 + 1)}\n')


class NeighborInfo:
    def __init__(self, ev, fid, angle, ang_dis, geo_dis):
        self.vids = ev  # 邻边的顶点id
        self.fid = fid  # 邻面的id
        self.angle = angle  # 夹角
        self.ang_dis = ang_dis
        self.geo_dis = geo_dis
        self.dis = 0


class Face:
    def __init__(self, vs, fv):
        self.vids = fv  # 3个顶点的id
        self.center = sum(vs) / 3  # 质心位置
        self.norm = np.cross(vs[1] - vs[0], vs[2] - vs[0])  # 法向
        norm_len = norm(self.norm)
        self.norm = self.norm if norm_len < 1e-12 else self.norm / norm_len
        self.label = 0  # 分割标签
        self.nbrs: List[NeighborInfo] = []  # 邻居面片的信息


def compute_dis(f0: Face, f1: Face, ev):  # ev是邻边的两个顶点位置
    # 计算两个面片之间的距离信息
    angle = np.arccos(np.dot(f0.norm, f1.norm))
    is_convex = np.dot(f0.norm, f1.center - f0.center) < 1e-12
    eta = 0.2 if is_convex else 1.0  # 根据是否是凸的决定eta
    ang_dis = eta * (1 - np.dot(f0.norm, f1.norm))

    axis, d0, d1 = ev[1] - ev[0], f0.center - ev[0], f1.center - ev[0]  # 3个共起点的向量
    axis_len, d0_len, d1_len = norm(axis), norm(d0), norm(d1)
    angle0 = np.arccos(np.dot(d0, axis) / d0_len / axis_len)
    angle1 = np.arccos(np.dot(d1, axis) / d1_len / axis_len)
    # 如果将两个面片展平，测地距离就是两质心相连的直线段，构成一个三角形，夹角就是angle0+angle1，用余弦定理就能算出距离
    geo_dis = d0_len * d0_len + d1_len * d1_len - 2 * d0_len * d1_len * np.cos(angle0 + angle1)

    return angle, ang_dis, geo_dis


class Edge:  # TODO: 这个Edge有必要存在吗？
    def __init__(self, ev, fid):
        self.vids = (ev[0], ev[1]) if ev[0] < ev[1] else (ev[1], ev[0])  # 两个顶点的id TODO:这个大小变换是否有问题
        self.fid = fid  # 面片id


class FlowEdge:
    def __init__(self, fro, to, cap, flow):
        self.fro = fro
        self.to = to
        self.cap = cap
        self.flow = flow


class Model:

    def __init__(self, vs, fs: List[List[int]]):
        self.vs = vs  # 所有顶点位置
        self.fs = [Face(vs[f], f) for f in fs]  # 所有面片
        es = []
        for i, f in enumerate(fs):
            es.extend([Edge((f[0], f[1]), i), Edge((f[1], f[2]), i), Edge((f[2], f[0]), i)])  # 所有棱边
        # 为所有面片找到与之相邻的面片，并计算相邻面片的角距离ang、测地距离geo和总距离
        visited_es = {}
        for e in es:
            if e.vids not in visited_es:
                visited_es[e.vids] = e.fid
            else:
                f0, f1 = visited_es[e.vids], e.fid
                angle, ang_dis, geo_dis = compute_dis(self.fs[f0], self.fs[f1], e.vids)
                self.fs[f0].nbrs.append(NeighborInfo(e.vids, f1, angle, ang_dis, geo_dis))
                self.fs[f1].nbrs.append(NeighborInfo(e.vids, f0, angle, ang_dis, geo_dis))  # TODO: 边的方向要考虑吗？？
        count = sum([len(f.nbrs) for f in self.fs])
        avg_ang_dis = sum([sum([n.ang_dis for n in f.nbrs]) for f in self.fs]) / count
        avg_geo_dis = sum([sum([n.geo_dis for n in f.nbrs]) for f in self.fs]) / count
        for f in self.fs:
            for n in f.nbrs:
                n.dis = 0.2 * n.ang_dis / avg_ang_dis + 0.8 * n.geo_dis / avg_geo_dis  # TODO: delta？？
        # f_dis[i][j] is the shortest distance from f[i] to f[j]
        # 初始化路径图，用于计算最短路
        self.f_dis = np.full((len(self.fs), len(self.fs)), np.inf)
        self.compute_shortest()

        # 初始化流图，用于计算最大流
        self.edges: List[FlowEdge] = []  # TODO: 是否把G和edges合并起来？像他一样
        # G[x]，点x的所有边在edges中的下标，最后两个是用于的源点和汇点
        self.G: List[List[int]] = [[] for _ in range(len(self.fs) + 2)]
        for i, f in enumerate(self.fs):
            for n in f.nbrs:
                self.edges.append(FlowEdge(i, n.fid, 1 / (1 + n.ang_dis / avg_ang_dis), 0))
                self.G[i].append(len(self.edges) - 1)

    def compute_shortest(self):
        # Dijkstra算法
        # https://leetcode.com/problems/network-delay-time/discuss/329376/efficient-oe-log-v-python-dijkstra-min-heap-with-explanation
        # https://gist.github.com/kachayev/5990802
        for start in range(len(self.fs)):
            self.f_dis[start] = 0
            min_heap = [(0, start)]  # vertex is face. minheap of (distance, vertex) tuples. Initialize with (0, start).
            visited = set()  # visited vertex

            while min_heap:
                cur_dis, cur = heapq.heappop(min_heap)
                if cur in visited:
                    continue
                visited.add(cur)

                for n in self.fs[cur].nbrs:
                    if n.fid in visited:
                        continue
                    dis = cur_dis + n.dis
                    if dis < self.f_dis[start][n.fid]:
                        self.f_dis[start][n.fid] = dis
                        heapq.heappush(min_heap, (dis, n.fid))

            assert len(visited) == len(self.fs)

    def compute_flow(self, f_types):  # TODO：怎么做好，对f_types不为0的这些面片求最大流
        # f_types 无关区域0 边界区域1，2 模糊区域3
        # TODO: 是否要区别无向图和有向图？还是说不用管，无向图就直接用两个有向边就行
        # Ford-Fulkerson增广路算法-EK算法
        # https://www.desgard.com/2020/03/03/max-flow-ford-fulkerson.html
        # https://oi-wiki.org/graph/flow/max-flow/
        edges, G = self.edges, self.G
        # 新建两个结点，一个源点，一个汇点
        G[-2], G[-1] = [], []
        start, target = len(G) - 2, len(G) - 1  # TODO:参数
        f_types.extend([4, 4])

        # a = [0.0] * (len(self.fs) + 2)  # a[x], BFS中x的父边给到的流量, TODO: 这里没必要？
        p = [-1] * (len(self.fs) + 2)  # p[x], BFS中x的父边

        # 初始化流分布图
        for i, f_type in enumerate(f_types):
            if f_type == 1:
                edges.append(FlowEdge(start, i, float('inf'), 0))
                G[start].append(len(edges) - 1)
            elif f_type == 2:
                edges.append(FlowEdge(i, target, float('inf'), 0))  # TODO: 搞清楚他的 ！=target的条件是什么情况
                G[i].append(len(edges) - 1)
        for i, g in enumerate(G):
            if f_types[i]:  # 其他无关面片，流量溢出去不用管？
                for j in g:
                    edges[j].flow = 0

        sum_flow = 0  # 总最大流
        while True:
            # STEP1: 从源点一直BFS，碰到汇点就停
            a = [0.0] * (len(self.fs) + 2)
            a[start] = float('inf')
            q, q_history = Queue(), []  # 用q_history存储历史
            q.put(start)
            while not q.empty():
                cur = q.get()
                # 遍历cur的所有边e，如果e没有被搜索到且还有残余流量，就流
                for eid in G[cur]:
                    e = edges[eid]
                    if f_types[e.to] != 0 and not a[e.to] and e.cap > e.flow:
                        p[e.to] = eid  # 设置父边
                        a[e.to] = min(a[cur], e.cap - e.flow)  # 设置流量
                        q.put(e.to)  # 继续搜
                        q_history.append(e.to)
                if a[target]:  # TODO: 可以放到while语句里？
                    break
            flow_add = a[target]
            if not flow_add:  # 已经搜索不到了，结束
                # BFS涉及的区域，都是1
                for i in q_history:
                    f_types[i] = 1
                for i in range(len(self.fs)):
                    if f_types[i] == 3:
                        f_types[i] = 2
                break
            # STEP2: 反向传播流量
            cur = target
            while cur != start:
                edges[p[cur]].flow += flow_add  # 增加路径的flow值
                # 减小反向路径的flow值
                for eid in G[cur]:
                    if edges[eid].to == edges[p[cur]].fro:
                        edges[eid].flow -= flow_add
                # edges[p[cur] + p[cur] % 2 * 2 - 1] -= flow_add
                cur = edges[p[cur]].fro
            sum_flow += flow_add
        return f_types


label_nums = 0


class Solver:
    def __init__(self, model, level, fids):
        self.model = model
        self.eps = 0.04  # 判断哪些属于模糊区域
        self.fids = fids  # 对哪些面片做分解
        self.level = level  # 第几层

        self.fs = model.fs[fids]  # TODO: 这个fs是否有必要？直接用self.model.fs[xx]就行？
        self.ori_fid2fid = {fid: i for i, fid in enumerate(fids)}
        self.f_dis = model.f_dis[fids][:, fids]

        self.global_max_dis = model.f_dis.max()  # 整个模型最远的面片距离
        local_max_dis_fids = np.unravel_index(self.f_dis.argmax(), self.f_dis.shape)  # 最远的一对面片
        self.loac_max_dis = self.f_dis[local_max_dis_fids]

        def two_way_reps():
            return 2, list(local_max_dis_fids)  # 2路分解

        def k_way_reps():
            # 选作为到其他各点距离之和最小的点初始点
            reps = [np.argmin(np.sum(self.f_dis, axis=1))]
            G = []
            # 使最近的其他种子最远
            for i in range(20):  # 20个试验点
                max_dis = 0
                choice = 0
                for fid in range(len(self.f_dis)):  # local
                    min_dis = np.min([self.f_dis[fid][reps]])
                    if min_dis > max_dis:
                        max_dis = min_dis
                        choice = fid
                reps.append(choice)
                G.append(max_dis)

            # 最大化G[num]-G[num+1]
            num = np.argmax([G[num] - G[num + 1] for num in range(len(G) - 2)]) + 2
            reps = reps[:num]
            return num, reps  # 种子数和代表点

        self.num, self.reps = k_way_reps()
        # K路分解初始化中的操作

        # 计算 相邻面片且标签相同 这些面片对夹角的极差
        max_ang, min_ang = 0, np.pi
        for f in self.fs:
            for n in f.nbrs:
                k = n.fid
                # TODO: 是否需要和他一样
                if model.fs[k].label == f.label:
                    min_ang = min(n.ang, min_ang)
                    max_ang = min(n.ang, max_ang)
        self.ang_diff = max_ang - min_ang

    def solve(self):
        global label_nums
        # TODO: 注意区分fid和ori_fid一些列东西
        # TODO: 如果直接改成sklearn聚类怎么样
        prob = np.zeros((self.num, len(self.f_dis)))

        def compute_assign_prob():
            # TODO: mask是什么，目前看来全为1？mask是对标签的种子是否重复的判定，如果种子重合，只考虑一个标签
            # 用初始种子计算概率
            for fid in range(len(self.f_dis)):  # 这里的fid, rep都是local的了
                if fid in self.reps:
                    prob[self.reps.index(fid)][fid] = 1
                    continue
                sum_prob = sum([1 / self.f_dis[fid][rep] for rep in self.reps])
                for rep in range(self.num):
                    prob[rep][fid] = 1 / self.f_dis[fid][self.reps[rep]] / sum_prob

        def assign():
            # TODO: mask是什么，目前看来全为1？
            # 给面片打标签
            for fid in range(len(self.f_dis)):
                label1, label2 = heapq.nlargest(2, range(self.num), prob[:, fid].take)
                prob1, prob2 = prob[label1][fid], prob[label2][fid]
                if prob1 - prob2 > self.eps:
                    self.fs[fid].label = label_nums + label1
                else:
                    self.fs[fid].label = 1024 + label1 * self.num + label2  # TODO:一个奇怪的定义

        def update_rep():
            # 计算P(f∈local_label)
            # rep_dis[k][i]表示第i个面片与第k个隐种子的距离，用第i个面片与其他同标签的所有面片的平均距离来表示，视为常量即可
            # TODO: 改成rep_dis[i][k]
            rep_dis = np.zeros((self.num, len(self.f_dis)))
            counts = np.zeros(self.num)  # 每个类别的数量
            # STEP1: 计算P
            # STEP1.1: 计算rep_dis
            for i in range(len(self.f_dis)):
                local_label = self.fs[i].lavel - label_nums
                if local_label < self.num:
                    counts[local_label] += 1
                    rep_dis[local_label] += self.f_dis[i]  # TODO: 可否直接改成一个np.sum，全部一起算？
            for rep in range(self.num):
                for i in range(len(self.f_dis)):
                    rep_dis[rep][i] = rep_dis[rep][i] / counts[rep] if counts[rep] else np.inf
            # STEP1.2: 计算概率P
            for i in range(len(self.f_dis)):
                sum_prob = sum([1 / (rep_dis[rep][i] + 1e-12) for rep in range(self.num)])
                for rep in range(self.num):
                    prob[rep][i] = 1 / rep_dis[rep][i] / sum_prob

            # STEP2: 计算开销
            rep_cost = np.dot(prob, self.f_dis)  # rep_cost[rep][i]表示第rep个标签用i做种子的开销
            reps = list(np.argmin(rep_cost, axis=1))
            cost = [rep_cost[i][rep] for i, rep in enumerate(reps)]

            # STEP3: 判断是否更新
            original_cost = [rep_cost[i][rep] for i, rep in enumerate(self.reps)]
            changed = any(
                [(c1 < c0 - 1e-12 and r1 != r0) for r1, r0, c1, c0 in zip(reps, self.reps, cost, original_cost)])
            if changed:
                self.reps = reps
            return changed

        def assign_fuzzy():
            for i in range(self.num):
                for j in range(self.num):  # 两片模糊区域间的面片两两分割
                    f_types = np.zeros(len(self.model.fs))  # global的大小
                    # STEP1: 确定具体分割的面片
                    # 找到哪些是模糊区域 3， 哪些是边界区域1，2， 哪些是无关区域0
                    fuzzy_i, fuzzy_j = 1024 + i * self.num + j, 1024 + j * self.num + i
                    for fid, f in zip(self.fids, self.fs):  # 这个fid是global的fid
                        if f.label == fuzzy_i or f.label == fuzzy_j:
                            f_types[fid] = 3
                            for n in self.fs.nbrs:
                                nf = self.model.fs[n.fid]
                                if nf.label == label_nums + i:
                                    f_types[n.fid] = 1
                                elif nf.label == label_nums + j:
                                    f_types[n.fid] = 2

                    # STEP2: 进行分割
                    f_types = self.model.compute_flow(f_types)

                    # STEP3: 执行分割结果
                    for fid in self.fids:  # local
                        if f_types[fid] == 1:
                            self.model.fs[fid].label = label_nums + i
                        elif f_types[fid] == 2:
                            self.model.fs[fid].label = label_nums + j
                        else:  # TODO: 没有必要？
                            assert f_types[fid] == 0

                    # 检查是否有孤立的面片
                    for fid in self.fids:
                        if not any([self.model.fs[n.fid].label == self.model.fs[fid].label
                                    for n in self.model.fs[fid].nbrs]):
                            assert False, "isolated face"

        for step in range(20):  # 迭代20轮
            # STEP1: 用初始种子计算概率
            compute_assign_prob()
            # STEP2: 给面片打标签
            assign()
            change = update_rep()
            if not change:
                break

        assign()
        update_rep()  # TODO 这是否必要？
        assign()  # TODO 这是否必要？
        assign_fuzzy()
        label_nums += self.num

        # 递归下去
        loac_max_patch_dis = np.max(self.model.f_dis[self.reps][:, self.reps])
        if self.level > 2 or loac_max_patch_dis / self.global_max_dis < 0.1:  # TODO: 把global的东西都放到model里
            return
        for k in range(self.num):
            fids = [fid for fid in self.fids if self.model.fs[fid].label % self.num == k]
            solver = Solver(self.model, self.level + 1, fids)
            if solver.loac_max_dis / solver.global_max_dis > 0.2 and solver.ang_diff > 0.3:
                solver.solve()

        # TODO: solverK呢， solverK会有一个mask，很怪！


def run():
    # 读取文件，包括顶点和面
    vertices, faces = read_ply()
    model = Model(vertices, faces)
    solver = Solver(model, 0, list(range(len(faces))))
    solver.solve()
    write_ply(model)


if __name__ == '__main__':
    run()

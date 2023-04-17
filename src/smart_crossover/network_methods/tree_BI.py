from typing import List, Tuple

import numpy as np
from scipy import sparse as sp

from smart_crossover.formats import OptTransport
from smart_crossover.network_methods.net_manager import OTManager
from smart_crossover.output import Basis


def tree_basis_identify(ot_manager: OTManager, flow_weights: np.ndarray) -> Tuple[Basis, int]:
    # Find a max-weight spanning tree.
    tree = max_weight_spanning_tree(ot_manager.ot, flow_weights)
    # Push the tree to a basic feasible solution of the OT problem.
    tree_vbasis, push_iter = push_tree_to_bfs(ot_manager, tree)
    tree_basis = Basis(tree_vbasis, np.concatenate([-np.ones(ot_manager.m-1), np.array([0])]))
    return tree_basis, push_iter


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def unite(self, x: int, y: int) -> None:
        x_root = self.find(x)
        y_root = self.find(y)

        if x_root == y_root:
            return

        if self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
        elif self.rank[x_root] > self.rank[y_root]:
            self.parent[y_root] = x_root
        else:
            self.parent[y_root] = x_root
            self.rank[x_root] += 1


def max_weight_spanning_tree(ot: OptTransport, flow_weights: np.ndarray) -> np.ndarray[np.int_]:
    def edge_list() -> List[Tuple[int, int, float, int]]:
        edges_list = []
        n_rows, n_cols = ot.M.shape
        for i in range(n_rows):
            for j in range(n_cols):
                ind = i * n_cols + j
                edges_list.append((i, j + n_rows, flow_weights[ind], ind))
        return edges_list

    edges = edge_list()
    edges.sort(key=lambda edge: edge[2], reverse=True)

    n = ot.s.size + ot.d.size
    uf = UnionFind(n)
    tree_edges = np.full(n - 1, -1, dtype=np.int_)

    edge_count = 0
    for edge in edges:
        u, v, weight, index = edge
        if uf.find(u) != uf.find(v):
            uf.unite(u, v)
            tree_edges[edge_count] = index
            edge_count += 1
            if edge_count == n - 1:
                break

    return tree_edges


def push_tree_to_bfs(ot_manager: OTManager, tree: np.ndarray[np.int_]) -> Tuple[np.ndarray, int]:
    ot = ot_manager.ot
    B = ot_manager.mcf.A.tocsc()[:-1, :][:, tree]
    # tree_solution is the solution of the equation: B * x = ot_manager.mcf.b[:-1]
    tree_basic_vars = sp.linalg.spsolve(B, ot_manager.mcf.b[:-1])
    tree_solution = np.zeros(ot_manager.n)
    tree_solution[tree] = tree_basic_vars
    tree_solution = tree_solution.reshape((ot.s.size, ot.d.size))

    push_iter = 0
    negative_flows = np.where(tree_solution < 0)

    # deal with the converse flows one by one
    for i in range(len(negative_flows[0])):
        # consider the ith negative flow: x_{I1, J1}
        I1, J1 = negative_flows[0][i], negative_flows[1][i]

        J2 = np.argmax(tree_solution[I1, :])
        I2 = np.argmax(tree_solution[:, J1])

        while tree_solution[I1, J1] < 0:
            assert (tree_solution[I2, J1] > 0) and (tree_solution[I1, J2] > 0)
            assert tree_solution[I2, J2] == 0

            # "Irrigation" - try to make the converse flow positive
            candidate_flows = np.array([-tree_solution[I1, J1], tree_solution[I1, J2], tree_solution[I2, J1]])
            theta = np.min(candidate_flows)
            flag = np.argmin(candidate_flows)
            tree_solution[I1, J1] += theta
            tree_solution[I2, J1] -= theta
            tree_solution[I1, J2] -= theta
            tree_solution[I2, J2] += theta

            if flag == 1:
                J2 = np.argmax(tree_solution[I1, :])
            elif flag == 2:
                I2 = np.argmax(tree_solution[:, J1])

            push_iter += 1

    vbasis = -np.ones(ot_manager.n)
    vbasis[tree_solution.ravel() > 0] = 0
    return vbasis, push_iter

from typing import List, Tuple

import numpy as np
from scipy import sparse as sp
from unionfind import unionfind

from smart_crossover.formats import OptTransport
from smart_crossover.network_methods.net_manager import OTManager
from smart_crossover.output import Basis


def tree_basis_identify(ot_manager: OTManager, flow_weights: np.ndarray) -> Basis:
    # Find a max-weight spanning tree.
    tree = max_weight_spanning_tree(ot_manager.ot, flow_weights)
    # Push the tree to a basic feasible solution of the OT problem.
    tree_vbasis = push_tree_to_bfs(ot_manager, tree)
    tree_basis = Basis(tree_vbasis, np.concatenate([-np.ones(ot_manager.m), np.array([0])]))
    return tree_basis


def max_weight_spanning_tree(ot: OptTransport, flow_weights: np.ndarray) -> np.ndarray[np.ndarray[np.int_]]:
    def edge_list() -> List[Tuple[int, int, float, int]]:
        edges_list = []
        n_rows, n_cols = ot.M.shape
        for i in range(n_rows):
            for j in range(n_cols):
                ind = i * n_cols + j
                edges_list.append((i, j, ot.M[i, j] * flow_weights[ind], ind))
        return edges_list

    edges = edge_list()
    edges.sort(key=lambda edge: edge[2], reverse=True)

    n = ot.s.size + ot.d.size
    uf = unionfind(n)
    tree_edges = np.empty(n - 1, dtype=int)

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


def push_tree_to_bfs(ot_manager: OTManager, tree: np.ndarray[np.ndarray[np.int_]]) -> np.ndarray:
    ot = ot_manager.ot
    B = ot_manager.mcf.A[:-1, :][:, tree]
    # tree_solution is the solution of the equation: B * x = ot_manager.mcf.b[:-1]
    tree_basic_vars = sp.linalg.solve(B, ot_manager.mcf.b[:-1])
    tree_solution = np.zeros(ot_manager.n)
    tree_solution[tree] = tree_basic_vars
    tree_solution.reshape((ot.s.size, ot.d.size))

    iter = 0
    aim = np.where(tree_solution < 0)[0]  # find converse flows

    # deal with the converse flows one by one
    for i in range(len(aim)):
        Ind = aim[i]

        # consider the ith converse flow: x_{I1, J1}
        J1 = int(np.ceil(Ind / ot.s.size))
        I1 = Ind % ot.s.size
        if I1 == 0:
            I1 = ot.s.size

        J2 = np.argmax(tree_solution[I1 - 1, :])
        I2 = np.argmax(tree_solution[:, J1 - 1])

        while tree_solution[I1 - 1, J1 - 1] < 0:
            assert (tree_solution[I2, J1 - 1] > 0) and (tree_solution[I1 - 1, J2] > 0)
            assert tree_solution[I2, J2] == 0

            # "Irrigation" - try to make the converse flow positive
            candidate_flows = np.ndarray([-tree_solution[I1 - 1, J1 - 1], tree_solution[I1 - 1, J2], tree_solution[I2, J1 - 1]])
            theta = np.min(candidate_flows)
            flag = np.argmin(candidate_flows)
            tree_solution[I1 - 1, J1 - 1] += theta
            tree_solution[I2, J1 - 1] -= theta
            tree_solution[I1 - 1, J2] -= theta
            tree_solution[I2, J2] += theta

            if flag == 1:
                J2 = np.argmax(tree_solution[I1 - 1, :])
            elif flag == 2:
                I2 = np.argmax(tree_solution[:, J1 - 1])

            iter += 1

    vbasis = -np.ones(ot_manager.n)
    vbasis[tree_solution > 0] = 0
    return vbasis

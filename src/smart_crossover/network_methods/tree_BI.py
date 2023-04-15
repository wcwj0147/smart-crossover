from typing import List, Tuple

import numpy as np
from scipy import sparse as sp
import networkx as nx

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
    G = nx.Graph()
    G.add_weighted_edges_from([(u, v, weight) for u, v, weight, _ in edges])

    tree_edges = nx.algorithms.tree.maximum_spanning_edges(G, algorithm='kruskal', data=False)
    tree_edge_indices = [next(index for u, v, _, index in edges if (edge == (u, v) or edge == (v, u))) for edge in tree_edges]

    return np.array(tree_edge_indices, dtype=np.int_)


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

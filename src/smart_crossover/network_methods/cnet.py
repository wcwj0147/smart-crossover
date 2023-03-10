import time
from typing import Tuple

import numpy as np
import scipy.sparse as sp

from smart_crossover.input import MCFInput
from smart_crossover.output import Output
from smart_crossover.solver_runner.utils import generate_runner


def cnet_mcf(mcf_input: MCFInput,
             x: np.ndarray,
             solver: str = "GRB") -> Output:
    sort_start_time = time.time()

    A = mcf_input.A
    b = mcf_input.b
    c = mcf_input.c
    l = mcf_input.l
    u = mcf_input.u

    m, n = b.size, c.size

    mask = x > u / 2
    x_hat = x * (~mask) + u * mask - x * mask
    x_hat[(x < 0) | (x > u)] = 0

    A_bar = A.multiply(~mask) - A.multiply(mask)

    A_barplus = A_bar.maximum(sp.csc_matrix((m, n)))
    A_barminus = (-A_bar).maximum(sp.csc_matrix((m, n)))

    f_1 = A_barplus @ x_hat
    f_2 = A_barminus @ x_hat
    f = np.maximum(f_1, f_2)
    f_inv = 1 / f

    row, col, a = sp.find(A_bar)
    val = f_inv[row] * x_hat[col]
    r = sp.csc_matrix((val * a, (row, col)), shape=(m, n))

    r_1 = sp.csr_matrix.max(r.multiply(sp.csr_matrix.sign(r)), axis=0)
    r_1 = r_1.toarray().reshape((n,))
    queue = np.argsort(-r_1)

    c_max = np.max(np.abs(c))
    c = c / c_max

    K = n
    COLUMN_GENERATION_RATIO = 2

    sort_time = time.time() - sort_start_time
    cg_start_time_1 = time.time()

    b_1 = b.copy()
    b_true = b - A.multiply(mask) @ (u * mask)
    sig = np.sign(b_true)
    sig[sig == 0] = 1
    c_1 = np.concatenate([c, K * np.ones(m) * sig], axis=0)
    ll = np.zeros(m)
    uu = np.Inf * np.ones(m)
    ll[sig < 0] = -np.Inf
    uu[sig < 0] = 0
    l_1 = np.concatenate([l, ll], axis=0)
    u_1 = np.concatenate([u, uu], axis=0)
    bas_1 = np.concatenate((-np.ones(n), np.zeros(m)), axis=0)
    bas_1[np.concatenate((mask, np.array([False]*m)), axis=0)] = -2
    T_1 = np.where(bas_1 == 0)[0]
    A_1 = sp.hstack((A, sp.identity(m)))
    A_1 = A_1.tocsr()

    left = 1
    k = int(1.2 * m)
    flag = True

    t_1 = cg_start_time_1 - time.time()
    t_2, t_3 = 0, 0
    iter_count = 0

    while flag:
        cg_start_time_2 = time.time()

        if left > len(queue):
            print(' ##### Column generation algorithm fails! #####')
            break
        right = min(k, len(queue))
        T_1 = list(set(T_1).union(set(queue[left:right])))
        T_1.sort()

        t_2 = t_2 + time.time() - cg_start_time_2

        sub_output, bas_1 = sub_solving(MCFInput(A_1, b_1, c_1, l_1, u_1), bas_1, T_1, solver)

        cg_start_time_3 = time.time()

        B = A_1[:, bas_1 == 0]
        sol = np.zeros(n+m)
        sol[T_1] = sub_output.x
        sol[bas_1 == -2] = u_1[bas_1 == -2]
        c_B = c_1[bas_1 == 0]
        p = sp.linalg.spsolve(B, c_B)
        r = c_1 - A_1.T @ p
        if (np.max(sol[n:n + m - 1] * np.sign(sol[n:n + m - 1])) < 1e-10) and (
                np.min(np.concatenate((r[bas_1 == -1], -r[bas_1 == -2]))) > -1e-6):
            flag = False
        r[bas_1 == -2] = -r[bas_1 == -2]
        temp = r[queue]
        loc = np.where(temp < 0)[0]
        k = max(int(COLUMN_GENERATION_RATIO * k), loc[0]) if len(loc) > 0 else int(COLUMN_GENERATION_RATIO * k)
        left = right + 1

        t_3 = time.time() - cg_start_time_3 + sub_output.runtime
        iter_count = iter_count + sub_output.iter_count

        runtime = sort_time + t_1 + t_2 + t_3
        obj_val_1 = np.dot(c_1.T, sol) * c_max
        x_1 = sol
        bas = bas_1[:n].reshape(1, n)

    return Output(x=x_1, obj_val=obj_val_1, runtime=runtime, iter_count=iter_count, vbasis=bas)


def sub_solving(mcf_input: MCFInput,
                basis: np.ndarray,
                T: list,
                solver: str) -> Tuple[Output, np.ndarray]:
    m, n = mcf_input.b.size, mcf_input.c.size
    A_sub = mcf_input.A[:, T]
    c_sub = mcf_input.c[T]
    l_sub = mcf_input.l[T]
    u_sub = mcf_input.u[T]
    bas = basis.copy()
    bas_sub = basis[T]
    T_c = list(set(range(n)).difference(set(T)))
    indup = np.where(bas == -2)[0]
    indup_rest = list(set(T_c) & set(indup))
    b_sub = mcf_input.b - mcf_input.A[:, indup_rest] @ mcf_input.u[indup_rest]
    mcf_input_sub = MCFInput(A_sub, b_sub, c_sub, l_sub, u_sub)

    sub_runner = generate_runner(solver)
    sub_runner.read_mcf_input(mcf_input_sub)
    sub_runner.add_warm_start_basis(vbasis=bas_sub, cbasis=-np.ones((m, 1)))
    sub_runner.run_network_simplex()
    basis[T] = sub_runner.return_basis()[0]
    return(Output(x=sub_runner.return_x(),
                  obj_val=sub_runner.return_obj_val(),
                  runtime=sub_runner.return_runtime(),
                  iter_count=sub_runner.return_iter_count()),
           basis)

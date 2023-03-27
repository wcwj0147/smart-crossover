import datetime

import gurobipy
import numpy as np
import scipy.sparse as sp

from smart_crossover.input import MCFInput
from smart_crossover.output import Output, Basis
from smart_crossover.solver_caller.caller import generate_caller


def cnet_mcf(mcf_input: MCFInput,
             x: np.ndarray,
             solver: str = "GRB") -> Output:
    sort_start_time = datetime.datetime.now()

    # preparation
    A = mcf_input.A
    b = mcf_input.b
    c = mcf_input.c
    l = mcf_input.l
    u = mcf_input.u
    m, n = b.size, c.size
    assert np.sum(b) == 0, "Supply is not equal to demand."
    assert np.all(l == 0), "There exist non-zero lower bounds."

    # reverse large flow
    mask_large_x = x > u / 2
    large_x_ind = np.where(mask_large_x)[0]
    x_hat = x * (~mask_large_x) + u * mask_large_x - x * mask_large_x
    x_hat[(x < 0) | (x > u)] = 0
    A_bar = A.multiply(~mask_large_x) - A.multiply(mask_large_x)

    # calculate sorted variables (flows)
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

    # cost rescaling
    c_max = np.max(np.abs(c))
    c = c / c_max

    # set big-M: K
    K = n * np.max(u)
    # set expanding ratio for column generation
    COLUMN_GENERATION_RATIO = 2

    sort_time = datetime.datetime.now() - sort_start_time
    cg_start_time_1 = datetime.datetime.now()

    # set extended problem
    assert np.sum(A.multiply(mask_large_x) @ (u * mask_large_x)) == 0, "Partial flow is not valid."
    b_true = b - A.multiply(mask_large_x) @ (u * mask_large_x)
    b_sign = np.sign(b_true)
    b_sign[b_sign == 0] = 1
    c_1 = np.concatenate([c, K * np.ones(m)])
    l_1 = np.concatenate([l, np.zeros(m)])
    u_1 = np.concatenate([u, np.Inf * np.ones(m)])
    A_1 = sp.hstack((A, sp.diags(b_sign)))
    A_1 = sp.vstack((A_1, sp.csr_matrix(np.concatenate([np.zeros(n), -b_sign]))))
    A_1 = A_1.tocsr()
    b_1 = np.concatenate([b, np.array([0])])
    vbasis_1 = np.concatenate((-np.ones(n), np.zeros(m)))
    vbasis_1[large_x_ind] = -2
    cbasis_1 = np.concatenate([-np.ones(m), np.zeros(1)])
    T_1 = np.where(vbasis_1 == 0)[0]

    left = 0
    k = int(1.2 * m)
    flag = True

    t_1 = cg_start_time_1 - datetime.datetime.now()
    t_2, t_3 = 0, 0
    iter_count = 0

    while flag:
        cg_start_time_2 = datetime.datetime.now()

        if left >= len(queue):
            print(' ##### Column generation algorithm fails! #####')
            break
        right = min(k, len(queue))
        T_1 = list(set(T_1).union(set(queue[left:right])))
        T_1.sort()

        t_2 = t_2 + datetime.datetime.now() - cg_start_time_2

        sub_output = sub_solving(MCFInput(A_1, b_1, c_1, l_1, u_1),
                                 Basis(vbasis_1, cbasis_1),
                                 T_1,
                                 solver)

        cg_start_time_3 = datetime.datetime.now()

        # Update basis
        vbasis_1, sol = -np.ones(n + m, dtype=int), np.zeros(n + m)
        vbasis_1[T_1], sol[T_1] = sub_output.basis.vbasis, sub_output.x
        vbasis_1[list(set(large_x_ind).difference(set(T_1)))] = -2
        cbasis_1 = sub_output.basis.cbasis
        sol[vbasis_1 == -2] = u_1[vbasis_1 == -2]

        # Calculate reduced cost from dual solution
        r = c_1 - A_1.T @ sub_output.y

        # # Use vbasis, cbasis to calculate r:
        # A_fullrowrank = A_1[cbasis_1 == -1, :]
        # B = A_fullrowrank[:, vbasis_1 == 0]
        # c_B = c_1[vbasis_1 == 0]
        # p = sp.linalg.spsolve(B, c_B)
        # r = c_1 - A_fullrowrank.T @ p

        # Check stop criterion
        if (np.max(np.abs(sol[n:n+m])) < 1e-8) and (
                np.min(np.concatenate((r[vbasis_1 == -1], -r[vbasis_1 == -2]))) > -1e-6):
            flag = False
        r[vbasis_1 == -2] = -r[vbasis_1 == -2]
        temp = r[queue]
        loc = np.where(temp < 0)[0]
        k = max(int(COLUMN_GENERATION_RATIO * k), loc[0]) if len(loc) > 0 else int(COLUMN_GENERATION_RATIO * k)
        left = right

        t_3 = datetime.datetime.now() - cg_start_time_3 + sub_output.runtime
        iter_count += sub_output.iter_count

    runtime = sort_time + t_1 + t_2 + t_3
    obj_val_1 = c_1.T @ sol * c_max
    x_1 = sol
    basis = Basis(vbasis_1[:n], cbasis_1[:m])

    return Output(x=x_1, obj_val=obj_val_1, runtime=runtime, iter_count=iter_count, basis=basis)


def sub_solving(mcf_input: MCFInput,
                basis: Basis,
                T: list,
                solver: str) -> Output:
    start = datetime.datetime.now()
    A_sub = mcf_input.A[:, T]
    c_sub = mcf_input.c[T]
    l_sub = mcf_input.l[T]
    u_sub = mcf_input.u[T]
    bas_sub = basis.vbasis[T]
    indup = np.where(basis.vbasis == -2)[0]
    indup_rest = list(set(indup).difference(set(T)))
    b_sub = mcf_input.b - mcf_input.A[:, indup_rest] @ mcf_input.u[indup_rest]
    mcf_input_sub = MCFInput(A_sub, b_sub, c_sub, l_sub, u_sub)

    # Check whether basis is a feasible starting basis
    BB = A_sub[:, bas_sub == 0]
    bb = b_sub - A_sub[:, bas_sub == -2] @ u_sub[bas_sub == -2]
    xx = sp.linalg.spsolve(BB[basis.cbasis == -1, :], bb[basis.cbasis == -1])
    assert np.count_nonzero(xx < 0) + np.count_nonzero(xx > u_sub[bas_sub == 0]) == 0, "The given basis is not feasible."

    sub_runner = generate_caller(solver)
    sub_runner.read_mcf_input(mcf_input_sub)
    sub_runner.add_warm_start_basis(Basis(bas_sub, basis.cbasis))
    # sub_runner.turn_off_presolve()
    sub_runner.run_simplex()
    return (Output(x=sub_runner.return_x(),
                   y=sub_runner.return_y(),
                   obj_val=sub_runner.return_obj_val(),
                   runtime=sub_runner.return_runtime() + datetime.datetime.now() - start,
                   iter_count=sub_runner.return_iter_count(),
                   basis=sub_runner.return_basis(),
                   rcost=sub_runner.return_reduced_cost()))


# # Debug
# goto_mps_path = get_project_root() / "data/goto"
# model = gurobipy.read("/Users/jian/Documents/2023 Spring/smart-crossover/data/goto/netgen_8_14a.mps")
# gur_runner = GrbRunner(tolerance=1e-2)
# gur_runner.read_model(model)
# x = np.load("/Users/jian/Documents/2023 Spring/smart-crossover/data/goto/x_netgen.npy")
# mcf_input = gur_runner.return_MCF_model()
# cnet_mcf(mcf_input, x, "GRB")

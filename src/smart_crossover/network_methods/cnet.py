import datetime

import gurobipy
import numpy as np
import scipy.sparse as sp

from smart_crossover import get_project_root
from smart_crossover.formats import MinCostFlow
from smart_crossover.output import Output, Basis
from smart_crossover.solver_caller.caller import SolverSettings
from smart_crossover.solver_caller.gurobi import GrbCaller
from smart_crossover.solver_caller.utils import generate_solver_caller


# Todo: Clean this algorithm using new methods.
def cnet_mcf(mcf: MinCostFlow,
             x: np.ndarray,
             solver: str = "GRB") -> Output:
    sort_start_time = datetime.datetime.now()

    # preparation
    A = mcf.A
    b = mcf.b
    c = mcf.c
    u = mcf.u
    m, n = b.size, c.size

    # reverse large flow
    mask_large_x = x > u / 2
    ind_fix_to_up = np.where(mask_large_x)[0]
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
    u_1 = np.concatenate([u, np.Inf * np.ones(m)])
    A_1 = sp.hstack((A, sp.diags(b_sign)))
    A_1 = sp.vstack((A_1, sp.csr_matrix(np.concatenate([np.zeros(n), -b_sign]))))
    A_1 = A_1.tocsr()
    b_1 = np.concatenate([b, np.array([0])])
    vbasis_1 = np.concatenate((-np.ones(n), np.zeros(m)))
    vbasis_1[ind_fix_to_up] = -2
    cbasis_1 = np.concatenate([-np.ones(m), np.zeros(1)])
    T_1 = np.where(vbasis_1 == 0)[0]

    left = 0
    k = int(1.2 * m)
    flag = True

    t_1 = datetime.datetime.now() - cg_start_time_1
    t_2, t_3 = datetime.timedelta(seconds=0), datetime.timedelta(seconds=0)
    iter_count = 0

    while flag:
        cg_start_time_2 = datetime.datetime.now()

        if left >= len(queue):
            print(' ##### Column generation algorithm fails! #####')
            break
        right = min(k, len(queue))
        T_1 = list(set(T_1).union(set(queue[left:right])))
        T_1.sort()

        t_2 = t_2 + (datetime.datetime.now() - cg_start_time_2)

        sub_output = sub_solving(MinCostFlow(A_1, b_1, c_1, u_1),
                                 Basis(vbasis_1, cbasis_1),
                                 T_1,
                                 solver)

        cg_start_time_3 = datetime.datetime.now()

        # Update basis
        vbasis_1, sol = -np.ones(n + m, dtype=int), np.zeros(n + m)
        vbasis_1[T_1], sol[T_1] = sub_output.basis.vbasis, sub_output.x
        vbasis_1[list(set(ind_fix_to_up).difference(set(T_1)))] = -2
        cbasis_1 = sub_output.basis.cbasis
        sol[vbasis_1 == -2] = u_1[vbasis_1 == -2]

        # Calculate reduced cost from dual solution
        r = c_1 - A_1.T @ sub_output.y

        # Check stop criterion
        if (np.max(np.abs(sol[n:n+m])) < 1e-8) and (
                np.min(np.concatenate((r[vbasis_1 == -1], -r[vbasis_1 == -2]))) > -1e-6):
            flag = False
        r[vbasis_1 == -2] = -r[vbasis_1 == -2]
        temp = r[queue]
        loc = np.where(temp < 0)[0]
        k = max(int(COLUMN_GENERATION_RATIO * k), loc[0]) if len(loc) > 0 else int(COLUMN_GENERATION_RATIO * k)
        left = right

        t_3 = (datetime.datetime.now() - cg_start_time_3) + sub_output.runtime
        iter_count += sub_output.iter_count

    runtime = sort_time + t_1 + t_2 + t_3
    obj_val_1 = c_1.T @ sol * c_max
    x_1 = sol
    basis = Basis(vbasis_1[:n], cbasis_1[:m])

    return Output(x=x_1, obj_val=obj_val_1, runtime=runtime, iter_count=iter_count, basis=basis)


def sub_solving(mcf: MinCostFlow,
                basis: Basis,
                T: list,
                solver: str) -> Output:
    start = datetime.datetime.now()
    A_sub = mcf.A[:, T]
    c_sub = mcf.c[T]
    l_sub = mcf.l[T]
    u_sub = mcf.u[T]
    bas_sub = basis.vbasis[T]
    indup = np.where(basis.vbasis == -2)[0]
    indup_rest = list(set(indup).difference(set(T)))
    b_sub = mcf.b - mcf.A[:, indup_rest] @ mcf.u[indup_rest]
    mcf_input_sub = MinCostFlow(A_sub, b_sub, c_sub, l_sub, u_sub)

    # Check whether basis is a feasible starting basis
    BB = A_sub[:, bas_sub == 0]
    bb = b_sub - A_sub[:, bas_sub == -2] @ u_sub[bas_sub == -2]
    xx = sp.linalg.spsolve(BB[basis.cbasis == -1, :], bb[basis.cbasis == -1])
    assert np.count_nonzero(xx < 0) + np.count_nonzero(xx > u_sub[bas_sub == 0]) == 0, "The given basis is not feasible."

    sub_caller = generate_solver_caller(solver)
    sub_caller.read_mcf(mcf_input_sub)
    sub_caller.add_warm_start_basis(Basis(bas_sub, basis.cbasis))
    end = datetime.datetime.now()
    sub_caller.run_simplex()
    return (Output(x=sub_caller.return_x(),
                   y=sub_caller.return_y(),
                   obj_val=sub_caller.return_obj_val(),
                   runtime=sub_caller.return_runtime() + (end - start),
                   iter_count=sub_caller.return_iter_count(),
                   basis=sub_caller.return_basis(),
                   rcost=sub_caller.return_reduced_cost()))


# Debug
goto_mps_path = get_project_root() / "data/goto"
model = gurobipy.read("/Users/jian/Documents/2023 Spring/smart-crossover/data/goto/netgen_8_14a.mps")
gur_runner = GrbCaller(SolverSettings())
gur_runner.read_model(model)
x = np.load("/Users/jian/Documents/2023 Spring/smart-crossover/data/goto/x_netgen.npy")
mcf_input = gur_runner.return_MCF()
cnet_mcf(mcf_input, x, "GRB")

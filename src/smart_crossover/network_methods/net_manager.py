from typing import Tuple
from typing_extensions import Protocol

import numpy as np
from scipy import sparse as sp

from smart_crossover.formats import MinCostFlow, OptTransport
from smart_crossover.output import Basis, Output
from smart_crossover.parameters import TOLERANCE_FOR_ARTIFICIAL_VARS, TOLERANCE_FOR_REDUCED_COSTS
from smart_crossover.solver_caller.caller import SolverSettings
from smart_crossover.solver_caller.solving import solve_mcf


class NetworkManager(Protocol):
    """ A protocol for network problem managers, including MinCostFlow and OptimalTransport.

    Attributes:
        m: The number of constraints in the original network problem.
        n: The number of variables in the original network problem.
        basis: The current basis solved from a subproblem.
    """

    m: int
    n: int
    basis: Basis

    def get_sorted_flows(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Get the sorted flows by calculating flow indicators.

        Args:
            x: A (non-vertex) flow solution of the network problem.

        Returns:
            A tuple of two arrays: the first array is the sorted indices of flows, and the second array is the flow indicators.
        """
        ...

    def recover_x_from_sub_x(self, x_sub: np.ndarray) -> np.ndarray:
        """ Recover the solution of the original network problem from the solution of the subproblem.

        Args:
            x_sub: A solution of the subproblem.

        Returns:
            The corresponding solution of the original network problem.
        """
        ...

    def recover_basis_from_sub_basis(self, basis_sub: Basis) -> Basis:
        """ Recover the basis of the original network problem from the basis of the subproblem.

        Args:
            basis_sub: A basis of the subproblem.

        Returns:
            The corresponding basis of the original network problem.
        """
        ...

    def solve_subproblem(self, solver: str, solver_settings: SolverSettings) -> Output:
        """ Solve the subproblem of the network problem.

        Args:
            solver: The solver to use.
            solver_settings: The settings for the solver.

        Returns:
            The output of the subproblem solution.
        """
        ...

    def recover_obj_val(self, obj_val: float) -> float:
        """ Recover the objective value of the original network problem from the objective value of the subproblem.

        Args:
            obj_val: The objective value of the subproblem.

        Returns:
            The corresponding objective value of the original network problem.
        """
        ...

    def check_optimality_condition(self, x: np.ndarray, y: np.ndarray) -> bool:
        """ Check the optimality condition for the original network problem with a pair of primal-dual solution (x, y) for the subproblem.

        Args:
            x: A primal solution of the subproblem.
            y: A dual solution of the subproblem.

        Returns:
            True if the optimality condition is satisfied; False otherwise.
        """
        ...

    def add_free_variables(self, ind_free: np.ndarray) -> None:
        """ Add free variables (here free means free to change values) to the subproblem of the network problem.

        Args:
            ind_free: The indices of free variables to add.
        """
        ...

    def update_subproblem(self):
        """ Update the subproblem of the network problem, after fixing or freeing several variables."""
        ...

    def set_basis(self, basis: Basis) -> None:
        """ Set a feasible basis for the current subproblem.

        Args:
            basis: The basis to set.
        """
        ...


class MCFManagerStd():
    """A class to manage the MCF problem and its subproblem in network crossover algorithms.

    Attributes:
        mcf (MinCostFlow): The current MCF problem.
        mcf_sub (MinCostFlow): The subproblem of the current MCF problem.

    """

    mcf: MinCostFlow

    def __init__(self, mcf: MinCostFlow) -> None:
        self.mcf = mcf
        self.m = self.mcf.b.size
        self.n = self.mcf.c.size
        self.var_info = {'non_fix': np.array(range(self.n), dtype=np.int64)}
        self.artificial_vars = np.array([])
        self.c_rescaling_factor = None

    def extend_by_bigM(self, bigM: float) -> None:
        """ Extend the MCF problem by the bigM method:
        add an artificial node and add artificial arcs to satisfy all the demands and supplies, with bigM as the cost.

        Args:
            bigM: The bigM value.
        """
        mask_fix_up = np.zeros(self.n, dtype=bool)
        mask_fix_up[self.var_info['fix_up']] = True
        b_true = self.mcf.b - self.mcf.A.multiply(mask_fix_up) @ (self.mcf.u * mask_fix_up)
        b_sign = np.sign(b_true)
        b_sign[b_sign == 0] = 1
        c_1 = np.concatenate([self.mcf.c, bigM * np.ones(self.m)])
        u_1 = np.concatenate([self.mcf.u, np.Inf * np.ones(self.n)])
        A_1 = sp.hstack((self.mcf.A, sp.diags(b_sign)))
        A_1 = sp.vstack((A_1, sp.csr_matrix(np.concatenate([np.zeros(self.n), -b_sign]))))
        b_1 = np.concatenate([self.mcf.b, np.array([0])])
        self.mcf = MinCostFlow(A_1, b_1, c_1, u_1)
        self.artificial_vars = np.array(range(self.n, self.n + self.m), dtype=int)
        self.var_info['non_fix'] = np.append(self.var_info['non_fix'], np.array(range(self.n, self.n + self.m), dtype=np.int64))

    def get_sorted_flows(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Get the sorted flows by calculating flow indicators.

        Args:
            x: A (non-vertex) flow solution () of the MCF problem.

        Returns:
            A tuple of two arrays: the first array is the sorted indices of flows, and the second array is the flow indicators.
        """
        # Reverse large flow.
        mask_large_x = x > self.mcf.u / 2
        x_hat = x * (~mask_large_x) + self.mcf.u * mask_large_x - x * mask_large_x
        x_hat[(x < 0) | (x > self.mcf.u)] = 0
        A_bar = self.mcf.A.multiply(~mask_large_x) - self.mcf.A.multiply(mask_large_x)

        # Calculate flows indicators (use r to represent at the first stage).
        A_barplus = A_bar.maximum(sp.csc_matrix((self.m, self.n)))
        A_barminus = (-A_bar).maximum(sp.csc_matrix((self.m, self.n)))
        f_1 = A_barplus @ x_hat
        f_2 = A_barminus @ x_hat
        f = np.maximum(f_1, f_2)
        f_inv = np.divide(1, f, out=np.zeros_like(f), where=f != 0)
        row, col, a = sp.find(A_bar)
        val = f_inv[row] * x_hat[col]
        r = sp.csc_matrix((val * a, (row, col)), shape=(self.m, self.n))
        r_1 = sp.csr_matrix.max(r.multiply(sp.csr_matrix.sign(r)), axis=0)
        flow_indicators = r_1.toarray().reshape((self.n,))
        # sort flows by indicators
        return np.argsort(flow_indicators)[::-1], flow_indicators

    def set_initial_basis(self) -> None:
        """ Set the initial basis for the extended MCF:
            all original variables are non-basic, and the artificial variables are basic."""
        vbasis = np.concatenate((-np.ones(self.n), np.zeros(self.m)))
        vbasis[self.var_info['fix_up']] = -2
        cbasis = np.concatenate([-np.ones(self.m), np.zeros(1)])
        self.set_basis(Basis(vbasis, cbasis))

    def set_basis(self, basis: Basis) -> None:
        """ Add a feasible basis for the current MCF.

        Args:
            basis: The basis to set.
        """
        self.basis = basis

    def update_subproblem(self) -> None:
        """ Update the sub problem by the information of variables. """
        self.mcf_sub = MinCostFlow(
            A=self.mcf.A[:, self.var_info['non_fix']],
            b=self.mcf.b - self.mcf.A[:, self.var_info['fix_up']] @ self.mcf.u[self.var_info['fix_up']],
            c=self.mcf.c[self.var_info['non_fix']],
            u=self.mcf.u[self.var_info['non_fix']]
        )

    def solve_subproblem(self, solver: str, solver_settings: SolverSettings) -> Output:
        """ Solve the sub problem. Use network simplex method for Cplex, and default method for other solvers.

        Args:
            solver: The solver to use.
            solver_settings: The settings for the solver.

        Returns:
            The output of the sub problem solution.
        """
        method = "network_simplex" if solver == "CPL" else "default"
        return solve_mcf(self.mcf_sub, solver=solver, method=method, warm_start_basis=Basis(self.basis.vbasis[self.var_info['non_fix']], self.basis.cbasis), settings=solver_settings)

    def fix_variables(self, ind_fix_to_low: np.ndarray, ind_fix_to_up: np.ndarray) -> None:
        """ Fix some variables to lower/upper bounds in the sub problem.

        Args:
            ind_fix_to_low: The indices of variables to be fixed to lower bounds.
            ind_fix_to_up: The indices of variables to be fixed to upper bounds.
        """
        self.var_info['fix_low'] = ind_fix_to_low
        self.var_info['fix_up'] = ind_fix_to_up
        self.var_info['non_fix'] = np.setdiff1d(range(len(self.mcf.c)), np.append(ind_fix_to_low, ind_fix_to_up))
        self.var_info['fix'] = np.setdiff1d(range(len(self.mcf.c)), self.var_info['non_fix'])

    def add_free_variables(self, ind_free_new: np.ndarray) -> None:
        """ Add some variables to the sub problem.

        Args:
            ind_free_new: The indices of variables to add.
        """
        self.var_info['non_fix'] = np.append(self.var_info['non_fix'], ind_free_new)
        self.var_info['fix'] = np.setdiff1d(self.var_info['fix'], ind_free_new)
        self.var_info['fix_low'] = np.setdiff1d(self.var_info['fix_low'], ind_free_new)
        self.var_info['fix_up'] = np.setdiff1d(self.var_info['fix_up'], ind_free_new)

    def recover_x_from_sub_x(self, x_sub: np.ndarray) -> np.ndarray:
        """ Recover the solution of the current LP from the solution of the sub problem.

        Args:
            x_sub: A solution of the sub problem.

        Returns:
            The corresponding solution of the original, extended problem. (Extend the sub solution by adding fixed variables.)
        """
        x = np.zeros(self.mcf.c.size)
        x[self.var_info['non_fix']] = x_sub
        x[self.var_info['fix_up']] = self.mcf.u[self.var_info['fix_up']]
        return x

    def recover_basis_from_sub_basis(self, basis_sub: Basis) -> Basis:
        """ Recover the basis of the current LP from a basis of the sub problem.

        Args:
            basis_sub: A basis of the sub problem.

        Returns:
            The corresponding basis of the original, extended problem. (Extend the sub basis by adding fixed variables.)
        """
        vbasis = -np.ones(self.mcf.c.size, dtype=int)
        vbasis[self.var_info['non_fix']] = basis_sub.vbasis
        vbasis[self.var_info['fix_up']] = -2
        cbasis = basis_sub.cbasis
        return Basis(vbasis, cbasis)

    def rescale_cost(self, factor: float) -> None:
        """ Rescale the objective function of the current MCF.

        Args:
            factor: Scale the objective function with ``1/factor''.
        """
        self.mcf.c = self.mcf.c / factor
        self.c_rescaling_factor = factor

    def recover_obj_val(self, obj_val: float) -> float:
        """ Recover the objective value of the original LP from the objective value of the current LP.

        Args:
            obj_val: The objective value of the scaled MCF.
        """
        return obj_val * self.c_rescaling_factor

    def get_reduced_cost_for_original_mcf(self, y: np.ndarray) -> np.ndarray:
        """ Get the reduced cost for the MCF.

        Args:
            y: The dual solution of the MCF.

        Returns:
            The reduced cost for the original MCF.
        """
        rcost = self.mcf.c - self.mcf.A.T @ y
        rcost[self.basis.vbasis == -2] = -rcost[self.basis.vbasis == -2]
        return rcost

    def check_optimality_condition(self, x: np.ndarray, y: np.ndarray) -> bool:
        """ Check the optimality condition for the current LP with a pair of primal-dual solution (x, y).

        Args:
            x: A primal solution of the MCF.
            y: A dual solution of the MCF.

        Returns:
            True if the optimality condition is satisfied; False otherwise.
        """
        artificial_vars_condition = np.all(
            x[self.artificial_vars] < TOLERANCE_FOR_ARTIFICIAL_VARS) if self.artificial_vars.size > 0 else True
        rcost_condition = np.all(self.get_reduced_cost_for_original_mcf(y) >= -TOLERANCE_FOR_REDUCED_COSTS)
        return artificial_vars_condition and rcost_condition


class OTManager:
    """A class to manage the OT problem and its subproblem in network crossover algorithms.

    Attributes:
        ot (OptimalTransport): The current OT problem.
        ot_sub (OptimalTransport): The subproblem of the current OT problem.
        m (int): The number of constraints in the original OT.
        n (int): The number of variables in the original OT.
        mask_sub_ot (np.ndarray[np.bool_]): The mask of the subproblem of the current OT problem.
        basis (Basis): A feasible basis for the current OT.
        mcf (MinCostFlow): The MCF form corresponding to the current OT.

    """

    ot: OptTransport
    m: int
    n: int
    m_ext: int
    n_ext: int
    mask_sub_ot: np.ndarray
    basis: Basis
    artificial_vars: np.ndarray
    mcf: MinCostFlow

    def __init__(self, ot: OptTransport) -> None:
        self.ot = ot
        self.m = ot.s.size + ot.d.size
        self.n = ot.s.size * ot.d.size
        self.mask_sub_ot = np.zeros(self.n, dtype=bool)
        self.artificial_vars = np.array([])

    def get_mcf(self) -> None:
        """ Get the MCF form of the current OT."""
        self.mcf = self.ot.to_MCF()

    def get_X(self, x: np.ndarray) -> np.ndarray:
        """ Get X from the vector x, such that X[i][j] = the flow from source i to destination j.

        Args:
            x: A solution of the OT in the vector form.

        Returns:
            A flow matrix X.
        """
        return x.reshape((self.ot.s.size, self.ot.d.size))

    def get_sorted_flows(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Get the sorted flows by calculating flow indicators.

        Args:
            x: A (non-vertex) flow solution of the OT problem.

        Returns:
            A tuple of two arrays: the first array is the sorted indices of flows, and the second array is the flow indicators.
        """
        X = self.get_X(x)
        flow_indicators = np.maximum(X / self.ot.s.reshape((self.ot.s.size, 1)), X / self.ot.d.reshape((1, self.ot.d.size)))
        return np.argsort(flow_indicators.flatten())[::-1], flow_indicators.flatten()

    def extend_by_bigM(self, bigM: float) -> None:
        """ Extend the OT problem with bigM method:
            add an artificial supplier and an artificial destination, and add artificial arcs to satisfy all the demands and supplies, with bigM as the cost.

        Args:
            bigM: The bigM value.
        """
        s_appended = np.append(self.ot.s, np.sum(self.ot.d))
        d_appended = np.append(self.ot.d, np.sum(self.ot.s))
        M_appended = np.vstack([
            np.hstack([self.ot.M, bigM * np.ones((self.ot.s.size, 1))]),
            np.hstack([bigM * np.ones((1, self.ot.d.size)), np.array([[0]])])
        ])
        self.mask_sub_ot = np.vstack([
            np.hstack([np.zeros((self.ot.s.size, self.ot.d.size),  dtype=np.bool_), np.ones((self.ot.s.size, 1), dtype=np.bool_)]),
            np.hstack([np.ones((1, self.ot.d.size), dtype=np.bool_), np.array([[1]], dtype=np.bool_)]),
        ])
        # self.artificial_vars all the variables in the self.mask_sub_ot such that it is 1:
        self.artificial_vars = np.where(self.mask_sub_ot.ravel())[0]
        self.ot = OptTransport(s_appended, d_appended, M_appended)

    def add_free_variables(self, ind_free: np.ndarray) -> None:
        """ Add variables to the sub OT problem.

        Args:
            ind_free: The indices of variables to add.
        """
        if self.artificial_vars.size > 0:
            mask_on_original_problem = self.mask_sub_ot[:-1, :-1]
            rows, cols = np.unravel_index(ind_free, mask_on_original_problem.shape)
            mask_on_original_problem[rows, cols] = True
            self.mask_sub_ot[:-1, :-1] = mask_on_original_problem
        else:
            self.mask_sub_ot[ind_free.ravel()] = True

    def set_basis(self, basis: Basis) -> None:
        """ Add a feasible basis to the current OT problem.

        Args:
            basis: The basis to set.
        """
        self.basis = basis

    def recover_x_from_sub_x(self, x_sub: np.ndarray) -> np.ndarray:
        """ Recover the solution of the current OT from the solution of the subproblem.

        Args:
            x_sub: A solution of the subproblem.

        Returns:
            The corresponding solution of the original (extended) OT.
        """
        x = np.zeros(self.ot.s.size * self.ot.d.size)
        x[self.mask_sub_ot.ravel()] = x_sub
        return x

    def recover_basis_from_sub_basis(self, basis_sub: Basis) -> Basis:
        """ Recover the basis of the current OT from the basis of the subproblem.

        Args:
            basis_sub: A basis of the subproblem.

        Returns:
            The corresponding basis of the original (extended) OT.
        """
        vbasis = -np.ones(self.ot.s.size * self.ot.d.size)
        vbasis[self.mask_sub_ot.ravel()] = basis_sub.vbasis
        return Basis(vbasis, basis_sub.cbasis)

    def get_sub_problem(self) -> MinCostFlow:
        """ Get the subproblem of the current OT problem. """
        A = self.mcf.A.tocsc()[:, self.mask_sub_ot.ravel()]
        c = self.ot.M.flatten()[self.mask_sub_ot.ravel()]
        u = self.mcf.u[self.mask_sub_ot.ravel()]
        return MinCostFlow(A=A, b=self.mcf.b, c=c, u=u)

    def solve_subproblem(self, solver: str, solver_settings: SolverSettings) -> Output:
        """ Solve the subproblem of the OT problem. Use network simplex method for Cplex, and default method for other solvers.

        Args:
            solver: The solver to use.
            solver_settings: The settings for the solver.

        Returns:
            The output of the subproblem solution.
        """
        method = "network_simplex" if solver == "CPL" else "default"
        return solve_mcf(self.get_sub_problem(), solver=solver, method=method, warm_start_basis=Basis(self.basis.vbasis[self.mask_sub_ot.ravel()], self.basis.cbasis), settings=solver_settings)

    def recover_obj_val(self, obj_val):
        """ For OT, we do not need to rescale and recover the objective value. """
        return obj_val

    def get_reduced_cost_for_original_OT(self, y: np.ndarray) -> np.ndarray:
        """ Get the reduced cost for the original OT.

        Args:
            y: A dual solution of the original OT.

        Returns:
            The reduced cost for the original OT.
        """
        return self.mcf.c - self.mcf.A.T @ y

    def check_optimality_condition(self, x: np.ndarray, y: np.ndarray) -> bool:
        """ Check the optimality condition for the current OT with a pair of primal-dual solution (x, y).

        Args:
            x: A primal solution of the OT.
            y: A dual solution of the OT.

        Returns:
            True if the optimality condition is satisfied; False otherwise.
        """
        artificial_vars_condition = np.all(x[self.artificial_vars][:-1] < TOLERANCE_FOR_ARTIFICIAL_VARS) if self.artificial_vars.size > 0 else True
        rcost_condition = np.all(self.get_reduced_cost_for_original_OT(y) >= -TOLERANCE_FOR_REDUCED_COSTS)
        return artificial_vars_condition and rcost_condition

    def update_subproblem(self):
        """Nothing else to update for OT. Only need to set the indices of free variables in other functions."""
        pass

    def set_initial_basis(self) -> None:
        """ Set the initial basis for the extended OT:
            all original variables are non-basic, and the artificial variables are basic."""
        vbasis = -np.ones(self.ot.s.size * self.ot.d.size)
        vbasis[self.artificial_vars] = 0
        cbasis = np.concatenate([-np.ones(self.m + 1), np.zeros(1)])
        self.basis = Basis(vbasis, cbasis)

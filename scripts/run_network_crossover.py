import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
from ot import sinkhorn

from smart_crossover import get_project_root
from smart_crossover.filehandling import write_results_to_pickle
from smart_crossover.formats import OptTransport, MinCostFlow
from smart_crossover.network_methods.algorithms import network_crossover
from smart_crossover.solver_caller.caller import SolverSettings
from smart_crossover.solver_caller.gurobi import GrbCaller
from smart_crossover.solver_caller.solving import solve_mcf, solve_ot


def load_opt_transport_instances(folder: str = get_project_root() / "data/ot_mnist") -> List[OptTransport]:
    instances = []
    file_names = sorted(os.listdir(folder))
    for file_name in file_names:
        if file_name.endswith('.ot'):
            file_path = os.path.join(folder, file_name)
            with open(file_path, 'rb') as f:
                ot = pickle.load(f)
            instances.append(ot)
    return instances


def load_min_cost_flow_instances(folder: Path = get_project_root() / "data", problem: str = '') -> List[MinCostFlow]:
    instances = []
    folder = folder / problem
    file_names = sorted(os.listdir(folder))
    for file_name in file_names:
        file_path = os.path.join(folder, file_name)
        if problem == 'goto':
            if file_name.endswith('.mcf') and file_name.startswith('goto_17_8'):
                with open(file_path, 'rb') as f:
                    mcf = pickle.load(f)
                instances.append(mcf)
        if file_name.endswith('.mps'):
            grb_caller = GrbCaller()
            grb_caller.read_model_from_file(file_path)
            mcf = grb_caller.return_mcf()
            mcf.name = file_name.split('.')[0]
            instances.append(mcf)
    return instances


def main(problem: str, test_object: str, barrierTol: float, pricing: str = ''):
    """
    Run network crossover on a set of instances.

    Args:
        problem: problem type, 'mcf', 'goto', or 'ot'
        test_object: test either 'crossover' or 'total'
        barrierTol: set the barrier tolerance
        pricing: set the simplex pricing strategy, see Gurobi documentation for more information

    """

    results = {}

    log_file_path = str(
        get_project_root() / "results" / "logs" / f"{problem}_results_{test_object}_{-round(np.log10(barrierTol))}_{pricing}.log")

    if problem == 'mcf' or problem == 'goto':
        mcf_instances = load_min_cost_flow_instances(problem=problem)

        for mcf in mcf_instances:
            if test_object == 'crossover':
                barrier_output = solve_mcf(mcf, method='barrier', solver='GRB',
                                           settings=SolverSettings(barrierTol=barrierTol, crossover='on',
                                                                   presolve='on', timeLimit=int(3600*1.5),
                                                                   simplexPricing=pricing,
                                                                   log_file=str(get_project_root() / "results" / "logs" / "mcf" / f"{mcf.name}_GRB_{-round(np.log10(barrierTol))}_{pricing}.log")))
                cnet_output = network_crossover(x=barrier_output.x_bar, mcf=mcf, method='cnet_mcf', solver='GRB',
                                                solver_settings=SolverSettings(simplexPricing=pricing, presolve='on'))
                results[mcf.name] = {'cnet': (cnet_output.runtime, cnet_output.iter_count)}

                with open(log_file_path, 'a') as log_file:
                    log_file.write(
                        f"{mcf.name}: "
                        f"cnet_runtime = {cnet_output.runtime}, cnet_iter_count = {cnet_output.iter_count}\n")

    if problem == 'ot':
        ot_instances = load_opt_transport_instances()
        for ot in ot_instances:

            if test_object == 'total':
                # grb_output = solve_ot(ot, method='default', solver='GRB', settings=SolverSettings(presolve='off', timeLimit=1000))
                cpl_output = solve_ot(ot, method='network_simplex', solver='CPL',
                                      settings=SolverSettings(simplexPricing=pricing, presolve='on'))
                start = datetime.now()
                sinkhorn_x = sinkhorn(ot.s, ot.d, ot.M, reg=10, numItermax=1000)
                sinkhorn_runtime = datetime.now() - start
                tnet_output = network_crossover(x=sinkhorn_x.flatten(), ot=ot, method='tnet', solver='CPL',
                                                solver_settings=SolverSettings(simplexPricing=pricing, presolve="on"))
                cnet_output = network_crossover(x=sinkhorn_x.flatten(), ot=ot, method='cnet_ot', solver='CPL',
                                                solver_settings=SolverSettings(simplexPricing=pricing, presolve="on"))
                results[ot.name] = {'grb': (None, None),
                                    'cpl': (cpl_output.runtime, cpl_output.iter_count),
                                    'sinkhorn': sinkhorn_runtime,
                                    'tnet': (tnet_output.runtime, tnet_output.iter_count),
                                    'cnet': (cnet_output.runtime, cnet_output.iter_count)}

                with open(log_file_path, 'a') as log_file:
                    log_file.write(
                        f"{ot.name}: \n"
                        f"   cpl_runtime = {cpl_output.runtime}, cpl_iter_count = {cpl_output.iter_count},\n "
                        f"   sinkhorn_runtime = {sinkhorn_runtime},\n "
                        f"   tnet_runtime = {tnet_output.runtime}, tnet_iter_count = {tnet_output.iter_count},\n "
                        f"   cnet_runtime = {cnet_output.runtime}, cnet_iter_count = {cnet_output.iter_count}\n")

            if test_object == 'crossover':
                barrier_output = solve_ot(ot, method='barrier', solver='GRB',
                                          settings=SolverSettings(barrierTol=barrierTol, crossover='on', presolve='on',
                                                                  simplexPricing=pricing,
                                                                  log_file=str(get_project_root() / "results" / "logs" / "ot" / f"{ot.name}_GRB_{-round(np.log10(barrierTol))}_{pricing}.log")))
                tnet_output = network_crossover(x=barrier_output.x_bar, ot=ot, method='tnet', solver='GRB',
                                                solver_settings=SolverSettings(simplexPricing=pricing, presolve="on"))
                cnet_output = network_crossover(x=barrier_output.x_bar, ot=ot, method='cnet_ot', solver='GRB',
                                                solver_settings=SolverSettings(simplexPricing=pricing, presolve="on"))
                results[ot.name] = {'cnet': (cnet_output.runtime, cnet_output.iter_count), 'tnet': (tnet_output.runtime, tnet_output.iter_count)}


    write_results_to_pickle(results, f"{problem}_results_{test_object}_{-round(np.log10(barrierTol))}_{pricing}.pickle")


if __name__ == "__main__":

    main(problem='mcf', test_object='crossover', barrierTol=1e-4, pricing='SE')

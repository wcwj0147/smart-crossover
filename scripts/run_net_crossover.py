import os
import pickle
from datetime import datetime
from typing import List

import numpy as np
from ot import sinkhorn

from smart_crossover import get_project_root
from smart_crossover.filehandling import write_results_to_pickle
from smart_crossover.formats import OptTransport, MinCostFlow
from smart_crossover.network_methods.algorithms import network_crossover
from smart_crossover.solver_caller.caller import SolverSettings
from smart_crossover.solver_caller.solving import solve_mcf, solve_ot


def load_opt_transport_instances(folder: str = get_project_root() / "data/ot_mnist") -> List[OptTransport]:
    instances = []
    file_names = sorted(os.listdir(folder))
    for file_name in file_names[53:54]:
        if file_name.endswith('.ot'):
            file_path = os.path.join(folder, file_name)
            with open(file_path, 'rb') as f:
                ot = pickle.load(f)
            instances.append(ot)
    return instances


def load_min_cost_flow_instances(folder: str = get_project_root() / "data/mcf") -> List[MinCostFlow]:
    instances = []
    file_names = sorted(os.listdir(folder))
    for file_name in file_names[0:2]:
        if file_name.endswith('.mcf'):
            file_path = os.path.join(folder, file_name)
            with open(file_path, 'rb') as f:
                mcf = pickle.load(f)
            instances.append(mcf)
    return instances


def main(solver: str, problem: str, test_object: str, barrierTol: float):

    results = {}

    if problem == 'mcf':
        mcf_instances = load_min_cost_flow_instances()
        for mcf in mcf_instances:

            if test_object == 'solver':
                solve_mcf(mcf=mcf, solver=solver, method='barrier',
                          settings=SolverSettings(barrierTol=barrierTol, crossover='on', presolve="off",
                                                         log_file=str(get_project_root() / "results" / "logs" / "mcf" /f"{mcf.name}_{solver}_{-round(np.log10(barrierTol))}.log")))

            if test_object == 'crossover':
                barrier_output = solve_mcf(mcf, method='barrier', solver=solver, settings=SolverSettings(barrierTol=barrierTol, crossover='off'))
                cnet_output = network_crossover(x=barrier_output.x_bar, mcf=mcf, method='cnet_mcf', solver=solver,
                                                solver_settings=SolverSettings(presolve='on'))
                results[mcf.name] = {'cnet': (cnet_output.runtime, cnet_output.iter_count)}

    if problem == 'ot':
        ot_instances = load_opt_transport_instances()
        for ot in ot_instances:

            if test_object == 'solver':
                solve_ot(ot, solver=solver, method='barrier',
                         settings=SolverSettings(barrierTol=barrierTol, presolve="off",
                                                 log_file=str(get_project_root() / "results" / "logs" / "ot" / f"{ot.name}_{solver}_{-round(np.log10(barrierTol))}.log")))

            if test_object == 'total':
                grb_output = solve_ot(ot, method='default', solver='GRB', settings=SolverSettings(presolve='off'))
                cpl_output = solve_ot(ot, method='network_simplex', solver='CPL',
                                      settings=SolverSettings(presolve='off'))
                start = datetime.now()
                sinkhorn_x = sinkhorn(ot.s, ot.d, ot.M, 1)
                sinkhorn_runtime = datetime.now() - start
                tnet_output = network_crossover(x=sinkhorn_x.flatten(), ot=ot, method='tnet', solver='CPL', solver_settings=SolverSettings(presolve="off"))
                cnet_output = network_crossover(x=sinkhorn_x.flatten(), ot=ot, method='cnet_ot', solver='CPL', solver_settings=SolverSettings(presolve="off"))
                results[ot.name] = {'grb': (grb_output.runtime, grb_output.iter_count),
                                    'cpl': (cpl_output.runtime, cpl_output.iter_count),
                                    'sinkhorn': sinkhorn_runtime,
                                    'tnet': (tnet_output.runtime, tnet_output.iter_count),
                                    'cnet': (cnet_output.runtime, cnet_output.iter_count)}

            if test_object == 'crossover':
                barrier_output = solve_ot(ot, method='barrier', solver=solver, settings=SolverSettings(barrierTol=barrierTol, crossover='off', presolve='off'))
                tnet_output = network_crossover(x=barrier_output.x, ot=ot, method='tnet', solver=solver, solver_settings=SolverSettings(presolve="off"))
                cnet_output = network_crossover(x=barrier_output.x, ot=ot, method='cnet_ot', solver=solver, solver_settings=SolverSettings(presolve="off"))
                results[ot.name] = {'cnet': (cnet_output.runtime, cnet_output.iter_count), 'tnet': (tnet_output.runtime, tnet_output.iter_count)}

    for instance_name, instance_results in results.items():
        print(f"{instance_name}")
        for method, result in instance_results.items():
            # print in the same line
            print(f"{method}:", end=' ')
            if method == 'sinkhorn':
                print(result)
                continue
            print(result[0], end=' ')
            print(result[1])

    # Todo !!! remember to remove the # in the following line before you run the code !!!
    write_results_to_pickle(results, f"{problem}_results_{solver}_{test_object}_{-round(np.log10(barrierTol))}.pickle")


if __name__ == "__main__":
    for test_object, barrierTol in [('crossover', 1e-2), ('crossover', 1e-8), ('solver', 1e-8)]:
        main(solver='GRB', problem='ot', test_object=test_object, barrierTol=barrierTol)
    # main(solver='GRB', problem='ot', test_object='total', barrierTol=1e-2)

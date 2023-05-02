import os
import pickle
from typing import List


from smart_crossover import get_project_root
from smart_crossover.formats import OptTransport, MinCostFlow
from smart_crossover.network_methods.algorithms import network_crossover
from smart_crossover.network_methods.sinkhorn import sinkhorn
from smart_crossover.solver_caller.caller import SolverSettings


def load_opt_transport_instances(folder: str = get_project_root() / "data/ot_mnist") -> List[OptTransport]:
    instances = []
    file_names = sorted(os.listdir(folder))
    for file_name in file_names[21:22]:
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


def main():
    # mcf_instances = load_min_cost_flow_instances()
    # mcf = mcf_instances[0]
    # # gur_output = solve_mcf(mcf, solver='GRB', presolve=0, method='network_simplex')
    # barrier_output = solve_mcf(mcf, method='barrier', solver='GRB', barrierTol=1e-2)
    # output = network_crossover(x=barrier_output.x_bar, mcf=mcf, method='cnet_mcf', solver='GRB',
    #                            solver_settings=SolverSettings(presolve=-1))

    ot_instances = load_opt_transport_instances()
    ot = ot_instances[0]
    # gur_output = solve_ot(ot, solver='GRB', presolve=-1, method='network_simplex')
    sinkhorn_output = sinkhorn(ot, reg=1)
    # # barrier_output = solve_ot(ot, method='barrier', solver='GRB', barrierTol=1e-2)
    # cpl_barrier_output = solve_ot(ot, method='barrier', solver='CPL', barrierTol=1e-2)
    output = network_crossover(x=sinkhorn_output.x, ot=ot, method='cnet_ot', solver='MSK', solver_settings=SolverSettings(presolve="on"))
    # print(gur_output)
    print(sinkhorn_output)
    # print(barrier_output)
    print(output)


if __name__ == "__main__":
    main()

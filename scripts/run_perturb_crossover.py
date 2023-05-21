import glob
import logging
import os
import datetime

from smart_crossover import get_project_root
from smart_crossover.lp_methods.algorithms import run_perturb_algorithm
from smart_crossover.solver_caller.caller import SolverSettings
from smart_crossover.solver_caller.solving import generate_solver_caller

EXCLUDED_INSTANCES = {'GRB': ['irish-electricity', 'cont11', 'set-cover-model', 'dlr1', 'physiciansched3-3', 'ns1688926', 'neos-5251015', 'thk_48', 'neos-3025225'],
                      'CPL': ['irish-electricity', 'cont11', 'set-cover-model', 'dlr1', 'thk_48', 'thk_63',
                              'neos-3025225', 'ns1687037', 'ns1688926', 'fhnw-binschedule1', 'Linf_520c'],
                      'MSK': ['irish-electricity', 'cont11', 'set-cover-model', 'dlr1', 'Linf_520c', 'ns1687037', 'ns1688926', 'fhnw-binschedule1', 'thk_48', 'thk_63', 'neos-3025225']}


def find_solved_problems(solver: str, method: str) -> list:
    """Find the problems that have been solved by the solver."""
    log_dir = get_project_root() / "results" / "logs" / "lp"
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    solved_problems = []

    for log_file in log_files:
        file_name = os.path.basename(log_file)
        base_name, _ = os.path.splitext(file_name)
        parts = base_name.rsplit("_")
        my_solver = parts[0]
        my_problem = "_".join(parts[1:-1])
        my_method = parts[-1]
        if my_solver == solver and my_method == method:
            solved_problems.append(my_problem)

    return solved_problems


def set_logger_head(solver: str) -> logging.Logger:
    """Configure the logger for the run."""
    logger = logging.getLogger()
    logger.handlers.clear()
    f_name = f"{solver}_" + str(datetime.datetime.now().strftime("%Y-%m-%d-%H_%M")) + ".log"
    handler = logging.FileHandler(filename=f_name, mode='w', encoding='utf-8')
    print(f"Writing log to {f_name}")
    logger.addHandler(handler)
    logger.setLevel(logging.CRITICAL)
    logger.critical("--------------- Start testing perturbation methods. ---------------")
    return logger


def main(solver: str, method: str):
    solved_problems = find_solved_problems(solver, method)

    outputs = {}
    file_path = get_project_root() / "data/lp/presolved/"

    OPTIMALITY_TOL = 1e-6
    BARRIER_TOL = 1e-8

    logger = set_logger_head(solver)

    for file_name in os.listdir(file_path):

        if not file_name.endswith('.mps'):
            continue
        base_name, _ = os.path.splitext(file_name)
        if base_name in EXCLUDED_INSTANCES[solver] or base_name in solved_problems:
            continue

        logger.critical(f"Testing {base_name}...")

        caller = generate_solver_caller(solver,
                                        solver_settings=SolverSettings(
                                            barrierTol=BARRIER_TOL,
                                            optimalityTol=OPTIMALITY_TOL,
                                            presolve="on",
                                            log_file=str(
                                                get_project_root() / f"results/logs/lp/{solver}_{base_name}_ori.log")
                                        ))
        caller.read_model_from_file(str(file_path / file_name))

        if method == 'ori':
            caller.run_barrier()
            continue

        lp = caller.return_genlp()

        perturb_output = run_perturb_algorithm(lp, solver=solver,
                                               perturb_method='primal',
                                               barrierTol=BARRIER_TOL,
                                               optimalityTol=OPTIMALITY_TOL,
                                               log_file=str(
                                                   get_project_root() / f"results/logs/lp/{solver}_{base_name}_{method}.log"))
        outputs[file_name] = perturb_output

    # write_results_to_pickle(outputs, f"lp_results_{solver}_ptb.pickle")


if __name__ == "__main__":

    main(solver="MSK", method="ptb")

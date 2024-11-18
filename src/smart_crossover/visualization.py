"""
This module is used in the result analysis notebooks to visualize the results.
See `notebooks/network_crossover_analysis.ipynb` and `notebooks/lp_crossover_analysis.ipynb for usage examples.
"""

import re
import os
import glob
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker

from smart_crossover import get_project_root
from smart_crossover.filehandling import read_results_from_pickle


def get_main_info_df_lp(log_dir: Path, solver: str, ptb_method: str) -> pd.DataFrame:
    # Get list of all log files
    log_files = glob.glob(os.path.join(log_dir, "*.log"))

    # Initialize an empty dict to store data
    data = {}

    # Define regular expressions to match the lines with the information we need
    group_ind = 2 if solver == 'GRB' else 1
    if solver == 'GRB':
        re_barrier = re.compile(r"Barrier solved model in (\d+) iterations and (\d+\.\d+) seconds")
        re_crossover_ori = re.compile(r"Solved in (\d+) iterations and (\d+\.\d+) seconds")
        re_ptb = re.compile(r"Solved in (\d+) iterations and (\d+\.\d+) seconds")
        re_ptb_simplex = re.compile(r"Solved in (\d+) iterations and (\d+\.\d+) seconds")
    elif solver == 'MSK':
        re_barrier = re.compile(r"Optimizer terminated. Time: (\d+\.\d+)")
        re_crossover_ori = re.compile(r"Basis identification terminated. Time: (\d+\.\d+)")
        re_ptb = re.compile(r"Optimizer terminated. Time: (\d+\.\d+)")
        re_ptb_simplex = re.compile(r"Simplex optimizer terminated. Time: (\d+\.\d+)")
    else:
        raise ValueError(f"Invalid solver: {solver}")

    # Loop over each log file
    for log_file in log_files:

        # Extract information from file name
        file_name = os.path.basename(log_file)
        base_name, _ = os.path.splitext(file_name)
        parts = base_name.rsplit("_")  # split the base name into parts from the end

        if parts[0] != solver:
            continue

        msk_flag = False if solver == 'MSK' else True

        method = parts[-1]  # remove .log extension
        problem = "_".join(parts[1:-1])
        # Initialize variables to store the runtimes
        runtime_barrier = None
        runtime_crossover_ori = None
        runtime_ptb = None
        runtime_ptb_simplex = None

        if problem not in data.keys():
            data[problem] = {}

        if method == 'ori':
            with open(log_file, 'r') as f:
                for line in f:
                    if runtime_barrier is None:
                        match = re_barrier.search(line)
                        if match:
                            runtime_barrier = float(match.group(group_ind))
                    if runtime_crossover_ori is None:
                        match = re_crossover_ori.search(line)
                        if match:
                            runtime_crossover_ori = float(match.group(group_ind))

            if solver == 'GRB':
                data[problem]['Barrier(ori)'] = runtime_barrier if runtime_barrier is not None else None
                data[problem]['Crossover(ori)'] = runtime_crossover_ori - runtime_barrier if runtime_crossover_ori is not None and runtime_barrier is not None else None
            elif solver == 'MSK':
                data[problem]['Crossover(ori)'] = runtime_crossover_ori if runtime_crossover_ori is not None and runtime_barrier is not None else None
                data[problem]['Barrier(ori)'] = runtime_barrier - runtime_crossover_ori if runtime_crossover_ori is not None and runtime_barrier is not None else None

        if method == ptb_method:
            with open(log_file, 'r') as f:
                for line in f:
                    if runtime_ptb is None and msk_flag:
                        match = re_ptb.search(line)
                        if match:
                            runtime_ptb = float(match.group(group_ind))
                    else:
                        match = re_ptb_simplex.search(line)
                        if match:
                            runtime_ptb_simplex = float(match.group(group_ind))
                    if "Optimizer terminated" in line:
                        msk_flag = True
            data[problem]['Perturb'] = runtime_ptb if runtime_ptb is not None else None
            data[problem]['Simplex(ptb)'] = runtime_ptb_simplex if runtime_ptb_simplex is not None else None

    # Convert data list to pandas DataFrame
    df = pd.DataFrame(data)
    df = df.transpose()
    df = df[["Barrier(ori)", "Crossover(ori)", "Perturb", "Simplex(ptb)"]]

    # Sort by ratio and drop off None
    df['Ratio'] = df['Crossover(ori)'] / df['Barrier(ori)']
    df = df.sort_values(by='Crossover(ori)', ascending=False)
    df = df.drop(columns='Ratio')
    # df = df.dropna()
    return df


def get_additional_info_df_lp(log_file: Path) -> pd.DataFrame:
    info = {}
    with open(log_file, 'r') as file:
        for line in file:
            if "Testing" in line:
                problem_name = line.split("Testing ")[1].strip("...\n")
                info[problem_name] = {}
            elif "Scale factor" in line:
                scale_factor = float(re.findall(r"\d+\.\d+", line)[0])
                info[problem_name]["Scale factor"] = scale_factor
            elif "Primal-dual" in line:
                match = re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line)
                if match:
                    gap = float(match[0])
                    info[problem_name]["Relative gap"] = gap
            elif "fixed variables" in line:
                fixed_var = float(re.findall(r"\d+", line)[0])
                info[problem_name]["Fixed Var"] = fixed_var
            elif "fixed constraints" in line:
                fixed_constr = float(re.findall(r"\d+", line)[0])
                info[problem_name]["Fixed Constr"] = fixed_constr

    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame(info)
    df = df.transpose()

    return df


def get_merged_df_lp(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    df1.reset_index(inplace=True)
    df2.reset_index(inplace=True)
    df1.rename(columns={'index': 'Problem'}, inplace=True)
    df2.rename(columns={'index': 'Problem'}, inplace=True)
    merged_df = df1.merge(df2, on='Problem')
    return merged_df


def get_complete_df_lp(log_dir: Path, log_file: Path, solver: str, ptb_method: str, simplified: bool = False) -> pd.DataFrame:
    df1 = get_main_info_df_lp(log_dir, solver, ptb_method)
    df2 = get_additional_info_df_lp(log_file)
    df = get_merged_df_lp(df1, df2)

    # drop the rows with nan in 'Crossover_ori':
    # df = df.dropna(subset=['Crossover(ori)'])
    df['Ptime'] = df['Perturb'] + df['Simplex(ptb)']
    # if df['Relative gap'] < 1e-8 Ptime = df['Perturb']
    df.loc[df['Relative gap'] < 1e-8, 'Ptime'] = df['Perturb']

    df['PDtime'] = df['Perturb'] + df['Simplex(ptb)']

    def format_sci(num):
        return "{:.1e}".format(num)

    df[['Relative gap']] = df[['Relative gap']].applymap(format_sci)

    if simplified:
        df = df.drop('Scale factor', axis=1)
        df['Fixed Var'] = df['Fixed Var'].round(0).astype(int)
        df = df.drop('Fixed Var', axis=1)
        df = df.drop('Fixed Constr', axis=1)
        df = df.drop('Simplex(ptb)', axis=1)
        #df = df.drop('Projector', axis=1)
    return df


def calculate_average_improvement_lp(df: pd.DataFrame, ptb_method: str) -> Tuple[float, float, float]:
    df['parallel'] = df[['Ptime', 'Crossover(ori)']].min(axis=1)

    # substitue NAN with 3600
    df['parallel'] = df['parallel'].fillna(3600)
    df['Ptime'] = df['Ptime'].fillna(3600)
    # # delete row with problem == neos-3025225
    # df = df[df['Problem'] != 'neos-3025225']
    average = (df['Ptime']).prod() ** (1 / len(df))
    average_solver = (df['Crossover(ori)']).prod() ** (1 / len(df))
    average_parralel = (df['parallel']).prod() ** (1 / len(df))
    # count how many rows have 'Perturb_method' < 'Crossover(ori)':
    num_improved = df[df['Ptime'] < df['Crossover(ori)']]['Ptime'].count()
    print(f'Number of improved problems: {num_improved}')
    return average, average_solver, average_parralel


def plot_df_lp(df: pd.DataFrame, path: str = '') -> None:
    # fill nan in df['Perturb'] with 3600:
    df['Perturb'] = df['Perturb'].fillna(3600)
    # fill <0.2 value with 0.2:
    df.loc[df['Perturb'] < 0.15, 'Perturb'] = 0.15
    df.loc[df['Crossover(ori)'] < 0.15, 'Crossover(ori)'] = 0.15
    # fill nan in df['Relative Gap'] with nan:
    df['Relative gap'] = df['Relative gap'].fillna(np.nan)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    color1, color2, color3 = 'Crimson', 'DodgerBlue', 'Goldenrod'
    plt.style.use('seaborn-whitegrid')
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['text.usetex'] = True

    # Create the double variable bar chart
    bar_width = 0.35  # Width of the bars
    index = np.arange(len(df))  # The x locations for the groups

    # Applying logarithmic scale to the left y-axis
    ax1.set_yscale('log')

    bars1 = ax1.bar(index, df['Perturb'], bar_width, color=color1, alpha=0.8, label='Perturbation Crossover')
    bars2 = ax1.bar(index + bar_width, df['Crossover(ori)'], bar_width, color=color2, alpha=0.8, label='Gurobi Crossover')

    # Create a second y-axis for the points chart
    ax2 = ax1.twinx()

    # Plot individual points for the relative gap
    relative_gap = pd.to_numeric(df['Relative gap'], errors='coerce')
    relative_gap = relative_gap.replace(0, 1e-15)
    points = (-np.log10(relative_gap)).values
    ax2.scatter(df.index, points, color=color3, label='Relative Objective Gap')

    # Set axis labels
    ax1.set_xlabel(r'$\it{optLP}$ benchmark problems')
    ax1.set_ylabel('Running Time (seconds)', color=color1)
    ax2.set_ylabel('Relative Gap', color=color3)

    tick_values = [-16, -12, -8, -4, 0, 4, 8, 12, 16, 20]  # Define the tick values
    tick_labels = [f'1e{-val}' for val in tick_values]  # Define the tick labels
    ax2.set_yticks(tick_values)
    ax2.set_yticklabels(tick_labels)

    # Set axis tick labels color and rotation
    ax1.tick_params(axis='y', colors=color1)
    ax2.tick_params(axis='y', colors=color3)

    # Set the x-ticks to be in the middle of the grouped bars
    ax1.set_xticks([])
    # ax1.set_xticks(index + bar_width / 2)
    # ax1.set_xticklabels(df.index, rotation=45)

    # Adding horizontal line on the secondary y-axis
    ax2.axhline(y=8, color=color3, linestyle='--', alpha=0.5)

    ax1.set_ylim([10**(-1), 10**4])
    ax2.set_ylim([-16, 20])

    # Disable the grid for the secondary y-axis
    ax2.grid(False)

    # Enable the grid for the primary y-axis
    ax1.grid(True)

    # Adding legends
    ax1.legend(loc='upper left', ncol=2, frameon=True)
    ax2.legend(loc='upper right', frameon=True)

    # Adjust the layout
    fig.tight_layout()

    # Save the plot if a path is provided
    if path:
        fig.savefig(path)

    # Show the plot
    plt.show()


def plot_df_lp_old(df: pd.DataFrame, path: str = '') -> None:
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color1, color2 = 'Indianred', 'SteelBlue'
    plt.style.use('seaborn-whitegrid')
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['text.usetex'] = True

    # Create the bar chart
    ratio = df['Perturb'] / df['Crossover(ori)']
    ratio = pd.to_numeric(ratio, errors='coerce')
    bars = (-np.log10(ratio)).values
    ax1.bar(df.index, bars, color=color1, alpha=1.0)

    # Create a second y-axis for the line chart
    ax2 = ax1.twinx()

    # Create the line chart
    relative_gap = pd.to_numeric(df['Relative gap'], errors='coerce')
    relative_gap = relative_gap.replace(0, 1e-15)
    points = (-np.log10(relative_gap)).values
    ax2.scatter(df.index, points, color=color2, label='Relative Gap')

    # Set axis labels
    ax1.set_xlabel(r'$\it{optLP}$ benchmark problems')
    ax1.set_ylabel('Running time Ratio (-log scale)', color=color1)
    ax2.set_ylabel('Relative Gap (-log scale)', color=color2)

    # Set axis tick labels color
    ax1.tick_params(axis='y', colors=color1)
    ax2.tick_params(axis='y', colors=color2)

    ax1.set_ylim([-1.5, 4])
    ax2.set_ylim([-16, 20])
    # Hide x tick labels
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax2.axhline(y=8, color='tab:orange', linestyle='--', alpha=0.5)

    ax2.grid(False)

    # ax1.set_xticks(np.arange(0, len(df.index), 5))
    ax1.grid(True)

    # It's also a good practice to update the axes layout after making adjustments
    fig.tight_layout()

    # Save the plot
    if path:
        fig.savefig(get_project_root() / 'results' / path)

    # Show the plot
    plt.show()


def get_grb_crossover_df_ot(log_dir: str, problem: str, precision: int, pricing: str,) -> pd.DataFrame:
    # Get list of all log files
    log_files = glob.glob(os.path.join(log_dir, "*.log"))

    # Initialize an empty dict to store data
    data = {}

    re_barrier = re.compile(r"Barrier (solved model in|performed) (\d+) iterations (and|in) (\d+\.\d+) seconds")
    re_crossover = re.compile(r"Solved in (\d+) iterations and (\d+\.\d+) seconds")

    for log_file in log_files:

        # Extract information from file name
        file_name = os.path.basename(log_file)
        base_name, _ = os.path.splitext(file_name)
        parts = base_name.rsplit("_")  # split the base name into parts from the end

        if parts[-2] != str(precision) or parts[-1] != pricing:
            continue
        if problem == 'ot':
            # name is a string connected by parts[1:-2]
            name = "_".join(parts[1:3])
            data[name] = {}
        else:
            name = "_".join(parts[:-3])
        if problem != 'goto' and name[0:4] == 'goto':
            continue

        with open(log_file, 'r') as f:
            for line in f:
                match = re_barrier.search(line)
                if match:
                    runtime_barrier = float(match.group(4))
                match = re_crossover.search(line)
                if match:
                    runtime_crossover = float(match.group(2))
                    iter_counts = int(match.group(1))

            data[name] = {'barrier': runtime_barrier, 'crossover': runtime_crossover - runtime_barrier, 'iter': iter_counts}

    df = pd.DataFrame(data)
    df = df.transpose()
    df = df[["barrier", "crossover", "iter"]]

    # sort df's rows by `name`
    df.sort_index(inplace=True)

    return df


def get_df_from_results_ot(data: dict, problem: str) -> pd.DataFrame:
    df_data = {}
    for instance, methods in data.items():
        index = instance[6:] if problem == 'ot' else instance
        for method, results in methods.items():
            if method == 'sinkhorn':
                if f"{method}_runtime" not in df_data:
                    df_data[f"{method}_runtime"] = {}
                df_data[f"{method}_runtime"][index] = results.total_seconds()
                continue
            if f"{method}_runtime" not in df_data:
                df_data[f"{method}_runtime"] = {}
            if f"{method}_iter" not in df_data:
                df_data[f"{method}_iter"] = {}
            df_data[f"{method}_runtime"][index] = results[0].total_seconds() if results[0] is not None else None
            df_data[f"{method}_iter"][index] = results[1] if results[1] is not None else None

    df = pd.DataFrame.from_dict(df_data)
    return df


def get_crossover_comparison_df_net(log_dir: str, results_path: Path, precision: int, pricing: str, problem: str) -> pd.DataFrame:
    df_solver = get_grb_crossover_df_ot(log_dir, problem, precision, pricing)
    df_net = get_df_from_results_ot(read_results_from_pickle(str(results_path / f"{problem}_results_crossover_{precision}_{pricing}.pickle")), problem=problem)
    df = get_merged_df_lp(df_solver, df_net)
    df['group'] = df.Problem.str.split('_').str[0]
    # df_grouped = df.groupby('group').mean() -> change to geometric mean
    df_grouped = df.groupby('group').agg(lambda x: np.exp(np.log(x+0.01).mean()))
    print(f"Table with {problem} results for {pricing} pricing and {precision} precision created, with {len(df_net)} instances.")
    df_grouped = df_grouped.round(2)
    return df_grouped


def get_total_time_comparison_df_net(results_path: Path, precision: int, pricing: str) -> pd.DataFrame:
    data = read_results_from_pickle(str(results_path / f"ot_results_total_{precision}_{pricing}.pickle"))
    df = get_df_from_results_ot(data, problem='ot')
    df['tnet_runtime'] = df['tnet_runtime'] + df['sinkhorn_runtime']
    df['cnet_runtime'] = df['cnet_runtime'] + df['sinkhorn_runtime']
    df.drop(columns=['sinkhorn_runtime'], inplace=True)
    df['group'] = df.index.str.split('_').str[0]
    df_grouped = df.groupby('group').agg(lambda x: np.exp(np.log(x+0.01).mean()))
    df_grouped = df_grouped.round(2)
    # df_grouped.drop(columns=['cpl_iter', 'grb_iter', 'tnet_iter', 'cnet_iter'], inplace=True)
    return df_grouped

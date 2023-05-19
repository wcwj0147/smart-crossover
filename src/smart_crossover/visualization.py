import re
import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def get_main_info_df(log_dir: Path, solver: str) -> pd.DataFrame:
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
    elif solver == 'CPL':
        re_barrier = re.compile(r"Barrier time = (\d+\.\d+) sec.")
        re_crossover_ori = re.compile(r"Total crossover time = (\d+\.\d+) sec.")
        re_ptb = re.compile(r"Total time on 10 threads = (\d+\.\d+) sec.")
        re_ptb_simplex = re.compile(r"Total time on 10 threads = (\d+\.\d+) sec.")
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

        if method == 'ptb':
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
    df = df.sort_values(by='Ratio', ascending=False)
    df = df.drop(columns='Ratio')
    df = df.dropna()
    return df


def get_additional_info_df(log_file: Path) -> pd.DataFrame:
    info = {}
    with open(log_file, 'r') as file:
        for line in file:
            if "Testing" in line:
                problem_name = line.split("Testing ")[1].strip("...\n")
                info[problem_name] = {}
            elif "Projector" in line:
                projector = float(re.findall(r"\d+\.\d+", line)[0])
                info[problem_name]["Projector"] = projector
            elif "Scale factor" in line:
                scale_factor = float(re.findall(r"\d+\.\d+", line)[0])
                info[problem_name]["Scale factor"] = scale_factor
            elif "Primal-dual" in line:
                match = re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line)
                if match:
                    projector = float(match[0])
                    info[problem_name]["Relative gap"] = projector
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


def get_merged_df(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    df1.reset_index(inplace=True)
    df2.reset_index(inplace=True)
    df1.rename(columns={'index': 'Problem'}, inplace=True)
    df2.rename(columns={'index': 'Problem'}, inplace=True)
    merged_df = df1.merge(df2, on='Problem')
    return merged_df


def get_complete_df(log_dir: Path, log_file: Path, solver: str, simplified: bool = False) -> pd.DataFrame:
    df1 = get_main_info_df(log_dir, solver)
    df2 = get_additional_info_df(log_file)
    df = get_merged_df(df1, df2)

    def format_sci(num):
        return "{:.1e}".format(num)

    df[['Relative gap']] = df[['Relative gap']].applymap(format_sci)

    if simplified:
        df = df.drop('Scale factor', axis=1)
        df['Fixed Var'] = df['Fixed Var'].round(0).astype(int)
        df = df.drop('Fixed Var', axis=1)
        df = df.drop('Fixed Constr', axis=1)
        df = df.drop('Projector', axis=1)
    return df


def plot_df(df: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color1, color2 = 'tab:blue', 'tab:orange'

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
    line = (-np.log10(relative_gap)).values
    ax2.plot(df.index, line, color=color2)

    # Set axis labels
    ax1.set_xlabel('Problem')
    ax1.set_ylabel('Runtime Ratio (log scale)', color=color1)
    ax2.set_ylabel('Relative Gap (log scale)', color=color2)

    # Set axis tick labels color
    ax1.tick_params(axis='y', colors=color1)
    ax2.tick_params(axis='y', colors=color2)

    ax1.set_ylim([-2, 4])
    ax2.set_ylim([-16, 16])

    # It's also a good practice to update the axes layout after making adjustments
    fig.tight_layout()

    # Show the plot
    plt.show()

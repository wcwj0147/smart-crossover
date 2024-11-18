import os
import glob
import pickle

import numpy as np
import scipy.sparse as sp

from smart_crossover.formats import MinCostFlow
from smart_crossover import get_data_dir_path


def parse_min_file(file_path: str, name: str) -> MinCostFlow:
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Find the line starting with 'p' and extract the number of nodes and arcs
    p_line = next(line for line in lines if line.startswith('p'))
    _, _, num_nodes, num_arcs = p_line.strip().split()
    num_nodes, num_arcs = int(num_nodes), int(num_arcs)

    b = np.zeros(num_nodes)

    node_lines = [line for line in lines if line.startswith('n')]
    for line in node_lines:
        node_data = line.strip().split()
        node_idx, node_b = int(node_data[1]), int(node_data[2])
        b[node_idx - 1] = node_b

    arc_lines = [line for line in lines if line.startswith('a')]
    arcs_data = [list(map(int, line.strip().split()[1:])) for line in arc_lines]
    A = sp.lil_matrix((num_nodes, num_arcs), dtype=int)
    c = np.zeros(num_arcs)
    u = np.zeros(num_arcs)

    for i, (tail, head, lower, upper, cost) in enumerate(arcs_data):
        A[head - 1, i] = -1
        A[tail - 1, i] = 1
        c[i] = cost
        u[i] = upper

    return MinCostFlow(A=A.tocsr(), b=b, c=c, u=u, name=name)


def main():
    input_folder = get_data_dir_path() / "goto"
    output_folder = get_data_dir_path() / "goto"

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read all .min files from the input folder
    min_files = glob.glob(os.path.join(input_folder, "*.min"))

    for min_file in min_files:
        # Save the MinCostFlow instance to the output folder with the same name (but with .mcf extension)
        basename = os.path.basename(min_file)
        new_name = os.path.splitext(basename)[0] + ".mcf"
        output_file = os.path.join(output_folder, new_name)

        # Process the .min file to a MinCostFlow instance
        min_cost_flow = parse_min_file(min_file, basename[:-4])

        with open(output_file, 'wb') as file:
            pickle.dump(min_cost_flow, file)


if __name__ == "__main__":
    main()

Smart crossover algorithms.

## Getting Started with the Problem

Check out the paper [From an Interior Point to a Corner Point: Smart Crossover](https://arxiv.org/abs/2102.09420/).
This package provides methods described in the paper.


## Description

The goal of this software is to implement the crossover algorithms proposed in the paper.

### Structure of 'src'

`smart_crossover.formats` contains the problem structures used in the project, including LP, min-cost flow, and optimal transport.

`smart_crossover.parameters` collects the parameters used in the algorithms.

`smart_crossover.output` and `smart_crossover.timer` are output and timer classes.

The two main algorithms included:
the **network crossover** methods in 
`smart_crossover.network_methods` and **perturbation crossover**
methods in `smart_crossover.lp_methods`. 

In both cases, there is a `algorithm` module that contains the main algorithm
and a `managemer` module that provides utilities for managing the algorithm.



## Building

Follow these steps to set up the environment:

1. Install Miniconda

You need to have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed to manage the environment. Download and install Miniconda by following the [official instructions](https://docs.conda.io/en/latest/miniconda.html).

2. Clone the repository:
   ```
   git clone https://github.com/INFORMSJoC/2022.0291.git
   cd smart-crossover
   ```
3. Create the environment using `environment.yml`:
   ```
   conda env create -f environment.yml -n smart_crossover
   ```
   **An important remark**: we use Gurobi 9.0.3, Cplex 12.9.0, and Mosek 9.3.14 in this project. If you have a different version of these solvers, you may need to modify the `environment.yml` file to match your version.
   
4. Activate the environment:
   ```
   conda activate smart_crossover
   ```

To use the environment with Jupyter Notebook, install the environment as a Jupyter kernel when the environment is activated:

Install the Jupyter kernel:
  ```
  python -m ipykernel install --user --name=smart_crossover  
  ```


## Usage example

Open `/notebooks/example.ipynb`, activate the environment, and run the notebook to test the algorithm.


## How to reproduce the results

1. Prepare the data:
   Follow the instruction in `data/README.md` to get the initial data.

2. Build the environment:
   Follow the instruction in the `Building` section to build the environment.

3. Run the experiments:
   For the network crossover results, run the script `scripts/run_network_crossover.py`.
   For the perturbation crossover results, run the script `scripts/run_perturb_crossover.py`.

4. Analyze the results:
   Follow the instruction in `notebooks/network_result_analysis.ipynb` and `notebooks/perturb_result_analysis.ipynb` to analyze the results.

Smart crossover algorithms.

Getting Started with the Problem
================================

Check out the paper [From an Interior Point to a Corner Point: Smart Crossover](https://arxiv.org/abs/2102.09420/).
This package provides methods described in the paper.


Structure of the Code
=====================
There are two major algorithms included:
the network crossover methods in 
`smart_crossover.network_methods` and perturbation crossover
methods in `smart_crossover.lp_methods`.

The problem types that can be solved are listed in `smart-crossover.format`.

Installation
==============

### Setting Up the Environment

To set up the required environment, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/wcwj0147/smart-crossover.git
   cd smart-crossover
   ```
2. Create the environment using `environment.yml`:
   ```
   conda env create -f environment.yml -n smart-crossover
   ```
3. Activate the environment:
   ```
   conda activate smart-crossover
   ```

### Jupyter Notebook Integration

To use the environment with Jupyter Notebook, install the environment as a Jupyter kernel when the environment is activated:

1. Install the Jupyter kernel:
  ```
  python -m ipykernel install --user --name=smart-crossover  
  ```
2. Open the Jupyter notebook and select the environment `smart-crossover`.


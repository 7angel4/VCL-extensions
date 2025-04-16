# Variational Continual Learning for Regression with Exponential Prior

**Candidate number: 1088143**


This project implements two methodological extensions to the **Variational Continual Learning (VCL)** ([*Nguyen et al. (2017)*](https://arxiv.org/pdf/1710.10628)) framework: 
1) Adapting VCL to **regression** tasks, and 
2) Using an **exponential prior** as opposed to the standard Gaussian prior.

This repository includes our implementations of the extended VCL framework, a reproduction of the SplitMNIST experiments (in jupyter notebooks) using this extended version, as well as a Python CLI for users to run those VCL experiments.

## Code attribution
Our implementation is partially adapted from the original authors' VCL implementation, which is available at: https://github.com/nvcuong/variational-continual-learning/. In particular, we reproduced their coreset selection algorithms to ensure faithful replication of their experimental setup. The remainder, however, is mostly based on our own understanding of their presented work in [*Nguyen et al. (2017)*](https://arxiv.org/pdf/1710.10628).


## Installation

To run the code, ensure you have the following dependencies installed:

1. **Python (3.7 or higher)**: This project is compatible with Python 3.7 or later.
2. **Required Python Packages**: These packages can be installed using `pip`.

```bash
pip install -r requirements.txt
```

## Usage

### Command-Line Interface (CLI)

The CLI supports three subcommands:

1. **run-custom**: Runs experiments using user-specified YAML configuration files.
2. **run-default**: Runs the default experiment involving 3 models: `Vanilla` (non-variational baseline with no coreset), `GaussianVCL` (with  standard Gaussian prior) and `ExpVCL` (with exponential prior). The last two uses K-center coresets of size 200.

### Example Commands:

1. **Run with user-provided YAML files:**
   ```bash
   python run.py run-custom --configs ./example_config.yaml --task_type regression --results_type final
   ```

   This command will run experiments using the specified YAML configurations (`./example_config.yaml`), performing a regression task, reporting final test results, and
   plotting those results.


3. **Run default experiment:**
   ```bash
   python run.py run-default --task_type classification --results_type both
   ```

   This command will run the default classification experiment, reporting both final and lifetime test results, without plotting those results.
   

## CLI Arguments

- `--config <path_to_yaml_file>`: Path to the YAML configuration file. This is a required argument unless all parameters are provided via the command line.
  
- `--task_type <task_type>`: Type of task to perform. Options are:
  - `classification`
  - `regression`
  
- `--results_type <results_type>`: Type of results to report. Options are:
  - `final`: Report mean final results (default).
  - `lifetime`: Report mean lifetime results.
  - `both`: Report both types of results above.
  
- `--show_vanilla`: If specified, include a baseline `VanillaNN` model in the experiment.
  
- `--print_progress`: If specified, print the training progress during the experiment.

- `--plot`: If specified, generate plots for those results (of `results_type`).

- `--filter <filter_string>`: Substring filter for model names during reporting and plotting (i.e., can suppress display of models you don't want to see).

- `--configs <config_file1> <config_file2> ...`: List of paths to YAML configuration files (for `run-custom` only).

## Repository Structure

- `alg/`: Python module for model/algorithm implementations, training, and testing.
- `utils/`: Python module for miscellaneous utility functions (e.g., visualisations).
- `config.py`: Python module for the experiment config (e.g., task type, prior for the model parameters, coreset size and selection algorithm, training hyperparmeters...).
- `experiments.py`: Python module for the VCL experiments.
- `run.py`: The main entry point for the CLI.
- `example_config.yaml`: Sample configuration file for the experiment.
- `requirements.txt`: List of dependencies required for the project.
- `data/`: Datasets (MNIST) for the experiment.
- `results/`: Sample experimental results (numerical outputs and plots).
- `jupyter_notebooks/`: Original Jupyter notebooks where our experiments were conducted.

## License

`Variational Continual Learning for Regression with Exponential Prior` is released under the MIT License.

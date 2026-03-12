# BP-DNN Hyperparameter Optimization Workflow

This framework is designed to optimize hyperparameters for the BP-DNN model used to predict activation energy (Ea) in biomass pyrolysis models. The workflow is designed to systematically evaluate multiple neural network configurations and identify the optimal model architecture and training parameters.

## Workflow Overview

The complete hyperparameter optimization workflow consists of the following sequential steps:

1. **Data Preparation**: Load and prepare data from the matlab_src directory
2. **Configuration Generation**: Generate hyperparameter configurations based on the hyperparameters.json file
3. **Hyperparameter Optimization**: Train and evaluate models with different hyperparameter configurations
4. **Results Analysis**: Analyze performance metrics across all configurations
5. **Best Model Generation**: Create and save the best performing model
6. **Results Visualization**: Generate visualizations to compare model performances
7. **Model Verification**: Compare optimized models with the baseline model

## Directory Structure

```
HyperparametersOptimization/
├── hyperparameters.json           # Hyperparameter configuration file
├── hyperParamOptimization.sh      # Cluster job submission script
├── main_workflow.m                # Main workflow orchestration script
├── README.md                      # Documentation
├── matlab_src/                    # Original source code and data
│   ├── bpDNN4Ea.m                 # Original BP-DNN implementation
│   ├── CopyrolysisFeedstock.mat   # Feedstock data
│   ├── nn*.m                      # Neural network utility functions
│   ├── SampleIn_*.mat             # Sample input data
│   └── Figures/                   # Original figures for reference
├── scripts/                       # Optimization workflow scripts
│   ├── analyzeResults.m           # Analyze optimization results
│   ├── countConfigurations.m      # Count total hyperparameter combinations
│   ├── generate_configurations.m  # Generate hyperparameter configurations
│   ├── generateBestModel.m        # Generate the best model from results
│   ├── hyperParamOptimization.m   # Core hyperparameter optimization function
│   ├── prepare_data.m             # Data preparation function
│   ├── runOptimization.m          # Optimization execution script
│   ├── setEarlyStoppingThreshold.m # Set threshold for early stopping
│   ├── trainWithEarlyStopping.m   # Training with early stopping capability
│   └── verifyModels.m             # Verify models against baseline
├── data/                          # Prepared data (created during workflow)
├── configs/                       # Hyperparameter configurations (created during workflow)
├── results/                       # Optimization results (created during workflow)
├── best_model/                    # Best model output (created during workflow)
└── figures/                       # Result visualizations (created during workflow)
```

### 3. Run Optimization

## Key Features

1. **Comprehensive Hyperparameter Space**: Explore combinations of learning rates, momentum coefficients, hidden layer structures, transfer functions, and training parameters.

2. **Early Stopping**: Implements adaptive early stopping for clearly underperforming configurations to save computational resources.

3. **Model Verification**: Compares the optimized model against the baseline model to quantify improvements.

4. **Cluster Compatibility**: Fully compatible with the NCSA Illinois Campus Cluster (ICC) for distributed computing.

5. **Visualization**: Generates publication-quality figures similar to those in the original matlab_src/Figures directory.

## Usage Instructions

### Local Execution

1. Run the main workflow script:
   ```matlab
   main_workflow
   ```

2. Follow the interactive prompts to run the different steps of the workflow.

### Cluster Execution

1. Prepare the job submission script:
   ```bash
   chmod +x hyperParamOptimization.sh
   ```

2. Submit jobs to the cluster:
   ```bash
   ./hyperParamOptimization.sh submit
   ```

3. After completion, analyze results:
   ```bash
   ./hyperParamOptimization.sh analyze
   ```

4. Verify models against baseline:
   ```bash
   ./hyperParamOptimization.sh verify
   ```

## Hyperparameter Configuration

The `hyperparameters.json` file contains the hyperparameter space to explore. Example configuration:

```json
{
  "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.7, 0.85, 1.0],
  "momentum": [0.8, 0.85, 0.9, 0.95, 0.97, 0.99],
  "hidden_layers": [
    [27, 27, 27],
    [27, 27, 27, 27],
    [27, 27, 27, 27, 27],
    [27, 54, 27],
    [54, 27, 54],
    [54, 54, 54]
  ],
  "transfer_functions": [
    ["logsig", "logsig", "logsig", "logsig", "logsig", "purelin"],
    ["tansig", "tansig", "tansig", "tansig", "tansig", "purelin"],
    ["logsig", "tansig", "logsig", "tansig", "logsig", "purelin"]
  ],
  "lr_increase_factor": [1.01, 1.03, 1.05, 1.1, 1.15],
  "lr_decrease_factor": [0.6, 0.7, 0.8, 0.9],
  "division_ratio": [
    [0.8, 0.1, 0.1],
    [0.9, 0.05, 0.05]
  ],
  "max_fail": [6, 10, 15, 20],
  "max_epochs": [2000, 4000, 6000, 8000, 10000]
}
```

## SLURM Job Configuration

Hyperparameter optimization jobs are submitted as SLURM array jobs, with each configuration running as an independent task. The analysis job requires more memory and runs using a separate SLURM script.

## Script Workflow and Cluster Compatibility

The `hyperParamOptimization.sh` script submitted to the UIUC NCSA ICC supercomputing platform has been optimized with the following features to execute the entire workflow:

1. **Path Consistency**: Ensures all scripts use the same working directory paths, avoiding path conflicts between different scripts
2. **Early Stopping Mechanism**: Adds intelligent early stopping that can save computational resources on the cluster
3. **Model Verification**: Integrates functionality to compare with the baseline model for validation of optimization effects
4. **Error Handling**: Improves error handling and logging, making tasks more robust on the cluster
5. **Resource Management**: Optimized settings for ICC resource scheduling

Steps to execute the workflow:

1. Submit hyperparameter optimization tasks on ICC using:
   ```bash
   ./hyperParamOptimization.sh submit
   ```

2. After all configurations are completed, run the analysis task:
   ```bash
   ./hyperParamOptimization.sh analyze
   ```

3. Verify the optimized model against the baseline model:
   ```bash
   ./hyperParamOptimization.sh verify
   ```

## Output

The workflow generates the following outputs:

1. **Trained Models**: Saved in the `results/config_*` directories.
2. **Performance Metrics**: Stored in CSV files and MATLAB .mat files.
3. **Visualizations**: Figures comparing different hyperparameters and model performances.
4. **Best Model**: The optimal model saved in the `best_model` directory.
5. **Model Comparison**: Comparison between baseline and optimized models in the `results/model_comparison` directory.

## Notes for NCSA ICC Users

1. Make sure to update the account and partition in the hyperParamOptimization.sh script if needed:
   ```bash
   #SBATCH --account=siqi-ic        # Replace with your actual account name
   #SBATCH --partition=IllinoisComputes  # Replace with appropriate partition
   ```

2. Set the correct working directory path in the submission script to match your environment:
   ```bash
   WORK_DIR="$HOME/GeneralizablePyrolysisModel/bpDNN_withAsh4Ea/HyperparametersOptimization"
   ```

3. Ensure the correct MATLAB module is loaded on the cluster:
   ```bash
   # Check available MATLAB versions
   module avail matlab
   
   # Load the appropriate version (current default in script: matlab/24.1)
   module load matlab/24.1
   ```
   
   Note: According to NCSA ICC documentation, MATLAB is available with campus concurrent licensing. To verify which toolboxes are available, you can run:
   ```bash
   module load matlab
   matlab -nodisplay -nosplash -r "ver; exit"
   ```

4. For larger hyperparameter spaces, consider adjusting the resource request parameters:
   ```bash
   #SBATCH --nodes=1
   #SBATCH --ntasks-per-node=1
   #SBATCH --time=48:00:00    # Increase for larger hyperparameter spaces
   ```
   
   Note: If your hyperparameter optimization requires parallel computing capabilities, be aware that the MATLAB Parallel Computing Toolbox on ICC is limited to running on a single node. You cannot span workers across multiple nodes.

5. The workflow supports early stopping for efficient resource utilization on the cluster. 
   When submitting jobs with the `submit` command, you'll be prompted about enabling early stopping.

6. Before submission, ensure all input files are properly copied to the data directory:
   ```bash
   # Check data directory content
   ls -la $HOME/GeneralizablePyrolysisModel/bpDNN_withAsh4Ea/HyperparametersOptimization/data
   ```

7. Monitor your job progress using standard SLURM commands:
   ```bash
   # Check job status
   squeue -u $USER
   
   # Check job details
   scontrol show job [JOB_ID]
   
   # View job output while running
   tail -f [JOB_ID].out
   ```

8. After all jobs complete, run the analysis and verification steps:
   ```bash
   ./hyperParamOptimization.sh analyze
   ./hyperParamOptimization.sh verify
   ```

9. If you encounter MATLAB license errors during peak usage times, consider scheduling your jobs during off-peak hours:
   ```bash
   # Submit job to run at a specific time (e.g., at 10:00 PM)
   sbatch --begin=22:00 hyperParamOptimization.sh submit
   ```

## Best Practices

1. Start with a smaller hyperparameter space for initial testing before running the full optimization.
2. Use early stopping to save computational resources, especially on shared cluster environments.
3. Back up important results regularly using the automatic archiving feature.
4. Check the baseline model performance first to establish a performance baseline.
5. For very large hyperparameter spaces, consider using a staged approach with progressive refinement.
6. Before submitting jobs on ICC, verify the working directory path is correctly set:
   ```bash
   # Check and edit hyperParamOptimization.sh
   nano hyperParamOptimization.sh
   # Ensure WORK_DIR path is set correctly
   ```
7. When using multiple node resources on ICC, consider adjusting resource request parameters in the job submission script:
   ```bash
   #SBATCH --nodes=1                # Single node
   #SBATCH --ntasks-per-node=1      # Single task configuration
   #SBATCH --time=48:00:00          # Adjust based on hyperparameter space size
   ```

## Extending the Framework

The workflow can be extended by:

1. Adding new hyperparameters to the configuration file.
2. Implementing more advanced optimization algorithms (e.g., Bayesian optimization).
3. Creating additional visualization scripts for deeper analysis.
4. Adding support for different neural network architectures or frameworks.

## Troubleshooting MATLAB on NCSA ICC

### Common Issues

1. **MATLAB License Errors**: If you encounter a license error message like below, it means all concurrent licenses are in use:
   ```
   License checkout failed.
   License Manager Error -4
   Maximum number of users for Distrib_Computing_Toolbox reached.
   ```
   
   **Solution**: Try running your job during off-peak hours or use the `-a` option with `qsub` to schedule the job for a later time.

2. **Parallel Computing Issues**: When attempting to use more workers than available on a single node:
   ```
   Error using parpool (line 103)
   You requested a minimum of X workers, but the cluster "local" has the NumWorkers property set to allow a maximum of Y workers.
   ```
   
   **Solution**: Limit the number of workers to the number of cores available on a single node. The Parallel Computing Toolbox on ICC can only utilize cores from a single node.

3. **Race Conditions with Multiple Parallel Jobs**: When multiple parallel MATLAB jobs start simultaneously:
   ```
   Failed to start pool.
   Error using parallel.Job/createTask (line 277)
   Only one task may be created on a communicating Job.
   ```
   
   **Solution**: Stagger job submissions using job dependencies:
   ```bash
   sbatch job1.sh
   sbatch --dependency=after:JobID1 job2.sh
   ```

4. **Path Issues**: If MATLAB cannot find your scripts or data files:
   
   **Solution**: Use absolute paths in your scripts or set the MATLAB path correctly:
   ```matlab
   addpath(genpath('/path/to/your/code'));
   ```

### Checking MATLAB Toolbox Availability

To verify which MATLAB toolboxes are available on the cluster:

```bash
module load matlab
matlab -nodisplay -nosplash -r "ver; exit"
```

This command will display all installed toolboxes and their versions.

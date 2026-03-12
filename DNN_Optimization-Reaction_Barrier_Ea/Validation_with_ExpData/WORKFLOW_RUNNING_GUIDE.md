# Running the Pyrolysis Kinetics Model Workflow

This guide explains how to run the pyrolysis kinetics model workflow on different platforms.

## Windows Systems

### Option 1: PowerShell Script (Recommended)

1. Open PowerShell by right-clicking the Start menu and selecting "Windows PowerShell"
2. Navigate to the workflow directory:
   ```powershell
   cd "path\to\PKP_validation"
   ```
3. If running PowerShell scripts is restricted, you may need to set execution policy temporarily:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   ```
4. Run the PowerShell test script:
   ```powershell
   .\test_workflow_powershell.ps1
   ```
5. Check the log files in the `ps_test_logs` directory for any issues

### Option 2: Batch File

1. Open Command Prompt
2. Navigate to the workflow directory:
   ```cmd
   cd path\to\PKP_validation
   ```
3. Run the batch test script:
   ```cmd
   test_workflow_windows.bat
   ```
4. Check the log files in the `win_test_logs` directory for any issues

## Linux/ICC Systems

### Local Testing on Linux

1. Open a terminal
2. Navigate to the workflow directory:
   ```bash
   cd /path/to/PKP_validation
   ```
3. Make the test script executable:
   ```bash
   chmod +x test_workflow_locally.sh
   ```
4. Run the test script:
   ```bash
   ./test_workflow_locally.sh
   ```
5. Check the log files in the `test_logs` directory for any issues

### Running on ICC Supercomputing Platform

1. Upload all necessary files to ICC using scp or rsync:
   ```bash
   rsync -av PKP_validation/ username@icc.ncsa.illinois.edu:~/path/on/icc/PKP_validation/
   ```

2. Login to ICC:
   ```bash
   ssh username@icc.ncsa.illinois.edu
   ```

3. Navigate to the workflow directory:
   ```bash
   cd ~/path/on/icc/PKP_validation/
   ```

4. Edit the job submission script to include your account details:
   ```bash
   nano submit_pyrolysis_job.sh
   ```
   Update the following lines:
   - `#SBATCH --account=your_account_name` with your actual ICC account name
   - `#SBATCH --mail-user=your_email@illinois.edu` with your email address

5. Make the script executable:
   ```bash
   chmod +x submit_pyrolysis_job.sh
   ```

6. Submit the job:
   ```bash
   sbatch submit_pyrolysis_job.sh
   ```

7. Monitor job status:
   ```bash
   squeue -u $USER
   ```

8. After job completion, results will be packaged in `pyrolysis_results_<jobid>.tar.gz`

## Troubleshooting

- **MATLAB not found**: Ensure MATLAB is installed and in your system PATH
- **Memory issues**: Increase memory allocation in the job script
- **Timeout errors**: Increase time allocation in the job script 
- **Parallel pool errors**: Ensure correct number of cores are allocated

For more details, please refer to the `README_ICC_Running_Guide.md` file.

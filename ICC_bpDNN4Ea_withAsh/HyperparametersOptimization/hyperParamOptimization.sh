#!/bin/bash
#
# Hyperparameter Optimization Script for BP-DNN
# For use on NCSA ICC UIUC supercomputing platform
#

# Function to print usage information
print_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start       Start the complete hyperparameter optimization workflow"
    echo "  help        Display this help message"
    echo ""
    echo "Example:"
    echo "  $0 start    # Start the complete workflow"
}

# Validate working directory
validate_work_dir() {
    if [ ! -d "$WORK_DIR" ]; then
        echo "ERROR: Working directory $WORK_DIR does not exist"
        echo "Please ensure the directory exists and try again"
        exit 1
    fi
}

# Function to run the complete workflow
run_workflow() {
    echo "======================================================================"
    echo "  BP-DNN Hyperparameter Optimization Workflow"
    echo "  $(date)"
    echo "======================================================================"
    
    # Validate working directory
    validate_work_dir
    
    # Calculate total configurations
    CONFIG_COUNT=$(matlab -nodisplay -nosplash -r "addpath('./scripts'); countConfigurations; exit" | grep "Total combinations:" | awk '{print $3}')
    
    if [ -z "$CONFIG_COUNT" ]; then
        echo "ERROR: Could not determine configuration count"
        exit 1
    fi
    
    echo "Detected $CONFIG_COUNT hyperparameter configurations to optimize"
    
    # Check if early optimization is desired
    echo -n "Enable early stopping for underperforming configurations? (y/n): "
    read EARLY_STOP
    
    if [[ "$EARLY_STOP" == "y" || "$EARLY_STOP" == "Y" ]]; then
        EARLY_STOP_FLAG="true"
        echo "Early stopping enabled. Configurations will be evaluated periodically."
    else
        EARLY_STOP_FLAG="false"
        echo "Early stopping disabled. All configurations will run to completion."
    fi
    
    # Set array job range
    ARRAY_RANGE="1-$CONFIG_COUNT"
    
    # Create unified workflow job script
    JOB_SCRIPT="hyperopt_workflow_$$.sh"
    cat > $JOB_SCRIPT << EOL
#!/bin/bash
#SBATCH --job-name=bp_dnn_workflow
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=workflow_%A_%a.out
#SBATCH --error=workflow_%A_%a.err
#SBATCH --account=siqi-ic
#SBATCH --partition=IllinoisComputes
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=siqi@illinois.edu

cd \$SLURM_SUBMIT_DIR

# Load required modules
module purge
module load matlab/24.1

# Get configuration ID from SLURM_ARRAY_TASK_ID
CONFIG_ID=\${SLURM_ARRAY_TASK_ID}

# Ensure we're using SLURM array job
if [ -z "\$CONFIG_ID" ]; then
    echo "ERROR: No configuration ID provided. Should be submitted as SLURM array job."
    exit 1
fi

# Output start information
echo "Starting hyperparameter optimization configuration ID: \$CONFIG_ID"
echo "Job ID: \$SLURM_ARRAY_JOB_ID._\$SLURM_ARRAY_TASK_ID"
echo "Hostname: \$(hostname)"
echo "Running time: \$(date)"

# Early stopping update (every 5th configuration)
if [[ "$EARLY_STOP_FLAG" == "true" && \$((\$CONFIG_ID % 5)) -eq 0 ]]; then
    echo "Updating early stopping threshold..."
    matlab -nodisplay -nosplash -singleCompThread -r "addpath('./scripts'); setEarlyStoppingThreshold('./results'); exit"
fi

# Function to run MATLAB command with retry mechanism
run_matlab_with_retry() {
    local cmd=\$1
    local log_file=\$2
    local MAX_RETRIES=3
    local RETRY_COUNT=0
    local SUCCESS=false

    while [ \$RETRY_COUNT -lt \$MAX_RETRIES ] && [ "\$SUCCESS" = false ]; do
        echo "Attempt \$((\$RETRY_COUNT+1)) of \$MAX_RETRIES to run MATLAB command: \$cmd"
    
    matlab -nodisplay -nosplash -singleCompThread -r "addpath('./scripts'); \$cmd; exit" > \$log_file 2>&1
    
    if grep -q "License checkout failed" \$log_file; then
        RETRY_COUNT=\$((\$RETRY_COUNT+1))
        echo "MATLAB license error encountered. Waiting 5 minutes before retry..."
        sleep 300
    else
        SUCCESS=true
        cat \$log_file
    fi
done

if [ "\$SUCCESS" = false ]; then
    echo "ERROR: Failed to acquire MATLAB license after \$MAX_RETRIES attempts."
    exit 1
fi

# Check for errors in MATLAB output
if grep -q "Error:" \$log_file; then
    echo "ERROR: MATLAB execution failed"
    cat \$log_file
    exit 1
fi
}

echo "======================================================================"
echo "Stage 1: Running Hyperparameter Optimization"
echo "======================================================================"
run_matlab_with_retry "args = {'\$CONFIG_ID'}; runOptimization" "matlab_output_\${CONFIG_ID}.log"

# Process results when the last configuration completes
if [ "\$CONFIG_ID" == "$CONFIG_COUNT" ]; then
    echo "Last configuration completed, processing results..."
    sleep 60 # Brief pause to ensure all files are synchronized
    
    echo "======================================================================"
    echo "Stage 2: Running Results Analysis"
    echo "======================================================================"
    run_matlab_with_retry "analyzeResults('./results')" "matlab_analysis.log"
    
    echo "======================================================================"
    echo "Stage 3: Model Verification"
    echo "======================================================================"
    run_matlab_with_retry "verifyModels('./best_model', './matlab_src')" "matlab_verify.log"
    
    # Create results archive
    TIMESTAMP=\$(date +"%Y%m%d_%H%M%S")
    RESULT_ARCHIVE="optimization_results_\${TIMESTAMP}.tar.gz"
    echo "Creating results archive: \$RESULT_ARCHIVE"
    tar -czf \$RESULT_ARCHIVE results/ best_model/ figures/
fi

# Output completion information
echo "All stages complete for configuration ID \$CONFIG_ID"
echo "Completion time: \$(date)"
EOL

    # Submit SLURM array job
    echo "Submitting hyperparameter optimization array job, range: $ARRAY_RANGE"
    JOB_ID=$(sbatch --array=$ARRAY_RANGE $JOB_SCRIPT | awk '{print $4}')
    
    # Clean up temporary job script
    rm $JOB_SCRIPT
    
    echo "Jobs submitted! Array Job ID: $JOB_ID"
    echo "Monitor job progress with: squeue -u \$USER -j $JOB_ID"
    echo "Monitor job status with: squeue -u \$USER -j \$JOB_ID"
}

# Main script execution
case "$1" in
    start)
        run_workflow
        ;;
    help|--help|-h)
        print_usage
        ;;
    *)
        echo "ERROR: Unknown command '$1'"
        print_usage
        exit 1
        ;;
esac

exit 0

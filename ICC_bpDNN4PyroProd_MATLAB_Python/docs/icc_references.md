# Illinois Campus Cluster (ICC) Documentation References

This document provides references to the Illinois Campus Cluster (ICC) documentation relevant to running the neural network training and SHAP analysis on the ICC high-performance computing environment.

## ICC Overview

The Illinois Campus Cluster Program (ICCP) provides high-performance computing resources to researchers at the University of Illinois. The cluster is designed to support a wide range of research computing needs and allows researchers to contribute computing hardware in exchange for prioritized access.

- [ICCP Main Website](https://campuscluster.illinois.edu/)
- [ICC User Guide](https://docs.ncsa.illinois.edu/systems/icc/en/latest/)

## Getting Started

- [Account Setup](https://docs.ncsa.illinois.edu/systems/icc/en/latest/accounts/index.html) - How to request an account and set up access
- [Connecting to ICC](https://docs.ncsa.illinois.edu/systems/icc/en/latest/user_guide/connecting.html) - Methods for connecting to the cluster
- [Data Management](https://docs.ncsa.illinois.edu/systems/icc/en/latest/user_guide/filesystems.html) - Overview of available filesystems and data management

## Job Submission

This project uses the SLURM workload manager to submit jobs to the ICC. The following resources provide information on job submission:

- [SLURM Overview](https://docs.ncsa.illinois.edu/systems/icc/en/latest/user_guide/running_jobs/slurm/slurm.html) - Introduction to SLURM
- [Job Submission](https://docs.ncsa.illinois.edu/systems/icc/en/latest/user_guide/running_jobs/slurm/submission.html) - Submitting jobs using SLURM
- [Job Monitoring](https://docs.ncsa.illinois.edu/systems/icc/en/latest/user_guide/running_jobs/slurm/monitoring.html) - Monitoring job status
- [Job Arrays](https://docs.ncsa.illinois.edu/systems/icc/en/latest/user_guide/running_jobs/slurm/job_arrays.html) - Running multiple similar jobs

## SLURM Commands

Key SLURM commands used in this project:

- `sbatch` - Submit a batch job
- `squeue` - View information about jobs in the queue
- `scancel` - Cancel a job
- `sinfo` - View information about nodes and partitions
- `sacct` - View accounting information about jobs
- `scontrol` - View or modify job configuration

## MATLAB on ICC

This project uses MATLAB for neural network training. The following resources provide information on using MATLAB on ICC:

- [MATLAB Module](https://docs.ncsa.illinois.edu/systems/icc/en/latest/applications/matlab/index.html) - Loading and using the MATLAB module
- [Parallel Computing in MATLAB](https://docs.ncsa.illinois.edu/systems/icc/en/latest/applications/matlab/parallel.html) - Using MATLAB's parallel computing capabilities

## Python on ICC

This project uses Python for SHAP analysis. The following resources provide information on using Python on ICC:

- [Python Overview](https://docs.ncsa.illinois.edu/systems/icc/en/latest/applications/python/index.html) - Using Python on ICC
- [Conda Environments](https://docs.ncsa.illinois.edu/systems/icc/en/latest/applications/python/conda.html) - Creating and managing Conda environments
- [Jupyter Notebooks](https://docs.ncsa.illinois.edu/systems/icc/en/latest/applications/jupyter/index.html) - Using Jupyter notebooks on ICC

## Storage and Data Management

- [Filesystems](https://docs.ncsa.illinois.edu/systems/icc/en/latest/user_guide/filesystems.html) - Overview of available filesystems
- [Transferring Data](https://docs.ncsa.illinois.edu/systems/icc/en/latest/user_guide/transferring.html) - Methods for transferring data to and from ICC
- [Archiving Data](https://docs.ncsa.illinois.edu/systems/icc/en/latest/user_guide/archiving.html) - Long-term data storage options

## Batch Script Examples

Example batch scripts for running MATLAB and Python jobs on ICC:

- [MATLAB Batch Script Example](https://docs.ncsa.illinois.edu/systems/icc/en/latest/applications/matlab/example.html)
- [Python Batch Script Example](https://docs.ncsa.illinois.edu/systems/icc/en/latest/applications/python/example.html)

## Partitions and Resources

- [Partitions](https://docs.ncsa.illinois.edu/systems/icc/en/latest/user_guide/partitions.html) - Available partitions and their configurations
- [Resource Limits](https://docs.ncsa.illinois.edu/systems/icc/en/latest/user_guide/resource_limits.html) - Resource limits for jobs
- [Job Priority](https://docs.ncsa.illinois.edu/systems/icc/en/latest/user_guide/priority.html) - How job priority is determined

## Troubleshooting

- [Common Issues](https://docs.ncsa.illinois.edu/systems/icc/en/latest/faqs/common_issues.html) - Solutions to common issues
- [Getting Help](https://docs.ncsa.illinois.edu/systems/icc/en/latest/help/index.html) - How to get help with ICC 
# MATLAB Documentation References

This document provides references to MATLAB documentation relevant to the neural network model training and optimization used in this project.

## MATLAB Neural Network Toolbox

The MATLAB Neural Network Toolbox provides essential functions for designing, training, visualizing, and simulating neural networks. It's the primary toolbox used for the DNN model in this project.

- [Neural Network Toolbox Documentation](https://www.mathworks.com/help/deeplearning/index.html)
- [Getting Started with Neural Network Toolbox](https://www.mathworks.com/help/deeplearning/getting-started-with-deep-learning-toolbox.html)

## Key Functions Documentation

### Network Creation and Training

- [feedforwardnet](https://www.mathworks.com/help/deeplearning/ref/feedforwardnet.html) - Create a feedforward neural network
- [train](https://www.mathworks.com/help/deeplearning/ref/train.html) - Train a neural network
- [trainlm](https://www.mathworks.com/help/deeplearning/ref/trainlm.html) - Levenberg-Marquardt backpropagation algorithm
- [trainscg](https://www.mathworks.com/help/deeplearning/ref/trainscg.html) - Scaled conjugate gradient backpropagation

### Transfer Functions

- [tansig](https://www.mathworks.com/help/deeplearning/ref/tansig.html) - Hyperbolic tangent sigmoid transfer function
- [logsig](https://www.mathworks.com/help/deeplearning/ref/logsig.html) - Log-sigmoid transfer function
- [purelin](https://www.mathworks.com/help/deeplearning/ref/purelin.html) - Linear transfer function

### Performance Functions

- [mse](https://www.mathworks.com/help/deeplearning/ref/mse.html) - Mean squared error performance function
- [mae](https://www.mathworks.com/help/deeplearning/ref/mae.html) - Mean absolute error performance function

### Data Preprocessing

- [mapminmax](https://www.mathworks.com/help/deeplearning/ref/mapminmax.html) - Normalize data to fall in the range [-1,1]
- [mapstd](https://www.mathworks.com/help/deeplearning/ref/mapstd.html) - Normalize data to have zero mean and unity standard deviation

### Visualization

- [plotperform](https://www.mathworks.com/help/deeplearning/ref/plotperform.html) - Plot network performance
- [plottrainstate](https://www.mathworks.com/help/deeplearning/ref/plottrainstate.html) - Plot training state values
- [plotregression](https://www.mathworks.com/help/deeplearning/ref/plotregression.html) - Plot linear regression

## Parameter Optimization

For grid search and parameter optimization:

- [Statistics and Machine Learning Toolbox](https://www.mathworks.com/help/stats/index.html)
- [Hyperparameter Optimization](https://www.mathworks.com/help/stats/hyperparameter-optimization-in-classification-learner-app.html)
- [bayesopt](https://www.mathworks.com/help/stats/bayesopt.html) - Bayesian optimization

## Parallel Computing

For accelerating model training:

- [Parallel Computing Toolbox](https://www.mathworks.com/help/parallel-computing/index.html)
- [parfor](https://www.mathworks.com/help/parallel-computing/parfor.html) - Parallel for-loops
- [parpool](https://www.mathworks.com/help/parallel-computing/parpool.html) - Create parallel pool

## Custom Neural Network Functions

The custom neural network functions used in this project (like `nntrain.m`, `nnff.m`, etc.) are based on MATLAB's neural network implementation patterns but have been customized for this specific application. They follow similar principles to those described in the MATLAB Neural Network Toolbox documentation.

## MATLAB File I/O

- [save](https://www.mathworks.com/help/matlab/ref/save.html) - Save workspace variables to file
- [load](https://www.mathworks.com/help/matlab/ref/load.html) - Load variables from file

## MATLAB Graphics and Visualization

- [figure](https://www.mathworks.com/help/matlab/ref/figure.html) - Create figure window
- [subplot](https://www.mathworks.com/help/matlab/ref/subplot.html) - Create axes in tiled positions
- [plot](https://www.mathworks.com/help/matlab/ref/plot.html) - 2-D line plot
- [title](https://www.mathworks.com/help/matlab/ref/title.html) - Add title to axes
- [xlabel, ylabel](https://www.mathworks.com/help/matlab/ref/xlabel.html) - Add axis labels
- [legend](https://www.mathworks.com/help/matlab/ref/legend.html) - Add legend to axes
- [saveas](https://www.mathworks.com/help/matlab/ref/saveas.html) - Save figure to file
- [savefig](https://www.mathworks.com/help/matlab/ref/savefig.html) - Save figure to .fig file 
Code for reproduction of the paper Multivariate Probabilsitic Prediction using Natural Gradient Boosting.
## TODO Link arxiv link to paper here.

The source code is supplied for the paper to allows others to do similar comparisons to what we do in the paper.

Some functions are only implemented for the two dimensional case, in which case a ValueError will occur.

Note that the implementation of multivariate Gaussian NGBoost is not provided here, that is provided in the package ngboost and can be installed via
```shell
pip install ngboost
```
The package `probdrift` which is supplied in this repository was specifically developed for the experiments in the paper, not for general & flexible use.
# Replication of experiments
Prior to replicating the experiments one must install the package "probdrift" given in this repository.
either clone the repository and run the following in the directory:
```shell    
pip install . 
```
Alternatively, install from github
```shell
pip install git+git://github.com/MikeOMa/mvn_prediction
```
## Simulation Replication

For the simulations given in Section 4 of the paper see the simulation_example folder

The results can be replicated by running the following within the simulation example folder:
```
for i in 500 1000 3000 5000 8000 10000
    python eval_model.py $i
    python eval_model.py $i vanilla
end
```
The vanilla option runs the simulation without a -x^2 +x term.
For the simulations given in Section 4 of the paper see the simulation_example folder

This creates a folder results_50_230521 containing the results (`_V` appened for vanilla).

These are summarised into Tables 1, S2 and S3 and Figure S1 in the paper.

## Application Replication

Due to dataset size it is supplied on zenado this must be downloaded seperately.

## TODO Link Zenado




## Software versions for replication

The versions of each package which were in the environment used to run the code are given in `replication.txt`, the python version used is 3.8.3.
The only important lines from this file should be
probdrift==0.1.0 ## The package in this repo
ngboost==0.3.9 ## The package for NGBoost
numpy==1.19.5 ## Used for random number generation
tensorflow==2.4.1 ## Neural network initialization and optimization
scikit-learn==0.23.2 ## Used for trees and vanilla gradient boosting

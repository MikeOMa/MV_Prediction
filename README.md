# Code for reproduction of the paper Multivariate Probabilsitic Prediction using Natural Gradient Boosting.

This repository is only for replication of results. An implementation allowing one to fit a multivariate gaussian distribution is in the `ngboost` package.
A minimal example (ngboost version 0.3.10) would be
```
import ngboost
import numpy as np 
from ngboost.distns import MultivariateNormal
N=1000
p=2
X = np.random.randn(N,10)
Y = np.random.randn(N,p)
# Create the mvn class
mvn_dist = MultivariateNormal(p)
model = ngboost.NGBRegressor(Dist=mvn_dist)
model.fit(X,Y)
# If one wants a list of predicted MVN scipy distributions:
model.pred_dist(X).scipy_distribution()
```

**TODO** arxiv link to paper here.


Note that the implementation of multivariate Gaussian NGBoost is not provided here, that is provided in the package ngboost and can be installed via
```shell
pip install ngboost
```
The source code is supplied for the paper to allows others to do similar comparisons to what we do in the paper.
# probdrift package
The package `probdrift` which is supplied in this repository was specifically developed for the experiments in the paper, not for general & flexible use.
The `probdrift` package is essentially a wrapper for NGBoost, scikit-learn gradient boosting and a tensorflow neural network, where all methods are adapted to output a p dimensional multivariate normal distribution. 

Some other useful elements of the package are:

- Some metrics applicable to multivariate gaussian responses in `probdrift.metrics`. In particular, an Energy Score estimator, RMSE, Negative log-liklihood, $\alpha\%$ coverage and prediction interval area.
- `probdrift.MVN_helpers` contains a function `elipse_points` to form the boundary of a Bi-Varate Gaussian $\alpha\%$ prediction inverval
- `probdrift.oceanfns` filtering functions used to process the North Atlantic Ocean dataset. 

To install either clone the repository and run the following in the directory:
```shell    
pip install . 
```
Alternatively, install from github
```shell
pip install git+git://github.com/MikeOMa/mvn_prediction
```

# Replication of Experiments

Prior to replicating the experiments one must install the package "probdrift" given in this repository. Follow instructions in the previous section to install it.

## Simulation Replication

For the simulations given in Section 4 of the paper see the simulation_example folder

The results can be replicated by running the following within the simulation example folder:
```
for i in 500 1000 3000 5000 8000 10000 do
    python eval_model.py $i
    python eval_model.py $i vanilla
done
```
The vanilla option runs the simulation without a -x^2 +x term.
For the simulations given in Section 4 of the paper see the simulation_example folder

This creates a folder results_50_230521 containing the results (`_V` appened for vanilla).

These are summarised into Tables 1, S2 and S3 and Figure S1 in the paper.

## Application Replication

Due to dataset size the North Atlantic Ocean (NAO) dataset is supplied on zenado this must be downloaded seperately.
**Important**: Once downloaded before running one must set the environment variable `data_file` to the path for the h5 file.
```
export data_file=PATH_TO_NAO_DATASET
```

**TODO** Link to Zenado where file is hosted.

To create the hyper parameter configurations files for the grid search run the following:
```shell
python generate_configuration.py
```
This creates and fills the `config_files_leaf` directory with configation files.
`dispatcher.py` reads the configuration file and calls the relevant fitting function, run as:
It reads the `$i` configuration file then fits the model specified in that file
```shell 

for i in 0...79 do 
python dispatcher.py $i
done
```
#### Note:
This will use around 5GB disk space as all models and predictions on the test data are stored in a results folder. To lessen the hard drive load
edit the file such that it does not save the models. 

Each set of model fits takes a  maximum of about 70 hours on a modern pc (ngboost with natural_graident=False will take longest).
This could be greatly reduced through using a lighter base learner such as LightGBM.

Finally, `summarise.py` will compute the metrics based on the predictions from dispatcher above from the `probdrift.metrics` module 
```shell

for i in 0...79 do 
python summarise.py $i
done
```
This script will create and fill `experiments/metrics` which are summarised using `notebooks/SummariseResults` to create the tables in the paper.

## Software versions for replication

The versions of the packages used to run the experiments in the paper are supplied to allow replication of results.
The versions of each package which were in the environment used to generate data and fit models are given in `replication.txt`, the python version used was 3.8.3.
The most important lines from this file are:
```
probdrift==0.1.0 ## The package in this repo
ngboost==0.3.9 ## The package for NGBoost
numpy==1.19.5 ## Used for random number generation
tensorflow==2.4.1 ## Neural network initialization and optimization
scikit-learn==0.23.2 ## Used for tree base learners and vanilla gradient boosting
scipy==1.6.1 ## Used within Indep NGB and NGB
```

These are note forced in requirements.txt or the installation of the package probdrift.

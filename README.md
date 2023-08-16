# M5GP 
M5GP Project.
Implementation of Genetic Programming algorithm in CUDA.
```
This is a Python implementation of  M5GP programming algorithm.
```
***
## Description:  
M5GP implements a Scikit-Learn type interface using Python, the necessary methods are available for its evaluation and use, in addition a Regressor type object compatible with Scikit-Learn in Python was defined.

Train/fit functions were defined in a base class of type Regressor.

Established a dictionary of type hyper_params or a list of dictionaries specifying the hyperparameter search space

A function has been defined that returns a sympy-compatible string that specifies the final model and can be manipulated in sympy.

The integration of the Numba library and cuML within M5GP was carried out with the objective of using it as a variant of evaluation of the models obtained through GP and to improve the efficiency and suitability of the results.

***
## Software code languajes, tools, and services used
```
Python, SciKit-Learn, NUMBA, PYCUDA CUML, SRBENCH
```
***
## Requirements, operating enviroments & dependencies 
Python > 3.8 version <br>
Conda  > 23.3 version <br>
Conda Environment for rapidsai > 23.04 <br>
Conda package scikit-cuda <br>
Conda package scikit-learn <br>
Conda package pycuda <br>

## Installation 
1. Install the conda environment rapidsai: <br>
conda create -n rapids-23.04 -c rapidsai -c conda-forge -c nvidia  rapids=23.04 python=3.8 cudatoolkit=11.5 [link](https://docs.rapids.ai/install) <br>
conda activate rapids-23.04 <br>

3. Install adition packages:
pip install scikit-cuda <br>
conda install -c conda-forge scikit-learn <br>
conda install -c conda-forge pycuda <br>

4. Download the M5GP source code:
git clone https://github.com/armandocardenasf/m5gp.git

***
## How to run:  
For execute M5GP type follows commands
```
> cd m5gp
> python m5gp.py
```

Set the parameters for M5GP execution

***
## Parameters:  
The folllow parameters are passed to Regressor object for M5GP execution modify the parameters accordingly to adjust to the desired evolutionary conditions

| Parameter Name     								| Default Value   | Description|
| -------- 								| -------- |------------|
|1.  Generations				| 30     |Total number of iterations of the main evolutionary loop. |
|2.  Individuals				| 256     |Number of individuals generated in the population.|
|3.  GenesIndividuals      | 128     |Number of genes of each individual in the population.|
|4.  MutationProb          | 0.10   |Mutation rate probability. For UMAD probability operator.| 
|5.  MutationDeleteRateProb     | 0.01      |Mutation delete probability.  For UMAD probability operator.|
|6.  SizeTournament          | 0.15   |Size of elitist tournament.| 
|7.  EvaluationMethod          | 0   |Error evaluation method. <br><b>M5GP native methods:</b><br>0=RMSE<br>1=R2 <br><b>cuML Methods:</b><br>2=LinearRegression<br> 3=Lasso Regression<br>4=Ridge regression<br>5=kernel Ridge Regression<br>6=ElasticNet Regression<br><b>cuML MiniBatch options:</b><br>7=MiniBatch none regularization (linear regression)<br>8=MiniBatch lasso regularization <br>9=MiniBatch ridge regularization <br>10=MiniBatch elasticnet regularization | 
|7.  MaxRandomConstant			| 80       |Maximum / Minimum range values of the random constants used. Whatever the value, negative constants of the same magnitude are also generated in the range of -Range to +Range.|
|8.  GenOperatorProb         | 0.5     |Probability of generating a gene corresponding to an operator (+ - * / sin cos, log ...).|
|9.  GenVariableProb         | 0.5     |Probability of generating a gene corresponding to a variable.|
|10.  GenConstantProb         | 0.5     |Probability of generating a gene corresponding to a constant.|
|11.  GenNoopProb         | 0.5     |Probability of generating a gene corresponding to Noop (no valid gene).|
|12.  UseOpIF         | 0.5     |Specifies whether IF conditional operators will be used on the individual.|
|13.  Log         | 0.5     |Determines if results will be recorded in Log file.|
|14.  Verbose         | 0.5     |Specifies if results are shown on the screen during program execution.|
|15.  Log Path                           | log/     |Directory where the files generated by M5GP will be stored.|



## Documentation:
```
The documentation of the library is a Doxygen documentation. The implementation has been done in order to use the library after a very quick reading of the documentation.
```

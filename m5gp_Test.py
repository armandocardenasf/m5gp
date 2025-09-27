#Branch IF 3.8

from m5gp import m5gpRegressor as m5gp
import m5gpGlobals as gpG
#from   sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import sympy as sym
from sympy import symbols, Mul, simplify, count_ops

##import kagglehub
### Download latest version
##path = kagglehub.dataset_download("nikolasgegenava/sneakers-classification")
##print("Path to dataset files:", path)


#load the data
#dataset = pd.DataFrame(pd.read_csv("/home/treelab/python-codes/data/Concrete/train_10107_1.txt" ,sep='\s+', header=None))
#dataset = pd.DataFrame(pd.read_csv("/home/acardenasf/pmlb/datasets5/589_fri_c2_1000_25/589_fri_c2_1000_25.tsv" ,sep='/s+', header=None))
#dataset = pd.DataFrame(pd.read_csv("/home/acardenasf/datasets/test_10107_1.csv" ,sep=' ', header=None))
#dataset = pd.DataFrame(pd.read_csv("/home/acardenasf/datasets/207_autoPrice.tsv" ,sep='\t', header=None))
dataset1 = pd.DataFrame(pd.read_csv("/home/acardenasf/datasets/344_mv.tsv" ,sep='\t', header=None))


print("Leyo dataset")
nrows = len(dataset1.index)
if (nrows > 10000):
    print("Hay mas de 10000")
    #dataset1 = dataset1.iloc[:10000]  #o df.head(10000)
    dataset1 = dataset1.sample(n=10000, random_state=42)

dataset = dataset1

nvar = dataset.shape[1] - 1
#print("Leyo X")
X = dataset.iloc[0:nrows, 0:nvar-1]
y = dataset.iloc[:nrows, nvar-1]

x_train = dataset.iloc[0:nrows, 0:nvar-1].to_numpy().astype(np.float32)
y_train = dataset.iloc[:nrows, nvar-1].to_numpy().astype(np.float32)
 
#X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.70,test_size=0.30,random_state=n)

#functions_set = ["+", "-", "*", "/", "sin", "cos", "tan", "tanh", "exp", "log", "abs", "sum","prod", "avg", "std"]
#Operadores = ["+", "-", "*", "/", "sin", "cos", "tan", "tanh", "exp", "log", "abs"]
functions_set = ["+", "-", "*", "/", "sin", "cos", "tan", "tanh", "exp", "log", "abs"]

print('Running m5gp ...')  
 
est = m5gp( generations=30, # number of generations (limited by default)
            Individuals=256, # number of individuals
            GenesIndividuals=128, # number of genes per individual
            mutationProb=0.1, # mutation rate probability
            mutationDeleteRateProb=0.05,  # mutation delete rate probality
            sizeTournament=0.15, # size of tournament
            evaluationMethod=2,  #error evaluation method 
                        # 0=RMSE, 
                        # 1=R2, 
                        #cuML Methods
                        # 2=LinearRegression, 3=Lasso Regression, 
                        # 4=Ridge regression, 5=kernel Ridge Regression,
                        # 6=ElasticNet Regression
                        #cuML MiniBatch options
                        # 7=MiniBatch none regularization (linear regression)
                        # 8=MiniBatch lasso regularization 
                        # 9=MiniBatch ridge regularization 
                        #10=MiniBatch elasticnet regularization 
            scorer=0,
            maxRandomConstant=10, #number of constants (-maxRandomConstant to maxRandomConstant)
            genOperatorProb=0.53, #probablity for generate Operators 
            genVariableProb=0.37, #probablity for generate variables 
            genConstantProb=0.05, #probablity for generate constants
            genNoopProb=0.05, #probablity for generate NOOP Operators 
			useOpIF=0, #Set if use IF operator
            functions_set = functions_set, # Set of operators for include into individuals 
            log=1, #save log files
			verbose=1, #Show menssages on execution
            logPath='log/' #path for logs
 )

# Model = [-10099,  -1002,  -1005, -10005,    878,    647, -10007,  -1003,  -1001,   -1000]
#Model =[-10009, -10006, -10006,  -1000,  -1002,  -1005,   -737,    113,  -1005, -10007, -10002, -10004,  -1003, -10007,  -1005, -10009,  -1004, -10010, -1003, -549.]
#est.getModelExpr(Model)
#exit(0)

#ea.cudacapabilities()

est.fit(x_train, y_train)

print("Complexity: ", est.complexity())
model = est.get_model()
print("Model: ",est.get_model())
#D = simplify(model)
#print(D)

yPredicted = est.predict(x_train)
#print("Y Data :\n", y_train)
#print("Y Predicted:\n", yPredicted)

mse = est.meanSquaredError(y_train, yPredicted)
print("mse: ", mse)
print("rmse:", est.rmse(y_train, yPredicted))
print ("R^2: ", est.R2(y_train, yPredicted))
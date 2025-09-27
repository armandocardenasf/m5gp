
from .src.m5gp import m5gp
import pandas as pd
 
hyper_params = [
        {
            'generations' : (30,),
            'Individuals' : (128,),
            'GenesIndividuals' : (128,),
            'mutationProb' : (0.1,),
            'sizeTournament' : (0.25,),
        }, 
        {
            'generations' : (30,),
            'Individuals' : (128,),
            'GenesIndividuals' : (128,),
            'mutationProb' : (0.1,),
            'sizeTournament' : (0.15,),
        },
        {
            'generations' : (30,),
            'Individuals' : (128,),
            'GenesIndividuals' : (128,),
            'mutationProb' : (0.1,),
            'sizeTournament' : (0.1,),
        },
        {
            'generations' : (30,),
            'Individuals' : (128,),
            'GenesIndividuals' : (256,),
            'mutationProb' : (0.1,),
            'sizeTournament' : (0.1,),
        },
        {
            'generations' : (30,),
            'Individuals' : (256,),
            'GenesIndividuals' : (256,),
            'mutationProb' : (0.15,),
            'sizeTournament' : (0.15,),
        },
        {
            'generations' : (50,),
            'Individuals' : (128,),
            'GenesIndividuals' : (128,),
            'mutationProb' : (0.1,),
            'sizeTournament' : (0.25,),
        },    
        {
            'generations' : (50,),
            'Individuals' : (256,),
            'GenesIndividuals' : (128,),
            'mutationProb' : (0.1,),
            'sizeTournament' : (0.15,),
        },                                  
    ]

functions_set = ["+", "-", "*", "/", "sin", "cos", "tan", "tanh", "exp", "log", "abs", "sum","prod", "avg", "std"]
functions_set = ["+", "-", "*", "/", "sin", "cos", "tan", "tanh", "exp", "log", "abs"]

# from .src.m5gp import m5gp
# from m5gp import m5gpRegressor as m5gp

# Create the pipeline for the model
print('Running m5gp ...')
est = m5gp.m5gpRegressor( 
            generations=30, # number of generations (limited by default)
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

def complexity(est):
    print("Complexity:", est.get_n_nodes())
    nodes = est.get_n_nodes()
    return nodes

def model(est):
    indiv = est.best_individual()
    return str(indiv)

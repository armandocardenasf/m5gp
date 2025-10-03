# *********************************************************************
# Name: m5gp.py
# Description: Modulo principal del sistema que implementa los
# metodos del ciclo evolutivo de GP, asi como la interface tipo SkLearn
# Se implementa la logica de ejecucion para funciones de numba y CuML
# *********************************************************************

from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import os
import sys 
import math
import copy
import pandas as pd
import numpy as np
import time
import gc
import cupy as cp
import torch

from numba import cuda
from numba.cuda.random import (create_xoroshiro128p_states,
                               xoroshiro128p_uniform_float32)
# import rmm 
# from rmm.allocators.cupy import rmm_cupy_allocator

this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(this_dir))
import m5gpGlobals as gpG
import m5gpCudaMethods as gpCuda
import m5gpCumlMethods as gpCuM
import m5gpMod1 as gpM1
import m5gpMod2 as gpM2


class m5gpRegressor(BaseEstimator):
  #method to initialize the class
  def __init__(self, 
            generations=50, 
            Individuals=500, 
            GenesIndividuals=1024, 
            mutationProb=0.15, 
            mutationDeleteRateProb=0.01, 
            sizeTournament=0.20,  
            evaluationMethod=0,        
            scorer=0,  
            maxRandomConstant=5, 
            genOperatorProb=0.54, 
            genVariableProb=0.35, 
            genConstantProb=0.10, 
            genNoopProb=0.001,  
            useOpIF=0,
            functions_set = ["+", "-", "*", "/", "sin", "cos", "exp", "log", "abs", "sum","prod", "avg", "std"],   
            log=1, 
            verbose=1, 
            logPath='log/'
            ):

    env = dict(os.environ)
    self.generations = generations
    self.Individuals=Individuals 
    self.GenesIndividuals=GenesIndividuals
    self.mutationProb=mutationProb 
    self.mutationDeleteRateProb=mutationDeleteRateProb  
    self.evaluationMethod=evaluationMethod
    self.scorer = scorer
    self.sizeTournament=sizeTournament 
    self.maxRandomConstant=maxRandomConstant 
    self.genOperatorProb=genOperatorProb 
    self.genVariableProb=genVariableProb 
    self.genConstantProb=genConstantProb 
    self.genNoopProb=genNoopProb  
    self.useOpIF=useOpIF
    self.nvar=0 
    self.nrowTrain=0 
    self.nrowTest=0
    self.log=log
    self.verbose=verbose
    self.logPath=logPath
    self.functions_set=functions_set
    self.model = ''
    self.m4gpModel = ''
    self.cuModel = 0
    self.bestIndividual = ''

    print("Initializing m5gp")

    # Check if CUDA and Device GPU are available
    if torch.cuda.is_available():
      # Get the device name
      device = torch.cuda.get_device_name(0)
      print(f"Using CUDA device: {device}")

      print("Initial memory info:")
      # Get total GPU memory
      total_memory = torch.cuda.get_device_properties(0).total_memory
      gpG.gpu_memory = total_memory
      print(f"Total GPU memory: {total_memory / (1024**3):.2f} GB")  # Convert to GB
      
      # Get current memory allocation
      allocated_memory = torch.cuda.memory_allocated(0)
      print(f"Allocated GPU memory: {allocated_memory / (1024**3):.2f} GB")

      # Get cached memory
      cached_memory = torch.cuda.memory_reserved(0)
      print(f"Cached GPU memory: {cached_memory / (1024**3):.2f} GB")

      # Free up unused cached memory
      torch.cuda.empty_cache()
      print("Unused cached memory freed")

      # Get allocated memory after clearing cache
      allocated_memory_after = torch.cuda.memory_allocated(0)
      print(f"Allocated GPU memory after clearing cache: {allocated_memory_after / (1024**3):.2f} GB")

      gpG.free_mem = total_memory - allocated_memory_after
      print(f"Free GPU memory : { gpG.free_mem / (1024**3):.2f} GB")

      # Pool de memoria (ajusta tamaño al GPU)
      # allocMem =  int((gpG.gpu_memory / (1024**3)) - 1)
      # rmm.reinitialize(pool_allocator=True, initial_pool_size=allocMem<<30)  # 5 GB
      # cp.cuda.set_allocator(rmm_cupy_allocator)

    else:
      print("CUDA is not available.")
      return

    # Verifica los operadores validos y construye el diccionario a utilizar
    # para generar la poblacion inicial
    if (len(self.functions_set) == 0):
      print("No se definieron operadores")
      exit(0)
    
    # Get valid functions (mathematical operators) allowed for generate individuals
    self.valid_functions_set = gpG.construir_lista_operadores_validos(self.functions_set)
    if (len(self.valid_functions_set) == 0):
      print("No se especificaron operadores validos")
      exit(0)
    
    # print("valid_functions_set")
    # print(self.valid_functions_set)

    fName = "M5GP_OpS.csv"
    if os.path.exists(fName):
        os.remove(fName)
    return


  #This method implement the evolution with M5GP  
  def fit(self, X_train, y_train):
    # Normalizar Probabilidades por seguridad
    totalProb = self.genOperatorProb + self.genVariableProb + self.genConstantProb + self.genNoopProb
    if totalProb <= 0.0:
          # fallback razonable
      p_op_n, p_var_n, p_const_n, p_noop_n = 0.53, 0.37, 0.05, 0.05
    else:
      inv = 1.0 / totalProb
      p_op_n, p_var_n, p_const_n, p_noop_n = self.genOperatorProb*inv, self.genVariableProb*inv, self.genConstantProb*inv, self.genNoopProb*inv

    self.genOperatorProb=p_op_n 
    self.genVariableProb=p_var_n 
    self.genConstantProb=p_const_n 
    self.genNoopProb=p_noop_n
    self.maxRandomConstant=np.float32(self.maxRandomConstant)   

    self.X_train = X_train
    self.y_train = y_train

    # train data    
    data=pd.DataFrame(self.X_train)
    data['target']=self.y_train

    self.nrowTrain = len(data.index)
    self.nrowTest = len(data.index)
    self.nrowPredict = len(data.index)
    self.nvar = data.shape[1] - 1
 
    print("Executing Fit - Method(", self.evaluationMethod ,") - ", gpCuM.cuGetMethodName(self), " Scorer:", self.scorer)
    print("nRows:", self.nrowTrain, "nVars:", self.nvar)

    #pesos = {op: 1.0/len(self.valid_functions_set) for op in self.valid_functions_set}
    #pesos = {op: 1/len(self.valid_functions_set) for op in self.valid_functions_set}

    #Initialize operators weigth
    pesos_por_id = {op: 1.0/len(self.valid_functions_set) for op in self.valid_functions_set}
    op_weights = np.array([pesos_por_id[int(oid)] for oid in self.valid_functions_set], dtype=np.float32)
      # Prepara la CDF una vez por generación (Numba)
    cdf = gpM2.preparar_sampler_operadores_rapido_numba(
        op_ids=self.valid_functions_set, op_weights=op_weights,
        epsilon=0.02, temperatura=1.0
    )
    
    # print("Pesos iniciales")
    # print(pesos_por_id)
    # print(op_weights)
    # print(cdf)

    # Store the size in bytes for initial population
    gpG.sizePopulation = self.Individuals * self.GenesIndividuals 
    gpG.sizeIndividuals = self.Individuals 
    gpG.sizeTournament = math.ceil(self.sizeTournament * self.Individuals)

    # Define vectors to work on device 
    self.model = np.zeros((self.GenesIndividuals ), dtype=np.float32) 

    #print("Initialize Individual")
    # *************************** Initialize population ********************************* 
    hInitialPopulation = gpM1.initialize_population(
                              self.Individuals,
                              self.nvar,
                              self.GenesIndividuals,
                              self.maxRandomConstant,
                              self.genOperatorProb,
                              self.genVariableProb,
                              self.genConstantProb,
                              self.genNoopProb,
                              self.useOpIF,
                              self.valid_functions_set,
                              cdf )
    # -- End of Initialize population --

    # print("Individuals:")
    # print(hInitialPopulation)
    # return
  
    # ***************************  Compute Individuals  ****************************
    hOutIndividuals = [] 
    hStack = []
    hStackIdx = []
    hStackModel = []
  

    #print ("Compute Individual")
    hOutIndividuals, hStack, hStackIdx, hStackModel = gpM1.compute_individuals(
            hInitialPopulation,
            self.X_train,
            self.Individuals,
            self.GenesIndividuals,
            self.nrowTrain,
            self.nvar,
            0 )
    # ****************** End of Compute Individuals **********************
    
    # Get the semantic matrix
    coefArr_p = []
    intercepArr_p = []    
    cuModel_p = []  
    stackBestModel_p = []

    coefArrNew = []
    intercepArrNew = [] 
    cuModelNew = []
    stackBestModelNew = []

    hFit = np.zeros((gpG.sizeIndividuals), dtype=np.float32)
    hFitNew = np.zeros((gpG.sizeIndividuals), dtype=np.float32)
    indexBestOffspring = 0
    indexWorstOffspring = 0


    #print("Compute Error")
    # ***************************** Compute ERROR ***********************************
    hFit, indexBestOffspring, indexWorstOffspring, coefArr_p, intercepArr_p, cuModel_p = gpM1.ComputeError(self,
                hOutIndividuals, 
                self.y_train, 
                self.Individuals, 
                self.nrowTrain,
                hStack, 
                hStackIdx,
                self.evaluationMethod)

    # Index of best individual of initialization
    indexBestIndividual_p = indexBestOffspring 

    del hStack
    del hStackIdx
    gc.collect()

    ajFit = 0
    if (self.evaluationMethod == 1) : #or (self.scorer == 2) :  
      ajFit = gpG.MAX_R2_NEG * (-1)
    trainFit = hFit[indexBestIndividual_p] - ajFit    
    print("Initial Index:", indexBestIndividual_p, " Initial Fit:", trainFit)


    # ***********************************************************************
    # ********************* GP Process Generation Cycle *********************
    # ***********************************************************************
    print ("Starting generational process")
    for generation in range(1,self.generations + 1):
      trainFit = 0
      testFit = 0
      coefArrNew = []
      intercepArrNew = [] 
      cuModelNew = []
      stackBestModelNew = []
      start_time = time.time()

      #print("Torneo")
      # *********************  Select Tournament  **********************
      hNewPopulation, hBestParentsTournament = gpM1.select_tournament(
                    hInitialPopulation,
                    hFit,
                    self.Individuals, 
                    self.GenesIndividuals )

      #print("Mutacion")
      # *********************  UMAD Mutation  **********************
      hNewPopulation = gpM1.umadMutation(self,
                                  hInitialPopulation,
                                  hBestParentsTournament,
                                  self.Individuals,
                                  cdf) 

      #print (hNewPopulation)
      # ***************************  Compute Individuals  ****************************
      hOutIndividuals, hStack, hStackIdx, hStackModel = gpM1.compute_individuals(
              hNewPopulation,
              self.X_train,
              self.Individuals,
              self.GenesIndividuals,
              self.nrowTrain,
              self.nvar,
              0 )
      
      # ***************************** Compute ERROR ***********************************
      hFitNew, indexBestOffspring, indexWorstOffspring, coefArrNew, intercepArrNew, cuModelNew = gpM1.ComputeError(self,
              hOutIndividuals, 
              self.y_train, 
              self.Individuals, 
              self.nrowTrain,
              hStack, 
              hStackIdx,
              self.evaluationMethod)

      oldFit = hFit[indexBestIndividual_p]
      newFit = hFitNew[indexBestOffspring]

      #print("hFit:", hFit[indexBestIndividual_p], " indexBestIndividual_p:", indexBestIndividual_p)
      #print("hFitNew:", hFitNew[indexBestOffspring], " indexBestOffspring:", indexBestOffspring)

      # *********************** FUNTIONS WEIGHT EVALUATION ***********************

      # Mejor individuo de la nueva generacion (BestOffspring)
      idx_a1 = indexBestOffspring * self.GenesIndividuals
      idx_b1 = indexBestOffspring * self.GenesIndividuals + self.GenesIndividuals
      mBestIndividual = hNewPopulation[idx_a1:idx_b1]

      #pesos = gpG.actualizar_pesos_por_fitness(ops, pesos, best, fit_prev=0.48, fit_curr=0.52, lower_is_better=True)
      #pesos = gpG.actualizar_pesos_por_fitness(self.valid_functions_set, pesos, mBestIndividual, fit_prev=0.48, fit_curr=0.52, lower_is_better=True)
      pesos_por_id = gpM2.actualizar_pesos_operadores(pesos_por_id, mBestIndividual, oldFit, newFit, self.valid_functions_set)
      op_weights = np.array([pesos_por_id[int(oid)] for oid in self.valid_functions_set], dtype=np.float32)

      # print(mBestIndividual)
      # print(self.valid_functions_set)
      # print(pesos_por_id)
      # print(op_weights)

      # Prepara la CDF una vez por generación
      cdf = gpM2.preparar_sampler_operadores_rapido_numba(
          op_ids=self.valid_functions_set, op_weights=op_weights,
          epsilon=0.02, temperatura=1.0
      )
      #print(cdf)

      # *********************** NEW SURVIVAL (Elitist) ***********************
      hNewPopulation, indexBestIndividual_p, coefArr_p, intercepArr_p, cuModel_p, stackBestModel_p = gpM1.Survival(self,
              indexBestIndividual_p,
              indexBestOffspring,
              indexWorstOffspring,
              hInitialPopulation,
              hNewPopulation,
              hFit,
              hFitNew,
              coefArr_p, 
              intercepArr_p, 
              cuModel_p,
              stackBestModel_p,
              coefArrNew,
              intercepArrNew,
              cuModelNew,
              stackBestModelNew)
      # *********************** END NEW SURVIVAL ***********************

      # ***********************    NEW REPLACE   ***********************
      hInitialPopulation, hFit = gpM1.replace(self,
                      hInitialPopulation,
                      hNewPopulation, 
                      hFit,
                      hFitNew)
      # *********************** END NEW REPLACE ***********************
      # print (hInitialPopulation)
      

			# Validate Best Individual with Test file for generation
			#/*trainFit = checkFitness(config, handle, dataFile, dInitialPopulation, indexBestIndividual_p, 0);*/
      ajFit = 0
      if (self.evaluationMethod == 1) : #or (self.scorer == 2) : 
        ajFit = gpG.MAX_R2_NEG * (-1)

      bestFitGeneration = hFit[indexBestIndividual_p]
      trainFit = hFit[indexBestIndividual_p] - ajFit

      # Obtenemos la longitud del stack del mejor papa
      BestIndividualLength = gpG.bestIndividualInfo(self, hInitialPopulation,  indexBestIndividual_p)
     

      if self.verbose == 1 :
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        print("Generation:",generation, " Best Index:",indexBestIndividual_p, " Length Indiv:",BestIndividualLength, " Train Fit:",trainFit, f" Time lapsed:{elapsed}")
			#end if

      del hStack
      del hStackIdx
      del hStackModel
      del hOutIndividuals 
      gc.collect()
      
      
      if hFitNew[indexBestIndividual_p] <= 0.00000000001 :
        break

    #end for 
    # ************* Fin de for (Ciclo Generacional) ****************

    # print(mBestIndividual)
    # print("Operadores validos:")
    # print(self.valid_functions_set)
    # print("Pesos por id:")
    # print(pesos_por_id)
    # print("Pesos:")
    # print(op_weights)

    # Obtenemos el mejor individuo
    idx_a1 = indexBestIndividual_p * self.GenesIndividuals
    idx_b1 = indexBestIndividual_p * self.GenesIndividuals + self.GenesIndividuals
    self.bestIndividual = hInitialPopulation[idx_a1:idx_b1]
    self.model = self.bestIndividual

    #print("Index bestIndividual:")
    #print(indexBestIndividual_p)
    #print("bestIndividual:")
    #print(self.bestIndividual)

    # Para caso de evaluaciones utilizando cuML se construye una 
    # expresion utilizando todas expresiones del stack que generan 
    # la matriz semantica 
    if (self.evaluationMethod >= 2 ) :
      self.cuModel = copy.deepcopy(cuModel_p)
      self.maxRandomConstant = gpG.MAX_CONSTANT

      #print("X_train:")
      #print(X_train)

      #X_train2 = X_train[0]
      #print("X_train2:")
      #print(X_train2)

      #sacamos el mejor modelo del stack de expresiones 
      stackBestModel_p = gpM1.getStackBestModel(
                  self.bestIndividual,
                  X_train,
                  self.Individuals,
                  self.GenesIndividuals,
                  self.nrowTrain,
                  self.nvar) 
    
      #print("stackBestModel_p:")
      #print(stackBestModel_p)

      # Se construye una nueva pila con las todas expresiones 
      # generadas y almacenadas en el stack del mejor modelo
      allStackExpr = gpG.getStackModelExpr(self, stackBestModel_p)
      
      #print("allStackExpr:")
      #print(allStackExpr)

      # De la cadena completa de expresiones obtenemos el numero  
      # de stacks de expresiones disponibles
      # 'X:Y:Z:<Expr1>', 'X:Y:Z:<Expr2>', .... , 'X:Y:Z:<ExprN'  
      # (0)X=Total de elementos, 
      # (1)Y=Numero de stacks, 
      # (2)Z=Elemento de este stack 
      tmpModelExpr = allStackExpr[0]
      tmp = tmpModelExpr.split(':')
      nStack = int(tmp[1])

      # Se crea una nueva pila para guardar las expresiones de cada 
      # elememento del stack y posteriormente formar el modelo tipo M4GP
      nvoModel = [] 
      m4gpModel = gpG.m4gpModel(self, stackBestModel_p, 
                                coefArr_p, 
                                intercepArr_p) 
      
      #print("m4gpModel:")
      #print(m4gpModel)

      #print("self.cuModel.coef_.shape:", self.cuModel.coef_.shape)
      #print("self.cuModel.coef_:", self.cuModel.coef_)

      # Reconstruccion del modelo.
      # Se agregan los coeficientes obtenidos del modelo de evaluacion cuML
      # Por cada expresion, tenemos un coefiente  
      
      for j in range(nStack):
        #Obtenemos las expresiones del stack
        tmp1 = m4gpModel.get()
        tmp2 = coefArr_p
        tmp3 = tmp2[nStack-j-1]
        if(math.isnan(tmp3) or math.isinf(tmp3)) :
          tmp3 = 0

        #print(tmp1)
        #print(tmp2)
        #print(tmp3)

        # Solo interesan expresiones cuyo coeficiente no sea cero
        if (tmp3 != 0) :
          #Se agrega el coeficiente al inicio de la expresion
          #Se le agrega un operador de multiplicacion (*)
          nvoModel.insert(0,float(-10003)) 
          nvoModel.insert(0,float(tmp3))
          nvoModel =  gpG.m4gpBuildExpr(tmp1, nvoModel)
          if (j >= 1):
            nvoModel.append(-10001)            
      #end for

      tmp3 = intercepArr_p
      if (math.isnan(float(tmp3)) or (math.isinf(float(tmp3)))) :
        tmp3 = 0

      # Solo interesan expresiones cuyo coeficiente no sea cero
      if (tmp3 != 0) :
      #insertamos el intercept a la expresion
        nvoModel.append(float(tmp3))
        nvoModel.append(-10001)  # Se agrega un operador de suma (+)
      
      nvoModel.append(-11111)
      self.m4gpModel = np.array(nvoModel)
      del nvoModel
    #end if

    #Free local memory
    del hFit
    del hFitNew
    del hInitialPopulation

    # Clear lists
    gpCuM.coefArr.clear()
    gpCuM.intercepArr.clear()
    gpCuM.cuModel.clear()
    gc.collect()
    
    print("Finished Fit.")
    return	 
  # Fin de def (fit)

  def predict(self, X_predict):
    if (len(self.bestIndividual) == 0) :
      print("No model available for predict")
      return
    
    print("Inicio predict: ", X_predict.shape)

    self.X_predict = X_predict

    # Get number of data rows for predict
    self.nrowPredict = self.X_predict.shape[0]
    hDataPredict = np.reshape(self.X_predict, -1)

    
    numIndividuals = 1
    hModelPopulation = self.bestIndividual  
    GenesIndiv = hModelPopulation.shape[0] # self.GenesIndividuals

    # ***************************  Compute Individuals  ****************************
    hOutIndividuals, hStack, hStackIdx, hStackModel = gpM1.compute_individuals(
            hModelPopulation,
            hDataPredict,
            numIndividuals,
            GenesIndiv,
            self.nrowPredict,
            self.nvar,
            0 )

    y_pred=[]

    stackBestModel_p = gpM1.getStackBestModel(
                hModelPopulation,
                self.X_predict,
                numIndividuals,
                GenesIndiv,
                self.nrowPredict,
                self.nvar) 
    #allModelExpr = gpG.getModelExpr(self, stackBestModel_p)   

    if (self.evaluationMethod < 2 ) :
      for i in range(self.nrowPredict):
        y_pred.append(hOutIndividuals[i])

      y_pred = np.array(y_pred)
    else :
      st = hStack.reshape(numIndividuals, self.nrowPredict * GenesIndiv)
      ind = st[0]
      ind2 = ind.reshape(self.nrowPredict, GenesIndiv)
      tt = int(hStackIdx[0])
      
      sX_train = ind2[:, :tt]     
      cX = cp.asarray(sX_train, dtype=cp.float64)
      y_predModel = self.cuModel.predict(cX)
      y_pred = cp.asnumpy(y_predModel)

      #Free local memory
      del st
      del ind
      del ind2
      del tt
      del sX_train
      del cX
      del y_predModel
    #End if

    #Free local objects memory
    del hStack
    del hStackIdx
    del hStackModel
    del hOutIndividuals 
    del hDataPredict
    del hModelPopulation
    gc.collect()

    return y_pred
  # Fin de def (predict)

  # def getModelExpr(self, model):
  #   allModelExpr = gpG.getStackModelExpr(self, model) 

  #   print(allModelExpr)
  #   tmpModelExpr = allModelExpr[0]
  #   tmp = tmpModelExpr.split(':')
  #   nStack = int(tmp[1])

  #   BestModelExpr = allModelExpr[nStack-1]
  #   tmp = BestModelExpr.split(':')
  #   indivLenght = tmp[0]
  #   nStack = tmp[1]
  #   complexity = tmp[2] 
  #   modelExpr = tmp[3]

  #   return modelExpr
  
  def best_individual(self):
    if ((self.model == 0).all()) :
      print("No model available")
      return
    
    if (self.evaluationMethod < 2 ) :
      model = self.model
    else :
      model = self.m4gpModel

    allModelExpr = gpG.getStackModelExpr(self, model) 

    tmpModelExpr = allModelExpr[0]
    tmp = tmpModelExpr.split(':')
    nStack = int(tmp[1])

    BestModelExpr = allModelExpr[nStack-1]
    tmp = BestModelExpr.split(':')
    indivLenght = tmp[0]
    nStack = tmp[1]
    complexity = tmp[2] 
    modelExpr = tmp[3]

    return modelExpr  
  # Fin de def (best_individual) 

  def get_model(self):
    return self.best_individual()
  # Fin de def (get_model) 

  def get_n_nodes(self):
    if (self.evaluationMethod < 2 ) :
      model = self.model
    else :
      model = self.m4gpModel     

    allModelExpr = gpG.getStackModelExpr(self, model) 
    tmpModelExpr = allModelExpr[0]
    tmp = tmpModelExpr.split(':')
    nStack = int(tmp[1])

    BestModelExpr = allModelExpr[nStack-1]

    tmp = BestModelExpr.split(':')
    nStack = tmp[1]
    nodes = tmp[2] 

    return str(nodes)
  # Fin de def (get_n_nodes) 

  def complexity(self):
    return self.get_n_nodes()
  # Fin de def (complexity) 
   
  def meanSquaredError(self, cY, YPred) :
    # if (len(cY) == 0 or type(YPred == 'NoneType') or len(YPred) ==0):
    #   print("Not cY or YPred providen")
    #   return
    
    npY = np.array(cY).astype('float32')

    npYPred = YPred
    #mse = mean_squared_error(npY, npYPred, squared=False)
    mse = mean_squared_error(npY, npYPred)
    return mse

  def rmse(self, cY, YPred) :
    # if (len(cY) == 0 or type(YPred == 'NoneType') or len(YPred) ==0):
    #   print("Not cY or YPred providen")
    #   return
    
    mse = self.meanSquaredError(cY, YPred) 
    mse = math.sqrt(mse)
    return mse
   		
  def R2(self, cY, YPred):
    # if (len(cY) == 0 or type(YPred == 'NoneType') or len(YPred) ==0):
    #   print("Not cY or YPred providen")
    #   return
    
    r2 = r2_score(cY, YPred)
    return r2
  
  def getStackExpr(self, Model) :
    self.nvar=7
    allModelExpr = gpG.getStackModelExpr(self, Model)
    print(allModelExpr)
    return
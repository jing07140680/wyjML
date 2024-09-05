# train.py
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from gpu_model import PyTorchMLPRegressor
import pandas as pd
from sklearn.metrics import r2_score
import torch

def main():
    print("CUDA Available:", torch.cuda.is_available())

    rf=pd.read_csv("../data/final1.csv")

    print(rf.head())
    #2-5#6-8#9-12
    spring = []#March 18-23
    for i in range(6):
        spring.append([3,i+18])
        sfinal = []#May 20-31
    for i in range(12):
        sfinal.append([5,i+20])

    #summer break June-Aug 25
    sbreak = []
    for i in range(30):
        sbreak.append([6,i+1])
    for i in range(31):
        sbreak.append([7,i+1])
    for i in range(24):
        sbreak.append([8,i+1])

 
    wfinal = []#Dec 21-30   wbreak:Dec-Jan13
    for i in range(10):
        wfinal.append([12,i+21])

    holiday=[[1,1],[1,21],[2,18]]+spring+[[5,12]]+sfinal+sbreak+[[9,2],[10,14],[11,11],[11,28],[11,29]]+wfinal
    #holiday


    rf['holiday']=[0]*len(rf)
    for i in holiday:
        for j in rf[(rf['Month']==i[0])&(rf['Day']==i[1])].index:
            rf.loc[j,'holiday']=1
      
    #May and June Weekends
    TS=[]
    trainm=[3,4,5]
    rftrain = rf.loc[rf['Month'].isin(trainm)]
    testm=[6]
    rftest = rf.loc[rf['Month'].isin(testm)]
    
    traindata = rftrain.iloc[:,4:]
    testdata = rftest.iloc[:,4:]
    
    trlabel = traindata[['count']]
    trainfeature = traindata.drop(['count','Month','Day','en','out','ds','supply'],axis=1)

    telabel = testdata[['count']]
    testfeature = testdata.drop(['count','Month','Day','en','out','ds','supply'],axis=1)

    trf_norm = (trainfeature - trainfeature.min()) / (trainfeature.max() - trainfeature.min())
    tef_norm = (testfeature - trainfeature.min()) / (trainfeature.max() - trainfeature.min())
    trf_norm.fillna(0, inplace = True)
    tef_norm.fillna(0, inplace = True)

    # Define the parameter grid
    param_grid = {
        'hidden_layer_sizes': [(10,20), (20,20), (30, 20)],
	'activation': ['relu', 'tanh'],
        'alpha': [0.001, 0.01, 0.1],
        'epochs': [10, 20],
        'batch_size': [8,16],
        #'solver': ['adam', 'sgd', 'lbfgs']  # Added solver parameter

    }

    # Initialize your custom model
    model = PyTorchMLPRegressor()

    # Set up GridSearchCV
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

    print(type(trlabel))
    # Fit GridSearchCV
    grid_search.fit(trf_norm, trlabel)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Evaluate the best model
    y_pred = best_model.predict(tef_norm)
    print("Best Parameters:", grid_search.best_params_)
    r2 = r2_score(y_pred,telabel)
    print ("Test ERROR = ", r2)

if __name__ == "__main__":
    main()

import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV




pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
rf=pd.read_csv("final1.csv")
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

# Create a parameter grid for hyperparameter tuning
param_grid = {
    'hidden_layer_sizes': [(10,20), (20,20), (30,20)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'lbfgs'],
    'alpha': [0.001, 0.01, 0.1,1]
}

# Create an MLPRegressor instance
model = MLPRegressor()

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(trf_norm, trlabel)

# Use the best model from grid search
best_model = grid_search.best_estimator_
print(best_model)

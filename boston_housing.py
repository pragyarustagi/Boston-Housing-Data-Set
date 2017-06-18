import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split

df = pd.read_csv("boston.csv")
#print(df.head())

def classification_model(model, data, predictors, outcome):
  
    X = data[predictors]
    y = data[outcome]
  
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33)
  
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return accuracy
    
model = RandomForestRegressor()
predictor_var = ['LSTAT', 'RM' , 'PT' , 'INDUS' , 'TAX'] 
outcome_var = 'MV'
scoring = []
for i in range(10):
    classification_model(model, df,predictor_var,outcome_var)
    scoring.append(classification_model(model, df,predictor_var,outcome_var))
    
print(np.mean(scoring))
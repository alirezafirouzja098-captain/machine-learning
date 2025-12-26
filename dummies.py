import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("homeprices_.csv")
dummies=pd.get_dummies(df.town)
merged = pd.concat([df,dummies],axis='columns')
final=merged.drop(['town','west windsor'],axis='columns')
model = linear_model.LinearRegression()

x = final.drop('price',axis='columns')
y = final.price
model.fit(x,y)

print(model.predict([[3215,0,1]]))
print(model.predict([[1450,1,0]]))
print(model.predict([[2240,0,0]]))


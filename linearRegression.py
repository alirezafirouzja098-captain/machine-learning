import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("homeprices2.csv")
median =df.bedrooms.median()
df.bedrooms=df.bedrooms.fillna(median)

model = linear_model.LinearRegression()
model.fit(df[['area','bedrooms','age']],df.price)

predict = [[3300,4,12],[1600,3,8]]
print(model.predict(predict))


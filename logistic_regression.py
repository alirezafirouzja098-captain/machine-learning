import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("insurance_data.csv")

x_train, x_test, y_train, y_test=train_test_split(df[['age']],df.bought_insurance,train_size=0.9)

model=LogisticRegression()
model.fit(x_train,y_train)

print(model.predict(x_test))
print(model.score(x_test,y_test))
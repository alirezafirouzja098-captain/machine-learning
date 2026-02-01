import pandas as pd
from sklearn.datasets import load_iris
iris=load_iris()
df =pd.DataFrame(iris.data,columns=iris.feature_names)
df['target']=iris.target
df0=df[:50]
df1=df[50:100]
df2=df[100:]
from sklearn.model_selection import train_test_split
x=df.drop(['target'], axis='columns')
y=df.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
print(knn.score(x_test,y_test))
from sklearn.metrics import confusion_matrix
y_pred=knn.predict(x_test)
print(confusion_matrix(y_test,y_pred))
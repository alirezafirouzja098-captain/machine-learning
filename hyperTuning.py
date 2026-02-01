from sklearn import svm,datasets
iris=datasets.load_iris()

import pandas as pd
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['flower']=iris.target
df['flower']=df['flower'].apply(lambda x:iris.target_names[x])


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.3)


from sklearn.model_selection import GridSearchCV

clf=GridSearchCV(svm.SVC(gamma='auto'),{
    'C':[1,10,20],
    'kernel':['rbf','linear']
},cv=5,return_train_score=False)
clf.fit(iris.data,iris.target)

db=pd.DataFrame(clf.cv_results_)

db[['param_C','param_kernel','mean_test_score']]
print(clf.best_params_)

from sklearn.model_selection import RandomizedSearchCV
rs=RandomizedSearchCV(svm.SVC(gamma='auto'),{
    'C':[1,10,20],
    'kernel':['rbf','linear']
},cv=5,return_train_score=False,n_iter=2)
rs.fit(iris.data,iris.target)
print(pd.DataFrame(rs.cv_results_)[['param_C','param_kernel','mean_test_score']])


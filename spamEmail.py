import pandas as pd
df=pd.read_csv("/home/alireza/git_folder/machine-learning/spam.csv")
df.groupby('Category'). describe()
df['spam']=df['Category'].apply(lambda x:1 if x=='spam' else 0)
df.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df.Message,df.spam,test_size=0.25)
from sklearn.feature_extraction.text import CountVectorizer
v=CountVectorizer()
x_train_count=v.fit_transform(x_train.values)
x_train_count.toarray()[:3]
from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(x_train_count,y_train)
x_test_count=v.transform(x_test)
model.score(x_test_count,y_test)
from sklearn.pipeline import Pipeline
clf=Pipeline([
    ('vectorizer',CountVectorizer()),
    ('nb',MultinomialNB())
])


clf.fit(x_train,y_train)

print(clf.score(x_test,y_test))
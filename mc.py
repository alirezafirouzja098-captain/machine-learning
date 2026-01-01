import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

digits = load_digits()

plt.gray()
plt.matshow(digits.images[0])

x_train, x_test, y_train, y_test= train_test_split(digits.data,digits.target,test_size=0.2)


model = LogisticRegression()

model.fit(x_train,y_train)

print(model.predict([digits.data[78]]))

y_pridicted = model.predict(x_test)
cm = confusion_matrix(y_test,y_pridicted)
print(cm)

plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('pridicted')
plt.ylabel('truth')




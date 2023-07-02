import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#DATA ANALYSIS

hd = pd.read_csv("heart disease classification dataset.csv")
hd
hd.head()
hd.describe()
hd.info()
df = pd.DataFrame(hd)
df

 # DATA VISUALIZATION

sns.countplot(x='sex',hue='target',data=df,palette='Set1')
sns.countplot(x='target',hue='target',data=df,palette='Set1')
sns.lmplot(x='thalach', y='trestbps',hue='target',data=df,palette='Set1')
sns.lmplot(x='thalach',y='chol',hue='target',data=df)
hd.hist(figsize=(11,11))

#DATA CLEANING AND PREPARATION

dummies=pd.get_dummies(hd)
hd=dummies.drop('Unnamed: 0',axis=1)
hd
hd=dummies.drop('target_no',axis=1)
hd
sns.histplot(hd['age'].dropna(),kde=False, bins=30)
hd['age'].plot.hist(bins=30)
import cufflinks as cf
cf.go_offline()
sns.heatmap(hd.isnull(),yticklabels=False, cbar=False, cmap='viridis')
hd.dropna(inplace=True)
sns.heatmap(hd.isnull(),yticklabels=False, cbar=False, cmap='viridis')
hd.keys()
corr=hd.corr()
corr
sns.heatmap(corr, cmap='magma')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(hd.drop('target_yes',axis=1))
hd['target_yes']
arr = hd['target_yes'].to_numpy()
arr
arr.reshape(-1,1)
from sklearn.model_selection import train_test_split
x = hd
y = np.ravel(arr)
x_test, x_train, y_test, y_train = train_test_split(x,y, test_size=0.3)

#SVM

from sklearn.svm import SVC
model = SVC()
model.fit(x_train,y_train)
pred = model.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(y_test, pred))
confusion_matrix(y_test, pred)
svm_accuracy = accuracy_score(y_test, pred)
svm_accuracy

#KMEANS

from sklearn.datasets import make_blobs
make_blobs
k = make_blobs(n_samples=100, n_features=2, centers=None, cluster_std=1.0,random_state=0)
k
plt.scatter(k[0][:,0],k[0][:,1], c=k[1], cmap='magma')
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(k[0])
kmeans.cluster_centers_
kmeans.labels_
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=1)
y_train
kn.fit(x_train,y_train)
predict = kn.predict(x_test)
confusion_matrix(y_test, predict)
print(classification_report(y_test, predict))
kmeans_accuracy = accuracy_score(y_test, predict)
kmeans_accuracy
kn_n = KNeighborsClassifier(n_neighbors=17)
kn_n.fit(x_train,y_train)
pd=kn_n.predict(x_test)
print(confusion_matrix(y_test,pd))
print('\n')
print(classification_report(y_test,pd))
knn_accuracy = accuracy_score(y_test,pd)
knn_accuracy

#LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
lm = LogisticRegression()
lm.fit(x_train,y_train)
prediction = lm.predict(x_test)
prediction
import sklearn.metrics as metrics
print(classification_report(y_test,prediction))
confusion_matrix(y_test,prediction)
print("Acurracy",metrics.accuracy_score(y_test,prediction))
print("Precision",metrics.precision_score(y_test,prediction))
print("Acurracy",metrics.recall_score(y_test,prediction))
model = ['SVM Model', 'KNN Model','Logistic Regression']
acc=[0.9414634146341463,0.9512195121951219,0.9804878048780488]
plt.bar(model,acc)
plt.title("Accuracy Rates of three models")
plt.show()
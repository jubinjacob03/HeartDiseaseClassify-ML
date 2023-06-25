import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# DATA ANALYSIS --

# Reading data using Pandas heart=pd.read_csv("Heart Disease.csv") heart
heart=pd.read_csv("Heart Disease.csv")
heart
heart.info()
heart.index
heart.columns
heart.isnull().value_counts()
sns.heatmap(heart.isnull(),yticklabels=False,cbar=False)
heart.drop(["education"],axis=1,inplace=True)
heart
heart["male"].fillna(method="ffill")
heart["age"].fillna(heart["age"].mean())
heart["currentSmoker"].fillna(0).value_counts()
heart["cigsPerDay"].fillna(heart["cigsPerDay"].mean(),inplace=True)
heart["BPMeds"].fillna(method="bfill",inplace=True)
heart["prevalentStroke"].fillna(method="ffill",inplace=True)
heart["diabetes"].fillna(method="bfill",inplace=True)
heart["totChol"].fillna(heart["totChol"].mean(),inplace=True)
heart["sysBP"].fillna(heart["sysBP"].mean(),inplace=True)
heart["diaBP"].fillna(heart["diaBP"].mean(),inplace=True)
heart["BMI"].fillna(heart["BMI"].mean(),inplace=True)
heart["heartRate"].fillna(heart["heartRate"].mean(),inplace=True)
heart["glucose"].fillna(heart["glucose"].mean(),inplace=True)
heart["TenYearCHD"].fillna(method="ffill",inplace=True)
heart
heart.isnull().value_counts()
heart.isnull()
heart.rename(columns={'male':'Sex'},inplace=True)
heart.columns
hc=heart.corr()
hc

# DATA VISUALIZATION --

sns.heatmap(heart.isnull(),yticklabels=False,cbar=False)
sns.heatmap(hc)
sns.pairplot(heart)
sns.distplot(heart["TenYearCHD"])
sns.distplot(heart["TenYearCHD"],kde=False)
sns.jointplot(x='heartRate',y='TenYearCHD',data=heart)
sns.jointplot(x='heartRate',y='TenYearCHD',data=heart,kind='kde')
x=heart[['Sex', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds','prevalentStroke', 'prevalentHyp','diabetes','totChol',
'sysBP','diaBP', 'BMI', 'heartRate', 'glucose']]
y=heart['TenYearCHD']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=7)
from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
log.fit(x_train,y_train)
pred=log.predict(x_test)
h2=pd.DataFrame({"predicted":pred,"actual":y_test})
h2

# PERFORMANCE EVALUATION --

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred)
TP,FN,FP,TN=confusion_matrix(y_test,pred).reshape(-1)
TP
FN
FP
TN
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

#Accuracy=(TN+TP)/(TN+TP+FN+FP)
accuracy_score(y_test,pred)

#Precision=(TP)/(TP+FN)
precision_score(y_test,pred)

#Recall=(TP)/(TP+FP)
recall_score(y_test,pred)

#F1-score=(2*Recall*Precision)/(Recall+Precision)
f1_score(y_test,pred)
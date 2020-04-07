# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 10:16:04 2020

@author: adeline
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

gender= pd.read_csv("brief/gender_submission.csv")
test= pd.read_csv("brief/test.csv")
train = pd.read_csv("brief/train.csv")

print(gender.shape)
print(test.shape)
print(train.shape)

print(train.head())
print(test.head())
print (train.describe())
print(train.isna().sum())

print(test.isna().sum())

print(train.info())
print(train.dtypes)
print(train.Cabin.unique())
print(train.Embarked.unique())
print(train.Pclass.unique())
print(train.Name.unique().sum())
print(train.Fare.unique())
print(train.Fare.mean())
print(train.Fare.max())
print(train.Fare.min())

#suppression ou remplacement des valeurs na
#j'ai choisi de remplacere l'age manquant par la valeur médiane
Age2 = train['Age'].median()
print(Age2)
train['Age'].fillna(Age2, inplace=True)

#je supprime les colonnes Cabin(77% de valeur manquante) et ticket car le numéro n'apporte rien
del train['Cabin']
del train['Ticket']

# je remplace les 2 valeurs manquantes du port d'embarquement par Southampton qui est le plus eleve
train["Embarked"].value_counts()
train.Embarked=train["Embarked"].fillna("S")
train.info()
print(train.head())
print(train.isna().sum())

# je fais la même chose pour la partie test
Age3 = test['Age'].median()
print(Age3)
test['Age'].fillna(Age3, inplace=True)

Fare2 = test['Fare'].mean()
print(Fare2)
test['Fare'].fillna(Fare2, inplace=True)

del test['Cabin']
del test['Ticket']

print(test.head())
print(test.isna().sum())

# Ne pas oublier que la variable cible 'survived' est présnte dans le train et
# pas dans le test pour la suite 

train["Age"].hist()

train.boxplot("Age")

train["Age"].plot(kind="box")

train["Fare"].hist()

train["Fare"].plot(kind="hist")

# qualitatif
train["Survived"].value_counts()

train["Pclass"].value_counts()
train["Sex"].value_counts()
train["Embarked"].value_counts()

plt.xlabel('Survivant')
plt.ylabel('Sex')
plt.scatter(train['Age'], train['Survived'], color='blue', label='Age')
plt.scatter(train['Fare'], train['Survived'], color='red', label='Fare')
plt.scatter(train['Pclass'], train['Survived'], color='green', label='Pclass')

plt.legend(loc=3, prop={'size':8})
plt.show()

train.corr()
train.plot(kind="scatter",x="Age",y="Fare")


# afficher une sélection
train[train["Age"]>60][["Sex","Pclass","Age","Survived"]]
train.boxplot(column="Age",by="Pclass")

from statsmodels.graphics.mosaicplot import mosaic
mosaic(train,["Pclass","Sex"])
mosaic(train,["Survived","Pclass"])
mosaic(train,["Survived","Sex"])

sns.factorplot('Sex',data=train,kind='count')
sns.factorplot('Pclass',data=train,kind='count')
sns.factorplot('Pclass',data=train,hue='Sex',kind='count')
sns.factorplot('Survived',data=train,kind='count',hue='Pclass')

def titanic_children(passenger):
    
    age , sex = passenger
    if age <16:
        return 'child'
    else:
        return sex

train['person'] = train[['Age','Sex']].apply(titanic_children,axis=1)
train.head(10)
sns.factorplot('Pclass',data=train,hue='person',kind='count')
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 09:24:59 2020

@author: Utilisateur
"""
import pandas as pd
import math

gender= pd.read_csv("data/gender_submission.csv")
test= pd.read_csv("data/test.csv")
train = pd.read_csv("data/train.csv")

gender.shape
test.shape
train.shape
gender.isna().sum()
test.isna().sum()
train.isna().sum()

#Le pourcentage de NaN dans le Train pour cabine 687/891
train.Cabin.isna().sum() /train.shape[0] *100
#Le pourcentage de NaN dans le Test pour cabine 327/418
test.Cabin.isna().sum() /test.shape[0] *100
#Le pourcentage de NaN dans le Train pour cabine 177/891
train.Age.isna().sum() /train.shape[0] *100
#Le pourcentage de NaN dans le Test pour cabine 86/418
test.Age.isna().sum() /test.shape[0] *100


def getbothNaN(df):
    nbrNan=0
    for index, i in df.iterrows():
        if math.isnan(i['Age']) and str(i["Cabin"])== "nan" : 
            nbrNan+=1
            df.drop(index, inplace=True)
    return nbrNan , df

def delageNaN(df):
    nbrNan=0
    for index , i in df.iterrows():
        if math.isnan(i['Age']):
            nbrNan+=1
            df.drop(index, inplace=True)
    for i in df.Age : 
        if i % 1 != 0 and i >1:
            i = round(i)
    return df

def delcabincol(df):
    df = df.drop(columns=['Cabin'])
    return df

def cleanEmbarked(df):
    for index , i in df.iterrows():
        if str(i["Embarked"])== "nan":
            df.drop(index, inplace=True)
            
def cleanFare(df):
    for index , i in df.iterrows():
        if math.isnan(i['Fare']) :
            df.drop(index, inplace=True)


trainAgeCabinNaN = getbothNaN(train)
testAgeCabinNaN = getbothNaN(test)
trainAgeCabinNaN /train.shape[0] *100
testAgeCabinNaN /test.shape[0] *100

def cleanage(df):
    #Supression des lignes avec NaN dans Age et Cabin
    nbrNan, df = getbothNaN(df)
    print(df)
    #Round + del des autres lignes
    df = delageNaN(df)
    #Supression de la colonne Cabin
    df = delcabincol(df)
    if df.Embarked.isna().sum() >= 1 :
        cleanEmbarked(df)
    if df.Fare.isna().sum() >= 1 :
        cleanFare(test)
    print("All cleaned")
    return df
test = cleanage(test)
train = cleanage(train)

test.to_csv("data/cleantest.csv")
train.to_csv("data/cleantrain.csv")
    

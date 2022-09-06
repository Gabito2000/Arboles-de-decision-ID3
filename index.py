import argparse
from ast import For
from datetime import date, datetime
import sys
import math
from xmlrpc.client import Boolean
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stat
import seaborn as sns
import numpy as np
import warnings
from imblearn.over_sampling import RandomOverSampler as oversampler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import f1_score as f1
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from sklearn.metrics import confusion_matrix
import g02_l2_core


warnings.filterwarnings("ignore")

df = pd.read_csv("./healthcare-dataset-stroke-data.csv")

#funcion para imprimir las metricas
def PrintMetrics(Y_test, predict):
    print("Accuracy -> "+str(acc (Y_test, predict)))
    print("Precision -> "+str(precision (Y_test, predict)))
    print("Recall -> "+str(recall (Y_test, predict)))
    print("F1 -> "+str(f1 (Y_test, predict)))
    print("Matriz de Confusión:")
    tn, fp, fn, tp = confusion_matrix(Y_test, predict).ravel()
    print("      1        0")
    print(" 1    {:<8} {:<8} ".format(tp, fp))
    print(" 0    {:<8} {:<8} ".format(fn, tn))



parser = argparse.ArgumentParser()
parser.add_argument("--algoritmo", type=str, default="g02", required=False, help="Algoritmo a usar, 'g02' o 'dtc' (DecisionTreeClassifier) por defecto se usa el desarrollado por el grupo 02,")
parser.add_argument("--oversample", type=int, default=0, required=False, help="Utilizar oversample, 0 = 'No' cualquier otro valor se toma como 'Si'")

args = parser.parse_args()

print("Utilizando parámetros, Algoritmo = "+args.algoritmo+", Oversample = "+str(args.oversample))


#modificamos los enumerados a valores reales discretos
df['ever_married'] = df['ever_married'].replace(['Yes', 'No'], [1,0])
df['gender'] = df['gender'].replace(['Male', 'Female', 'Other'], [0,1,2])
df['work_type'] = df['work_type'].replace(['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'], [0,1,2,3,4])
df['Residence_type'] = df['Residence_type'].replace(['Urban', 'Rural'], [1,0])
df['smoking_status'] = df['smoking_status'].replace(['formerly smoked', 'never smoked', 'smokes', 'Unknown'], [0,1,2,3])
del df["id"] #el id no puede ir ya que hace sobreajuste


if (args.algoritmo == 'g02'):
        dfDistribuida = df.copy()
        if (args.oversample == '1'):
                #distribucion del stroke
                ros = oversampler(random_state=42)
                X = dfDistribuida[["gender", "age", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type",
                        "avg_glucose_level", "bmi", "smoking_status"]]
                Y = dfDistribuida.stroke
                X, Y = ros.fit_resample(X,Y)
                dfDistribuida = X
                dfDistribuida["stroke"] = Y

        df_train, df_test = train_test_split(dfDistribuida, test_size=0.2, random_state=42)

        #en el conjunto de entrenamiento, cambiamos los valores null por la media
        media = df_train['bmi'].median()
        df_train['bmi'] = df_train['bmi'].replace(['N/A'], [media])
        df_train['bmi'] = df_train['bmi'].fillna(media)

        #Prueba con maximo niveles
        ID3_tree = g02_l2_core.ID3_DecisionTree(df_train, None)
        g02_l2_core.SaveId3Tree(f"G02_ID3_tree_Full.txt", ID3_tree)

        predict = g02_l2_core.TestID3Tree(df_test, ID3_tree)
        PrintMetrics(df_test['stroke'].values, predict)




if (args.algoritmo == 'dtc'):
        #prueba con el algoritmo de sklearn
        dfSkLearnDistribuida = df.copy()
        if (args.oversample == '1'):
                ros = oversampler(random_state=42)
                X = dfSkLearnDistribuida[["gender", "age", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type",
                        "avg_glucose_level", "bmi", "smoking_status"]]
                Y = dfSkLearnDistribuida.stroke
                X, Y = ros.fit_resample(X,Y)
                dfSkLearnDistribuida = X
                dfSkLearnDistribuida["stroke"] = Y

        #sklearn no soporta nan, entonces como son pocos, optamos por ponerles tambien la media para estas pruebas
        media = dfSkLearnDistribuida['bmi'].median()
        dfSkLearnDistribuida['bmi'] = dfSkLearnDistribuida['bmi'].replace(['N/A'], [media])
        dfSkLearnDistribuida['bmi'] = dfSkLearnDistribuida['bmi'].fillna(media)

        X = dfSkLearnDistribuida[["gender", "age", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type",
                "avg_glucose_level", "bmi", "smoking_status"]]
        Y = dfSkLearnDistribuida.stroke
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        dt_clf = tree.DecisionTreeClassifier(criterion="entropy")
        dt_clf = dt_clf.fit(X_train, Y_train)
        Y_pred = dt_clf.predict(X_test)
        PrintMetrics(Y_test, Y_pred)

    

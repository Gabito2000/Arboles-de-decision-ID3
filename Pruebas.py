import pandas as pd
import matplotlib.pyplot as plt
import statistics as stat
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler as oversampler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import f1_score as f1
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from sklearn.metrics import confusion_matrix
import math
import g02_l2_core
import numpy as np

#Carga de datos y prueba con sicklearn:
df = pd.read_csv("./healthcare-dataset-stroke-data.csv")

#reajuste de distribucion

"""
sns.countplot(data = df, x="heart_disease")
plt.show()
ros = oversampler(random_state=42)
X = df[["gender", "age", "hypertension", "stroke", "ever_married", "work_type", "Residence_type",
        "avg_glucose_level", "bmi", "smoking_status"]]
Y = df.heart_disease
X, Y = ros.fit_resample(X,Y)
df = X
df["heart_disease"] = Y
sns.countplot(data = df, x="heart_disease")
plt.show()
Accuracy -> 0.8828696925329429
Precision -> 0.8731748018356279
Recall -> 0.9561443581544085
F1 -> 0.9127780200610555
      1       0
 1    2093     304
 0    96       922

"""

"""sns.countplot(data = df, x="hypertension")
plt.show()
ros = oversampler(random_state=42)
X = df[["gender", "age", "heart_disease", "stroke", "ever_married", "work_type", "Residence_type",
        "avg_glucose_level", "bmi", "smoking_status"]]
Y = df.hypertension
X, Y = ros.fit_resample(X,Y)
df = X
df["hypertension"] = Y
sns.countplot(data = df, x="hypertension")
plt.show()
Accuracy -> 0.8505603985056039
Precision -> 0.8399814471243042
Recall -> 0.9306269270298048
F1 -> 0.8829839102876645
      1       0
 1    1811     345
 0    135      921
"""

"""sns.countplot(data = df, x="work_type")
plt.show()
ros = oversampler(random_state=42)
X = df[["gender", "age", "heart_disease", "stroke", "ever_married", "hypertension", "Residence_type",
        "avg_glucose_level", "bmi", "smoking_status"]]
Y = df.work_type
X, Y = ros.fit_resample(X,Y)
df = X
df["work_type"] = Y
sns.countplot(data = df, x="work_type")
plt.show()

Accuracy -> 0.8957522123893805
Precision -> 0.7890861844954525
Recall -> 0.946985446985447
F1 -> 0.8608551854476729
      1       0
 1    1822     487      
 0    102      3239  
"""

"""sns.countplot(data = df, x="gender")
plt.show()
ros = oversampler(random_state=42)
X = df[["work_type", "age", "heart_disease", "stroke", "ever_married", "hypertension", "Residence_type",
        "avg_glucose_level", "bmi", "smoking_status"]]
Y = df.gender
X, Y = ros.fit_resample(X,Y)
df = X
df["gender"] = Y
sns.countplot(data = df, x="gender")
plt.show()

Accuracy -> 0.875481766973021
Precision -> 0.7569493941553813
Recall -> 0.9307624890446976
F1 -> 0.8349056603773584
      1       0
 1    1062     341
 0    79       1891
"""

"""sns.countplot(data = df, x="stroke")
plt.show()
ros = oversampler(random_state=42)
X = df[["gender", "age", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type",
        "avg_glucose_level", "bmi", "smoking_status"]]
Y = df.stroke
X, Y = ros.fit_resample(X,Y)
df = X
df["stroke"] = Y

sns.countplot(data = df, x="stroke")
plt.show()
"""

""
"""sns.countplot(data = df, x="gender")
plt.show()
sns.countplot(data = df, x="age")
plt.show()
sns.countplot(data = df, x="hypertension")
plt.show()
sns.countplot(data = df, x="heart_disease")
plt.show()
sns.countplot(data = df, x="ever_married")
plt.show()
sns.countplot(data = df, x="work_type")
plt.show()
sns.countplot(data = df, x="Residence_type")
plt.show()
sns.countplot(data = df, x="avg_glucose_level")
plt.show()
sns.countplot(data = df, x="bmi")
plt.show()
sns.countplot(data = df, x="smoking_status")
plt.show()"""

#del df["id"]
df['ever_married'] = df['ever_married'].replace(['Yes', 'No'], [1,0])
df['gender'] = df['gender'].replace(['Male', 'Female', 'Other'], [0,1,2])
df['work_type'] = df['work_type'].replace(['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'], [0,1,2,3,4])
df['Residence_type'] = df['Residence_type'].replace(['Urban', 'Rural'], [1,0])
df['smoking_status'] = df['smoking_status'].replace(['formerly smoked', 'never smoked', 'smokes', 'Unknown'], [0,1,2,3])



dfGlobal = df.copy()

#1022 test size porque es el 20%, 5110 datos en total
df_train, df_test = train_test_split(dfGlobal, test_size=0.2, random_state=42)

#en el conjunto de entrenamiento, cambiamos los valores null por la media
media = stat.median(df_train['bmi'])
df_train['bmi'] = df_train['bmi'].replace(['N/A'], [media])
df_train['bmi'] = df_train['bmi'].fillna(media)

#Prueba con 13 niveles
maxTreeLevels = 13
ID3_tree = g02_l2_core.ID3_DecisionTree(df_train, maxTreeLevels)
g02_l2_core.SaveId3Tree(f"G02_ID3_tree_{maxTreeLevels}.txt", ID3_tree)

predict = []

for index, row in df_test.iterrows():
        eval = g02_l2_core.EvaluateTable(row, ID3_tree)
        predict.append(eval)

pred = np.array(predict, dtype=int)
print("Accuracy -> "+str(acc (df_test['stroke'].values, predict)))
print("Precision -> "+str(precision (df_test['stroke'].values, predict)))
print("Recall -> "+str(recall (df_test['stroke'].values, predict)))
print("F1 -> "+str(f1 (df_test['stroke'].values, pred)))
tn, fp, fn, tp = confusion_matrix(df_test['stroke'].values, predict).ravel()
print("      1       0")
print(" 1    {:<8} {:<8} ".format(tp, fp))
print(" 0    {:<8} {:<8} ".format(fn, tn))
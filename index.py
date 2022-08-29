import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import f1_score as f1
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall

warnings.filterwarnings("ignore")

df = pd.read_csv("./healthcare-dataset-stroke-data.csv")

#print (df.head())

#print (df.shape)

#print(df.isnull().sum())

#print (df.describe())

""""
sns.countplot(data = df, x="gender")
plt.show()
sns.countplot(data = df, x="work_type")
plt.show()
sns.countplot(data = df, x="Residence_type")
plt.show()
sns.countplot(data = df, x="smoking_status")
plt.show()
"""


df['bmi'] = df['bmi'].replace(['N/A'], [0.0])
df['bmi'] = df['bmi'].fillna(0.0)
df['ever_married'] = df['ever_married'].replace(['Yes', 'No'], [1,0])
df['gender'] = df['gender'].replace(['Male', 'Female', 'Other'], [0,1,2])
df['work_type'] = df['work_type'].replace(['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'], [0,1,2,3,4])
df['Residence_type'] = df['Residence_type'].replace(['Urban', 'Rural'], [1,0])
df['smoking_status'] = df['smoking_status'].replace(['formerly smoked', 'never smoked', 'smokes', 'Unknown'], [0,1,2,3])

X = df[["gender", "age", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type",
        "avg_glucose_level", "bmi", "smoking_status"]]
Y = df.stroke

#1022 test size porque es el 20%, 5110 datos en total
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1022, random_state=42)

dt_clf = tree.DecisionTreeClassifier(max_depth=10, criterion="entropy")
dt_clf = dt_clf.fit(X_train, Y_train)

Y_pred = dt_clf.predict (X_test)

print("F1 -> "+str(f1 (Y_test, Y_pred)))
print("Accuracy -> "+str(acc (Y_test, Y_pred)))
print("Precision -> "+str(precision (Y_test, Y_pred)))
print("Recall -> "+str(recall (Y_test, Y_pred)))

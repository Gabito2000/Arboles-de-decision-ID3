from ast import For
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
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

dt_clf = tree.DecisionTreeClassifier(max_depth=4, criterion="entropy")
dt_clf = dt_clf.fit(X_train, Y_train)

Y_pred = dt_clf.predict (X_test)

print("F1 -> "+str(f1 (Y_test, Y_pred)))
print("Accuracy -> "+str(acc (Y_test, Y_pred)))
print("Precision -> "+str(precision (Y_test, Y_pred)))
print("Recall -> "+str(recall (Y_test, Y_pred)))

     
	
"""
Fase 1
	- Acomodar datos en 1 archivo solo
	- Ajustar tipos de datos
	- rellenar datos faltantes
	- Todos los valores a reales
	- Enumerados: crear columnas binarias
	- Normalizar los valores
        - los atributos continuos, cambiarlos como se vio en el teoricos, diapo arboles 33

	
Fase 2
	- Separar conjuntos Entrenamiento, Validacion, Test
	- Misma distribucion: aleatorio
	
"""


class G02Tree():
        def __init__(self, idColumn):
                self.IdColumn = idColumn
                self.Nodes= []

class G02TreeSheet(G02Tree):
        def __init__(self, idColumn, returnValue):
                super().__init__(idColumn)
                self.ReturnValue= returnValue

class G02TreeNode():
        #valor, arboles o un valor de retorno
        def __init__(self, value):
                self.Value = value
                self.SubTree = None                

def GetColumnDescriptor(df, col):
        desc = [col]
        isContinuo = col == "age" or col ==  "avg_glucose_level" or col ==  "bmi"
        desc.append(isContinuo)
        if(isContinuo):      #es continuo          
                #tomo solo la columna en cuestion y el stroke
                aux = df[[col, "stroke"]]
                #ordeno por la primer columna
                aux = aux.sort_values(col)
                values = np.array([])
                stroke = -1
                for index, row in aux.iterrows():
                        if(stroke == -1):
                                stroke = row[1]
                        elif(stroke != row[1]):                                
                                stroke = row[1]
                                if(values[values == row[0]].size ==0):
                                        values = np.append(values, row[0])
                        #hay valores que que se repiten y tienen valores diferentes de stroke
                desc.append(values)
        else:
                desc.append(df[col].value_counts().index)
        return desc
           

def GetBestAtt(pdf): #retornar el mejor atributo para separara
        return pdf.columns[0] #por ahora retornamos el primer atributo
        #TODO usar algun algoritmo del teorico, entropia?



dfGlobal = df.copy()
def ID3_DecisionTree(pdf):
        dataf = pdf.copy()
        #Elegir un atributo
        idColumn = GetBestAtt(dataf) # o Afuera de la funcion?
        #Crear una raíz
        ret = G02Tree(idColumn)

        #• Si todos los ej. tienen el mismo valor → etiquetar con ese valor
        uniqueStrokes = dataf.stroke.unique()
        if(uniqueStrokes.size == 0):
                return G02TreeSheet(idColumn, uniqueStrokes[0])
        
        #• Si no me quedan atributos → etiquetar con el valor más común
        if(dataf.columns.size == 1):
                return G02TreeSheet(idColumn, dataf.stroke.mode())

        #• En caso contrario:
        #    ‣ La raíz pregunta por A, atributo que mejor clasifica los ejemplos

        #    ‣ Para cada valor vi de A 
        coldesc = GetColumnDescriptor(dfGlobal, idColumn) #obtiene los posibles valores de una columna, teniendo en cuenta los valores continuos
        oldvalue = 0
        for vi in coldesc[2]: #dfGlobal el df original
                #๏ Genero una rama
                node = G02TreeNode(vi)
                #๏ Ejemplosvi={ejemplos en los cuales A=vi }
                if(coldesc[1]): #es continuo                        
                        ejemplosvi = dataf[dataf[idColumn] >= oldvalue and dataf[idColumn] < vi]
                        oldvalue = vi
                else: 
                        ejemplosvi = dataf[dataf[idColumn] == vi]
                #๏ Si Ejemplosvi es vacío → etiquetar con el valor más probable
                if(len(ejemplosvi) == 0):
                        node.SubTree = G02TreeSheet(idColumn, dataf.stroke.mode())
                else: #En caso contrario → ID3(Ejemplosvi, Atributos -{A})
                        #Atributos -{A}
                        del ejemplosvi[idColumn]
                        node.SubTree = ID3_DecisionTree(ejemplosvi)
                ret.Nodes.append(node)
        
        return ret
                


              


        
         
        #        
        #        
        #        ๏ En caso contrario → ID3(Ejemplosvi, Atributos -{A})
        #         
        return ret


ID3_DecisionTree(df)
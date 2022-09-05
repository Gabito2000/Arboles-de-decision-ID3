from datetime import  datetime
import sys
import math
import numpy as np
import warnings


warnings.filterwarnings("ignore")

#########################################################################################
# ESPECIFICACION DE LAS CLASES PARA LA GENERACION DEL ARBOL
#########################################################################################

class G02Tree():
        def __init__(self, idColumn):
                self.IdColumn = idColumn
                self.Nodes= []
                self.DefaultReturn = 0
                self.DelayTime = 0

class G02TreeSheet(G02Tree):
        def __init__(self, idColumn, returnValue):
                super().__init__(idColumn)
                self.ReturnValue= returnValue

class G02TreeNode():
        #valor, arboles o un valor de retorno
        def __init__(self, value):
                self.Value = value
                self.SubTree = None     
class G02TreeContNode():
        #valor, arboles o un valor de retorno
        def __init__(self, value, isFirstSet):
                self.Value = value
                self.SubTree = None  
                self.IsFirstSet = isFirstSet                      

#########################################################################################
# FUNCIONES UTILES EN EL ALGORITMO ID3
#########################################################################################

def GetColumnDescriptor(df, col):        
        desc = [col]
        isContinuo = col == "age" or col ==  "avg_glucose_level" or col ==  "bmi"
        desc.append(isContinuo)
        if(isContinuo):      #es continuo          
                #tomo solo la columna en cuestion y el stroke
                aux = df[[col, "stroke"]]
                #elimino los repetidos ya que afecta el performance
                aux = aux.drop_duplicates()
                #ordeno por la primer columna                
                aux = aux.sort_values(col)                
                currStroke = -1
                vAnt  = 0 
                bestCutvalue = -1
                bestGain = -1
                for index, row in aux.iterrows():
                        if(currStroke == -1):
                                currStroke = row[1]
                        elif(currStroke != row[1]): #encontre un cambio, deberia calcular la ganancia y guardar ese valor                               
                                currStroke = row[1]                                
                                vCorte = (row[0] + vAnt)/2
                                #calculamos la ganancia de esta divicion
                                ejemplosvi = df[df[col] <= vCorte]
                                gain = 1 - GetEntropy(ejemplosvi)*(len(ejemplosvi.index) / len(df.index) )
                                ejemplosvi = df[df[col] > vCorte]
                                gain = gain - GetEntropy(ejemplosvi)*(len(ejemplosvi.index) / len(df.index) )

                                if(gain > bestGain):
                                        bestGain = gain
                                        bestCutvalue = vCorte
                        vAnt = row[0]                       
                values = np.array([bestCutvalue]) 
                desc.append(values)
        else:
                desc.append(df[col].value_counts().index)
        
        return desc

def GetFullColumnDescriptor(table): 
        descs = {}
        for col in table.columns:
                descs[col] = GetColumnDescriptor(table, col)
        return descs

def GetEntropy(table):
        #Entropia(S) = − p+ . log(p+) − p− . log(p−)
        if(len(table) ==0):
                return 0
        p_plus = len(table[table["stroke"] == 1]) / len(table)
        p_min = len(table[table["stroke"] == 0]) / len(table)
        if(p_min == 1 or p_plus == 1):
                return 1
        entropia = - p_plus*math.log(p_plus) - p_min*math.log(p_min)
        return entropia

def GetAttGanancia(table, colName, colinfo): #Gan(S, Ded) = 1 − 1/4.E(SDed=Alta) − 2/4.E(SDed=Media) − 1/4.E(SDed=Baja)     
        if(colinfo[1]):
                vCorte = colinfo[2][0]
                ejemplosvi = table[table[colName] <= vCorte]
                gAcumul = 1 - GetEntropy(ejemplosvi)*(len(ejemplosvi.index) / len(table.index) )
                ejemplosvi = table[table[colName] > vCorte]
                gAcumul = gAcumul - GetEntropy(ejemplosvi)*(len(ejemplosvi.index) / len(table.index) )            
                return gAcumul
        else:
                gAcumul = 1
                for vi in colinfo[2]:
                        ejemplosvi = table[table[colName] == vi]
                        gAcumul = gAcumul - GetEntropy(ejemplosvi)*(len(ejemplosvi.index) / len(table.index) )

                return gAcumul


def GetBestAtt(table, colinfos): #retornar el mejor atributo para aplicar con Id3, tambien la descripcion de las columnas por performance
        bestAttr = table.columns[0]
        bestGan = -1
        for col in table.columns:
                if(col  != "stroke"):
                        g = GetAttGanancia(table, col, colinfos[col])
                        if(g>bestGan):
                                bestGan = g
                                bestAttr = col

        return bestAttr


#########################################################################################
# ALGORITMO ID3
#########################################################################################
def ID3_DecisionTree(pdf, maxLevels):
        FullColInfos = GetFullColumnDescriptor(pdf)
        if(maxLevels == None):
                maxLevels = sys.maxsize
        return __ID3_DecisionTree(pdf, maxLevels, FullColInfos)

def __ID3_DecisionTree(pdf, maxLevels, FullColInfos):
        dateInit = datetime.now()
        if(maxLevels ==0):
                return G02TreeSheet("ID3 Max Level", pdf["stroke"].mode().values[0])

        dataf = pdf.copy()
        
        #• Si todos los ej. tienen el mismo valor → etiquetar con ese valor
        uniqueStrokes = dataf.stroke.unique()
        if(uniqueStrokes.size == 1):
                return G02TreeSheet("SAME RETURN", uniqueStrokes[0])
        
        #• Si no me quedan atributos → etiquetar con el valor más común
        if(dataf.columns.size == 1):
                return G02TreeSheet("EMPTY_ATTS", dataf["stroke"].mode().values[0])
        
        idColumn = GetBestAtt(dataf, FullColInfos) #Elegir un atributo y retorna la descripcion de valores para esa eleccion
        colInfo = FullColInfos[idColumn]        

        #Crear una raíz
        ret = G02Tree(idColumn)
        ret.DefaultReturn = dataf["stroke"].mode().values[0]                

        #    ‣ Para cada valor vi de A 
        if(colInfo[1]):#es continuo
                #๏ Genero una rama
                vi = colInfo[2][0]
                node = G02TreeContNode(vi, True)  
                #creo arbol menor
                ejemplosvi = dataf[dataf[idColumn] <= vi]
                #๏ Si Ejemplosvi es vacío → etiquetar con el valor más probable
                if(len(ejemplosvi) == 0):
                        node.SubTree = G02TreeSheet("MEDIA", dataf["stroke"].mode().values[0])
                elif(ejemplosvi.stroke.unique().size ==1):
                        #• Si todos los ej. tienen el mismo valor → etiquetar con ese valor
                        node.SubTree = G02TreeSheet("UNIQUE", ejemplosvi.stroke.unique()[0])
                else: #En caso contrario → ID3(Ejemplosvi, Atributos -{A})                        
                        del ejemplosvi[idColumn] #Atributos -{A}
                        node.SubTree = __ID3_DecisionTree(ejemplosvi, maxLevels-1, FullColInfos)
                ret.Nodes.append(node)

                # creo arbol mayor 
                node = G02TreeContNode(vi, False)  
                ejemplosvi = dataf[dataf[idColumn] > vi]
                #๏ Si Ejemplosvi es vacío → etiquetar con el valor más probable
                if(len(ejemplosvi) == 0):
                        node.SubTree = G02TreeSheet("MEDIA", dataf["stroke"].mode().values[0])
                elif(ejemplosvi.stroke.unique().size ==1):
                        #• Si todos los ej. tienen el mismo valor → etiquetar con ese valor
                        node.SubTree = G02TreeSheet("UNIQUE", ejemplosvi.stroke.unique()[0])
                else: #En caso contrario → ID3(Ejemplosvi, Atributos -{A})                        
                        del ejemplosvi[idColumn] #Atributos -{A}
                        node.SubTree = __ID3_DecisionTree(ejemplosvi, maxLevels-1, FullColInfos)
                ret.Nodes.append(node)  
        else:
                for vi in colInfo[2]: #dfGlobal el df original
                        #๏ Ejemplosvi={ejemplos en los cuales A=vi }
                        node = G02TreeNode(vi)
                        ejemplosvi = dataf[dataf[idColumn] == vi]
                        #๏ Si Ejemplosvi es vacío → etiquetar con el valor más probable
                        if(len(ejemplosvi) == 0):
                                node.SubTree = G02TreeSheet("MEDIA", dataf["stroke"].mode().values[0])
                        elif(ejemplosvi.stroke.unique().size ==1):
                                #• Si todos los ej. tienen el mismo valor → etiquetar con ese valor
                                node.SubTree = G02TreeSheet("UNIQUE", ejemplosvi.stroke.unique()[0])
                        else: #En caso contrario → ID3(Ejemplosvi, Atributos -{A})                                
                                del ejemplosvi[idColumn] #Atributos -{A}
                                node.SubTree = __ID3_DecisionTree(ejemplosvi, maxLevels-1, FullColInfos)
                        ret.Nodes.append(node)
        
        time = datetime.now() - dateInit
        ret.DelayTime = math.ceil(time.total_seconds()*1000)
        return ret

#########################################################################################
# FUNCIONES SOBRE EL ARBOL ID3
#########################################################################################
                
def __EvaluateTable(item, id3Tree):        
        if(type(id3Tree) is G02TreeSheet):
                return id3Tree.ReturnValue
        v = item[id3Tree.IdColumn]
        for node in id3Tree.Nodes:                
                if(type(node) == G02TreeContNode):    
                        if(node.IsFirstSet and v <= node.Value):
                                return __EvaluateTable(item, node.SubTree) 
                        if((not node.IsFirstSet) and v > node.Value):
                                return __EvaluateTable(item, node.SubTree)                                           
                else:
                        if(v == node.Value):
                                return __EvaluateTable(item, node.SubTree)

        #Retornar un valor por defecto, por ejemplo el promedio de las hojas
        return id3Tree.DefaultReturn

def TestID3Tree(X_Test, id3Tree):        
        predict = []
        for index, row in X_Test.iterrows():
                eval = __EvaluateTable(row, id3Tree)
                predict.append(eval)

        return np.array(predict, dtype=int)

def TreeToString(id3Tree, nivel):
    ret = ""
    for node in id3Tree.Nodes: 
        for i in range(0, nivel):
            ret += "       "               
        if(type(node) == G02TreeContNode):
            ret += id3Tree.IdColumn + " "   
            ret +="<= " if node.IsFirstSet else "> " + " "
            ret += f"{node.Value}" + " "                                                            
        else:
            ret +=id3Tree.IdColumn + " "
            ret +=f"{node.Value}" + " "
        
        if(type(node.SubTree) is G02TreeSheet):            
            ret +="===> YES - "+ node.SubTree.IdColumn + "\n" if node.SubTree.ReturnValue == 1 else "===> NO - "+ node.SubTree.IdColumn + "\n"
        else:
                if(node.SubTree.DelayTime> 2000):
                        ret += "(" + str(node.SubTree.DelayTime/1000) + " s) " 
                else: 
                        ret += "(" + str(node.SubTree.DelayTime) + " ms) "
                ret +="\n"
                ret +=TreeToString(node.SubTree, nivel + 1)   
    return ret          

def SaveId3Tree(fileName, id3tree):
    strTree = TreeToString(id3tree, 0)
    if(id3tree.DelayTime> 2000):
        strTree = "Tiempo Total " + str(id3tree.DelayTime/1000) + " segundos \n" + strTree
    else: 
        strTree += "Tiempo Total " + str(id3tree.DelayTime) + " milisegundos \n" + strTree
    strTree = strTree
    f = open(fileName, "w")
    f.write(strTree)
    f.close()


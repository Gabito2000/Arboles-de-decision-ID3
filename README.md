# Arboles-de-decision-ID3

Tarea 2 de aprendizage automatico

Para ejecutar la solución previamente se deben instalar las dependencias de la misma, las cuáles están en el archivo "requirements.txt".
Para ésto corremos el siguiente comando.

    pip install -r .\requirements.txt

Luego se corre la solución ejecutando el comando.

    py index.py --algoritmo=A --oversample=O

Donde "A" denota el algoritmo a utiliza, y toma los valores "g02" y "dtc", el primer valor se utiliza para correr el algoritmo implementado por el grupo 02, y el segundo para correr el algoritmo DecisionTreeClassifier de la librería sklearn.
Y "O" toma un valor entero que indica si se utiliza un oversampleo de los datos o no, 0 para no utilizar oversampleo y cualquier otro valor para utilizarlo.

Un ejemplo de ejecución con éstos parámetros es el siguiente.

    py index.py --oversample=1 --algoritmo=g02

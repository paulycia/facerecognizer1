import cv2
import os
import numpy as np
dataPath = 'C:/Users/DESKTOP/Desktop/TI-2023/ReconocimientoFacial/DATOS'#Cambia a la ruta donde hayas almacenado Data
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)
labels = []
facesData = []
label = 0
for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print('Leyendo las imágenes_____________')
    for fileName in os.listdir(personPath):
        print('Rostros: ', nameDir + '/' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(personPath+'/'+fileName,0))
        image = cv2.imread(personPath+'/'+fileName,0) ## usamos escala de grises con 0
        cv2.imshow('image',image)
        cv2.waitKey(10)
    label = label + 1
print('labels= ',labels)
print(' version Opencv '+ cv2.__version__)
#print(help(cv2.face))
print('Número de etiquetas 0: ',np.count_nonzero(np.array(labels)==0))
print('Número de etiquetas 1: ',np.count_nonzero(np.array(labels)==1))
print('Número de etiquetas 2: ',np.count_nonzero(np.array(labels)==2))
#####_________creando archivo de entrenamiento __________________________
#face_recognizer = cv2.face.EigenFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
#face_recognizer = cv2.face.EigenFaceRecognizer_create()

##________Entrenando El reconocedor____________
print("Entrenando________espere unos segundos por favor__")
face_recognizer.train(facesData,np.array(labels))
##__________Guardando modelo Obtenido__________
face_recognizer.write('modeloLBPHFace.xml')
#face_recognizer.write('modeloEigenFace1.xml')
#face_recognizer.write('modeloEigenFace1.xml')

##___________________________________________________

import cv2
import os
import imutils
from gtts import gTTS

#####################################################################
###  Este programa lee un video y genera las caras para el algoritmo.
###  Usuario debe ingresar nombre del rostro y nombre del Video.
#######################################################################
speech = gTTS('Por favor ingrese nombre del rostro que agregaremos al programa de reconocimiento facial')
personName = input(" Por favor ingrese nombre de nuevo Rostro: ")
archivo_MP4 = input(" Por favor ingrese nombre del Video (Video debe estar en C:/Users/DESKTOP/Desktop/TI-2023/ReconocimientoFacial/   ')  ")

#dataPath = 'C:/Users/Gaby/Desktop/Reconocimiento Facial/Data'#Cambia a la ruta donde hayas almacenado Data
dataPath = 'C:/Users/DESKTOP/Desktop/TI-2023/ReconocimientoFacial/DATOS'
#personPath = dataPath + '/' + archivo_MP4
personPath = dataPath + '/' + personName
print('Carpeta a crear: ',personPath)
if not os.path.exists(personPath):
    print('Carpeta creada: ',personPath)
    os.makedirs(personPath)

#cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap = cv2.VideoCapture('C:/Users/DESKTOP/Desktop/TI-2023/ReconocimientoFacial/'+archivo_MP4)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
count = 0

while True:
    
    ret, frame = cap.read()
    if ret == False: break
    frame =  imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(personPath + '/rotro_{}.jpg'.format(count),rostro)
        count = count + 1
    cv2.imshow('frame',frame)

    k =  cv2.waitKey(3)
    if k == 27 or count >= 300:
        break

cap.release()
cv2.destroyAllWindows()
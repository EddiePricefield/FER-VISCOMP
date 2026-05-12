from contextlib import nullcontext

import cv2
import matplotlib

## motor gráfico backend pra funcionar o .show() no arch ##
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

## XML Disponibilizado pela OpenCV para reconhecimento facial (2001) ##
face_cascade = cv2.CascadeClassifier('XMLs/haarcascade_frontalface_default.xml')

## Importando o video ##
cam = cv2.VideoCapture(1)
img_erro = cv2.imread('imagens/error.png')

if not cam.isOpened():
    cam = cv2.VideoCapture(0)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

## Para evitar que ele pegue apenas o primeiro frame do video, precisa do loop ##
while True:
    ret, frame = cam.read()

    if not ret or frame is None:
        frame = img_erro.copy()

    ## Transformando em Tons de Cinza, para auxiliar na detecção ##
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ## Aplicando a Transformação CLAHE para testes adicionais ##
    gray_clahe = clahe.apply(gray)
    cv2.imshow("clahe", gray_clahe)

    ## Usando o detector ##
    faces = face_cascade.detectMultiScale(gray_clahe, scaleFactor=1.15, minNeighbors=7, minSize=(25, 25))

    ## Desenhando os retângulos onde ele detectou os rostos ##
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Resultado do Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cam.release()
cv2.destroyAllWindows()
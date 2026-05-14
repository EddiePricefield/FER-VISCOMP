import cv2
import matplotlib

## motor gráfico backend pra funcionar o .show() no arch ##
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

## XML Disponibilizado pela OpenCV para reconhecimento facial (2001) ##
face_cascade = cv2.CascadeClassifier('XMLs/haarcascade_frontalface_default.xml')

## Importando o video ##
video = cv2.VideoCapture('videos/vines.mp4')
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

## Para evitar que ele pegue apenas o primeiro frame do video, precisa do loop ##
while True:
    ret, frame = video.read()

    frame_resized = cv2.resize(frame, None, fx=0.5, fy=0.5)

    ## Transformando em Tons de Cinza, para auxiliar na detecção ##
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    ## Aplicando a Transformação CLAHE para testes adicionais ##
    gray_clahe = clahe.apply(gray)

    ## Usando o detector ##
    faces = face_cascade.detectMultiScale(gray_clahe, scaleFactor=1.15, minNeighbors=6, minSize=(30, 30))

    ## Desenhando os retângulos onde ele detectou os rostos ##
    for (x, y, w, h) in faces:
        cv2.rectangle(frame_resized, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Resultado do Video", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

video.release()
cv2.destroyAllWindows()
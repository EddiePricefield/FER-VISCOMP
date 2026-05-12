import cv2
import matplotlib

## motor gráfico backend pra funcionar o .show() no arch ##
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

## XML Disponibilizado pela OpenCV para reconhecimento facial (2001) ##
face_cascade = cv2.CascadeClassifier('XMLs/haarcascade_frontalface_default.xml')

## Importando a imagem e redimensionando para ficar melhor na tela do meu PC ##
img = cv2.imread("imagens/rostos.avif")
img_resized = cv2.resize(img, None, fx=0.75, fy=0.75)
frame = img_resized.copy()

## Transformando em Tons de Cinza, para auxiliar na detecção ##
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

## Aplicando a Transformação CLAHE para testes adicionais ##
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray_clahe = clahe.apply(gray)

## Usando o detector ##
faces = face_cascade.detectMultiScale(gray_clahe, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

## Desenhando os retângulos onde ele detectou os rostos ##
for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

## Plotando as imagens usadas e o resultado final ##

imagens = [img_resized, gray, gray_clahe]
titulos = ["Original", "Grayscale", "Clahe"]

fig, axes = plt.subplots(1, 3, figsize=(10, 6), num="imagens utilizadas")
for i in range(3):
    img_rgb = cv2.cvtColor(imagens[i], cv2.COLOR_BGR2RGB) ## Pra ficar certo no matplotlib, to tendo que transformar de BGR pra RGB
    axes[i].imshow(img_rgb)
    axes[i].set_title(titulos[i])
    axes[i].axis('off')

plt.tight_layout()
plt.show()

## Plotando o resultado final ##

plt.figure(num="Resultado do Teste", figsize=(10, 5))
img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb, aspect='equal')
plt.tight_layout(pad=0)
plt.axis('off')
plt.show()
import cv2
import matplotlib
import numpy
import onnxruntime

## motor gráfico backend pra funcionar o .show() no arch ##
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

## Detector de Rosto YuNet, que usa o ONNX nativamente pelo OpenCV ##
detector_rostos = cv2.FaceDetectorYN.create("ONNXs/face_detection_yunet_2023mar.onnx", "", (0, 0))

## Selecionando o modelo ONNX ##
session = onnxruntime.InferenceSession("ONNXs/emotion-ferplus-12-int8.onnx")

input_name = session.get_inputs()[0].name
emocoes = ['Neutro', 'Feliz', 'Surpreso', 'Triste', 'Bravo', 'Nojo', 'Medo', 'Deboche']

## Importando a imagem e redimensionando para ficar melhor na tela do meu PC ##
img = cv2.imread("imagens/rostos.avif")
# img_resized = cv2.resize(img, None, fx=0.75, fy=0.75)
img_resized = img.copy()
frame = img_resized.copy()

## Transformando em Tons de Cinza, para auxiliar na detecção ##
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

## Aplicando a Transformação CLAHE para testes adicionais ##
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray_clahe = clahe.apply(gray)

## Passando o tamanho atual do frame para o YuNet, porque ele precisa saber disso ##
altura, largura = frame.shape[:2]
detector_rostos.setInputSize((largura, altura))

## O YuNet consegue identificar o rosto com a imagem colorida, então não precisa de pré-processamento ##
_, faces = detector_rostos.detect(frame)

## Desenhando os retângulos onde ele detetou os rostos e aplicando a deteção de emoções ##
if faces is not None:
    for rosto in faces:
        # O YuNet devolve várias posições, até dos olhos... Mas os 4 primeiros são o suficiente ##
        x, y, w, h = int(rosto[0]), int(rosto[1]), int(rosto[2]), int(rosto[3])

        ## Trava de segurança: evita que coordenadas negativas fora da tela quebrem o código ##
        x, y = max(0, x), max(0, y)
        x2, y2 = min(largura, x + w), min(altura, y + h)

        ## Retangulinho azul no rosto detetado ##
        cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)

        ## Prepara o recorte da face para o modelo ONNX
        face = gray_clahe[y:y2, x:x2]
        face_resized = cv2.resize(face, (64, 64))
        face_processed = numpy.array(face_resized, dtype=numpy.float32).reshape(1, 1, 64, 64)

        ## Faz a predição e escreve o texto
        predicoes = session.run(None, {input_name: face_processed})
        emocao_detectada = emocoes[numpy.argmax(predicoes[0])]
        cv2.putText(frame, emocao_detectada, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

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

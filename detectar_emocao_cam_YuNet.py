import cv2
import numpy
import onnxruntime

## Detector de Rosto YuNet, que usa o ONNX nativamente pelo OpenCV ##
detector_rostos = cv2.FaceDetectorYN.create("ONNXs/face_detection_yunet_2023mar.onnx", "", (0, 0))

## Selecionando o modelo ONNX ##
session = onnxruntime.InferenceSession("ONNXs/emotion-ferplus-12-int8.onnx")

input_name = session.get_inputs()[0].name
emocoes = ['Neutro', 'Feliz', 'Surpreso', 'Triste', 'Bravo', 'Nojo', 'Medo', 'Deboche']

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

    ## Redimensionar a tela, para usar menos processamento ##
    # frame = cv2.resize(frame, None, fx=0.75, fy=0.75)

    ## Passando o tamanho atual do frame para o YuNet, porque ele precisa saber disso ##
    altura, largura = frame.shape[:2]
    detector_rostos.setInputSize((largura, altura))

    ## O YuNet consegue identificar o rosto com a imagem colorida, então não precisa de pré-processamento ##
    _, faces = detector_rostos.detect(frame)

    ## Transformando em Tons de Cinza, para auxiliar na predição das emoções ##
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ## Aplicando a Transformação CLAHE para testes adicionais ##
    gray_clahe = clahe.apply(gray)

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

    cv2.imshow("Resultado do Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cam.release()
cv2.destroyAllWindows()
import os
from enum import Enum

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import sys
import cv2
import numpy
import onnxruntime
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6 import uic
from enum import Enum, auto
from pathlib import Path

class TelaInicial(QMainWindow):

    def __init__(self):
        super().__init__()
        uic.loadUi('UIs/tela_inicial.ui', self)

        self.btnWebcam.clicked.connect(self.abrir_webcam)
        self.btnArquivo.clicked.connect(self.abrir_arquivos)

        self.tela_webcam = None
        self.tela_arquivos = None
        self.arqErrado = True

    def abrir_webcam(self):
        if self.tela_webcam is None:
            self.tela_webcam = TelaWebcam(self)

        self.tela_webcam.show()
        self.tela_webcam.iniciar_deteccao = True
        self.tela_webcam.iniciar_cam()
        self.hide()

    def abrir_arquivos(self):
        if self.tela_arquivos is None:
            self.tela_arquivos = TelaArquivos(self)

        self.tela_arquivos.selecionar_arquivo()

        if not self.arqErrado:
            self.tela_arquivos.show()
            self.hide()

class TelaArquivos(QMainWindow):

    class TipoArquivo(Enum):
        IMAGEM = auto()
        VIDEO = auto()
        DESCONHECIDO = auto()

    def __init__(self, tela_inicial):
        super().__init__()
        uic.loadUi('UIs/tela_arquivos.ui', self)

        self.tela_inicial = tela_inicial
        self.btnVoltarTela.clicked.connect(self.voltar)
        self.btnSelecArquivo.clicked.connect(self.selecionar_arquivo)
        self.tipo_arquivo = self.TipoArquivo.DESCONHECIDO
        self.caminho = None
        self.cam = None
        self.timer = None

    def selecionar_arquivo(self):

        ## Explorador de Arquivos nativo do Qt6 ##
        caminho_arquivo, _ = QFileDialog.getOpenFileName(
            self,
            "Selecione uma Imagem ou Vídeo",
            "",
            "Arquivo de Imagem (*.png *.jpg *.jpeg);; Arquivo de Vídeo (*.mp4 *.avi *.mkv);;Todos os Arquivos (*)"
        )

        ## Caso feche sem selecionar algo ##
        if not caminho_arquivo:
            self.tela_inicial.arqErrado = True
            return None, None

        extImg = ['.png', '.jpg', '.jpeg', '.bmp', '.webm', '.avif']
        extVideo = ['.mp4', '.avi', '.mkv', '.mov']

        if Path(caminho_arquivo).suffix.lower() in extImg:
            tipo_arquivo = self.TipoArquivo.IMAGEM
            self.tela_inicial.arqErrado = False
        elif Path(caminho_arquivo).suffix.lower() in extVideo:
            tipo_arquivo = self.TipoArquivo.VIDEO
            self.tela_inicial.arqErrado = False
        else:
            tipo_arquivo = self.TipoArquivo.DESCONHECIDO
            self.tela_inicial.arqErrado = True

        self.iniciar_arquivo(caminho_arquivo, tipo_arquivo)

    def iniciar_arquivo(self, caminho, tipo_arquivo):
        if tipo_arquivo is self.TipoArquivo.DESCONHECIDO:
            QMessageBox.information(self, "Erro: Arquivo não suportado", "O arquivo selecionado possui uma extensão não suportada pelo programa! Selecione outro arquivo.")
            return

        ## Carregar Modelos ##
        self.detector_rostos = cv2.FaceDetectorYN.create("ONNXs/face_detection_yunet_2023mar.onnx", "", (0, 0))
        self.session = onnxruntime.InferenceSession("ONNXs/emotion-ferplus-12-int8.onnx")

        self.input_name = self.session.get_inputs()[0].name
        self.emocoes = ['Neutro', 'Feliz', 'Surpreso', 'Triste', 'Bravo', 'Nojo', 'Medo', 'Deboche']

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        if tipo_arquivo is self.TipoArquivo.IMAGEM:
            img = cv2.imread(caminho)
            self.processar_arquivo(img)

        elif tipo_arquivo is self.TipoArquivo.VIDEO:
            self.cam = cv2.VideoCapture(caminho)

            if not self.cam.isOpened():
                self.img_erro = cv2.imread('imagens/error.png')
                return

            # Inicia o loop
            self.timer = QTimer()
            self.timer.timeout.connect(self.atualizar_frame_video)
            self.timer.start(30)

    def atualizar_frame_video(self):
        if self.cam is None or not self.cam.isOpened():
            return

        ret, frame = self.cam.read()

        if not ret or frame is None:
            self.timer.stop()
            return

        self.processar_arquivo(frame)

    def processar_arquivo(self, frame):

        altura, largura = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_clahe = self.clahe.apply(gray)

        self.detector_rostos.setInputSize((largura, altura))
        _, faces = self.detector_rostos.detect(frame)

        self.ratio = 1.0

        if largura > self.viewArquivo.width() or altura > self.viewArquivo.height():
            self.ratio = min(self.viewArquivo.width() / largura, self.viewArquivo.height() / altura)

        frame_resized = cv2.resize(frame, (int(largura * self.ratio), int(altura * self.ratio)))

        if faces is not None:
            for rosto in faces:
                x, y, w, h = int(rosto[0]), int(rosto[1]), int(rosto[2]), int(rosto[3])
                x, y = max(0, x), max(0, y)
                x2, y2 = min(largura, x + w), min(altura, y + h)

                face = gray_clahe[y:y2, x:x2]

                if face.shape[0] == 0 or face.shape[1] == 0:
                    continue

                face_resized = cv2.resize(face, (64, 64))
                face_processed = numpy.array(face_resized, dtype=numpy.float32).reshape(1, 1, 64, 64)
                predicoes = self.session.run(None, {self.input_name: face_processed})
                emocao_detectada = self.emocoes[numpy.argmax(predicoes[0])]

                x_tela = int(x * self.ratio)
                y_tela = int(y * self.ratio)
                x2_tela = int(x2 * self.ratio)
                y2_tela = int(y2 * self.ratio)

                cv2.rectangle(frame_resized, (x_tela, y_tela), (x2_tela, y2_tela), (255, 0, 0), 2)

                pos_y_texto = y_tela - 5

                if pos_y_texto < 10:
                    pos_y_texto = y_tela + 15

                cv2.putText(frame_resized, emocao_detectada, (x_tela - 1, pos_y_texto + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (111, 111, 111), 2)
                cv2.putText(frame_resized, emocao_detectada, (x_tela, pos_y_texto), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        rgb_image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB).astype(numpy.uint8)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w

        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.viewArquivo.setPixmap(pixmap)

    def voltar(self):
        self.detector_rostos = None
        self.session = None

        if self.timer is not None:
            self.timer.stop()

        if self.cam is not None:
            self.cam.release()

        self.tela_inicial.show()
        self.hide()

class TelaWebcam(QMainWindow):

    def __init__(self, tela_inicial):
        super().__init__()
        uic.loadUi('UIs/tela_webcam.ui', self)

        self.spinLargura.valueChanged.connect(self.alterar_tamanho_cam)
        self.spinAltura.valueChanged.connect(self.alterar_tamanho_cam)

        self.tela_inicial = tela_inicial
        self.iniciar_deteccao = True

        self.btnVoltarTela.clicked.connect(self.voltar)

    def iniciar_cam(self):
        if not self.iniciar_deteccao:
            return

        ## Verificar Câmeras Disponíveis ##
        self.cameras = self.procurar_cam(max_index=5) or [0]
        self.camAtual = self.cameras[0]
        self.comboCamera.clear()
        for index in self.cameras:
            self.comboCamera.addItem(f"Câmera {index}", index)
        self.comboCamera.setCurrentIndex(0)
        self.comboCamera.currentIndexChanged.connect(self.mudar_cam)

        self.viewCamera.setScaledContents(True)
        self.viewCamera.setFixedSize(self.spinLargura.value(), self.spinAltura.value())

        ## Configurar Câmera ##
        self.cam = cv2.VideoCapture(self.camAtual)

        ## Carregar Modelos ##
        self.detector_rostos = cv2.FaceDetectorYN.create("ONNXs/face_detection_yunet_2023mar.onnx", "", (0, 0))
        self.session = onnxruntime.InferenceSession("ONNXs/emotion-ferplus-12-int8.onnx")

        self.input_name = self.session.get_inputs()[0].name
        self.emocoes = ['Neutro', 'Feliz', 'Surpreso', 'Triste', 'Bravo', 'Nojo', 'Medo', 'Deboche']

        self.img_erro = cv2.imread('imagens/error.png')
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        ## Timer ##
        self.timer = QTimer()
        self.timer.timeout.connect(self.atualizar_frame)
        self.timer.start(30)

    def atualizar_frame(self):
        if self.cam is None or not self.cam.isOpened():
            return

        if not self.iniciar_deteccao:
            return

        ret, frame = self.cam.read()

        if not ret or frame is None:
            if self.img_erro is not None:
                frame = self.img_erro.copy()
            else:
                return

        cv2.resize(frame, (self.spinLargura.value(), self.spinAltura.value()))

        altura, largura = frame.shape[:2]
        self.detector_rostos.setInputSize((largura, altura))
        _, faces = self.detector_rostos.detect(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_clahe = self.clahe.apply(gray)

        if faces is not None:
            for rosto in faces:
                x, y, w, h = int(rosto[0]), int(rosto[1]), int(rosto[2]), int(rosto[3])

                x, y = max(0, x), max(0, y)
                x2, y2 = min(largura, x + w), min(altura, y + h)

                ## Desenhar o Retângulo ##
                if self.checkRetangulo.isChecked():
                    cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)

                face = gray_clahe[y:y2, x:x2]
                face_resized = cv2.resize(face, (64, 64))
                face_processed = numpy.array(face_resized, dtype=numpy.float32).reshape(1, 1, 64, 64)

                predicoes = self.session.run(None, {self.input_name: face_processed})
                emocao_detectada = self.emocoes[numpy.argmax(predicoes[0])]

                ## Desenhar a Emoção Acima do Retângulo ##
                if self.checkTexto.isChecked():
                    cv2.putText(frame, emocao_detectada, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        ## Conversão do PixMap para exibir no QLabel ##
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(numpy.uint8)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.viewCamera.setPixmap(pixmap)
        self.viewCamera.repaint()

    def procurar_cam(self, max_index=5):
        available = []
        for index in range(max_index + 1):
            cam = cv2.VideoCapture(index)
            if cam.isOpened():
                available.append(index)
                cam.release()
            else:
                print("Não encontrou câmera no index: ", index)
        return available

    def mudar_cam(self, combo_index):
        camera_index = self.comboCamera.itemData(combo_index)
        if camera_index is None or camera_index == self.camAtual:
            return

        self.camAtual = camera_index
        if self.cam is not None and self.cam.isOpened():
            self.cam.release()
            self.cam = cv2.VideoCapture(self.camAtual)
            if not self.cam.isOpened():
                print(f"Não foi possível acessar a câmera {self.camAtual}.")

    def trocar_Cam(self, combo_index):
        if combo_index < 0:
            return
        camera_index = self.comboCamera.itemData(combo_index)
        if camera_index is None:
            return
        if camera_index == self.camAtual:
            return
        self.camAtual = camera_index

        if self.cam is not None and self.cam.isOpened():
            self.cam.release()

        self.cam = cv2.VideoCapture(self.camAtual)
        if not self.cam.isOpened():
            print(f"Não foi possível acessar a câmera {self.camAtual}.")

    def alterar_tamanho_cam(self):
        self.viewCamera.setFixedSize(self.spinLargura.value(), self.spinAltura.value())

    def voltar(self):
        self.iniciar_deteccao = False
        self.timer.stop()
        self.detector_rostos = None
        self.session = None

        if self.cam is not None:
            self.cam.release()

        self.tela_inicial.show()
        self.hide()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TelaInicial()
    window.show()
    sys.exit(app.exec())
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6 import uic

class TelaInicial(QMainWindow):

    def __init__(self):
        super().__init__()
        uic.loadUi('UIs/tela_inicial.ui', self)

        self.btnWebcam.clicked.connect(self.abrir_webcam)

        self.tela_webcam = None

    def abrir_webcam(self):
        if self.tela_webcam is None:
            self.tela_webcam = TelaWebcam(self)

        self.tela_webcam.iniciar()
        self.tela_webcam.show()
        self.hide()


class TelaWebcam(QMainWindow):

    def __init__(self, tela_inicial):
        super().__init__()
        uic.loadUi('UIs/tela_webcam.ui', self)

        self.tela_inicial = tela_inicial

        self.btnVoltarTela.clicked.connect(self.voltar)

    def iniciar(self):
        print("Iniciar a detecção e análise de faces pela webcam")

    def parar(self):
        print("Parar a detecção e análise de faces pela webcam")

    def voltar(self):
        self.parar()
        self.tela_inicial.show()
        self.hide()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TelaInicial()
    window.show()
    sys.exit(app.exec())
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6 import uic

class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Carrega o ficheiro .ui
        uic.loadUi('UIs/tela_inicial.ui', self)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec())
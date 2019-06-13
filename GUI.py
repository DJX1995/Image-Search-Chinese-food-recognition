import sys
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QPushButton, QFileDialog, QWidget, QVBoxLayout
from PyQt5.QtGui import QPixmap
import numpy as np


class Window(QWidget):

    def __init__(self):
        super().__init__()
        self.init_window()
        self.show_index = 0

    def init_window(self):
        #hbox = QHBoxLayout(self)
        #layout = QVBoxLayout()

        self.setGeometry(250, 100, 1600, 900)
        self.setWindowTitle("Chinese Food Recognition")

        # create buttons
        self.btn1 = QPushButton("Show Sample Images", self)
        self.btn1.move(50, 20)
        self.btn1.clicked.connect(self.bt_show_image)
        self.btn2 = QPushButton("Open a Target Image", self)
        self.btn2.move(50, 350)
        self.btn2.clicked.connect(self.bt_open_image)
        self.btn3 = QPushButton("Show Recognition Result", self)
        self.btn3.move(50, 700)
        self.btn3.clicked.connect(self.lb_show_result)

        # create a list of labels for the purpose of displaying sample images
        self.lb_sp_im = []
        for i in range(5):
            self.lb_sp_im.append(QLabel(self))
            self.lb_sp_im[i].setFixedSize(280, 200)
            self.lb_sp_im[i].move(300*i+50, 100)

        # create a label showing an image to recognize
        self.lb_target = QLabel(self)
        self.lb_target.setFixedSize(300, 200)
        self.lb_target.move(50, 440)

        # create a label to display the recognition result
        self.lb_result = QLabel(self)
        self.lb_result.setFixedSize(100, 100)
        self.lb_result.move(55, 700)


    def bt_show_image(self):
        self.show_index = np.random.randint(3, 1000)
        for i in range(5):
            self.lb_sp_im[i].setPixmap(QPixmap("./resultscang/{}.jpg".format(np.random.randint(3, 1000))).scaled(280, 200))

    def bt_open_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Image", "./", 'Image files(*.jpg *.gif *.png)')
        self.lb_target.setPixmap(QPixmap(fname).scaled(self.lb_target.width(), self.lb_target.height()))

    def lb_show_result(self):
        self.lb_result.setText("Mapo Tofu")



if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())



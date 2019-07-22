import sys
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QFileDialog, QWidget, QTextEdit
from PyQt5.QtGui import QPixmap
import numpy as np
import CNN_Net
from keras.preprocessing.image import img_to_array, load_img
from load import normalize


class Window(QWidget):

    def __init__(self):
        super().__init__()
        self.init_window()
        self.show_index = 0
        self.cnn = CNN_Net.define_CNN()
        self.image_path = None
        self.food_list = {
            0: "Potato silk",
            1: "Baby Cabbage",
            2: "Mapo Tofu",
            3: "Fried beans"
        }

    def init_window(self):

        self.setGeometry(250, 100, 1000, 400)
        self.setWindowTitle("Chinese Food Recognition")

        # create buttons
        self.btn1 = QPushButton("Show Recognition Result", self)
        self.btn1.move(400, 50)
        self.btn1.clicked.connect(self.bt_show_image)
        self.btn2 = QPushButton("Open a Target Image", self)
        self.btn2.move(50, 50)
        self.btn2.clicked.connect(self.bt_open_image)

        # create a list of labels for the purpose of displaying sample images
        self.lb_sp_im = []
        for i in range(3):
            self.lb_sp_im.append(QLabel(self))
            self.lb_sp_im[i].setFixedSize(280, 200)
            self.lb_sp_im[i].move(300*i+50, 440)

        # create a label showing an image to recognize
        self.lb_target = QLabel(self)
        self.lb_target.setFixedSize(100, 100)
        self.lb_target.move(120, 200)

        self.lb_target2 = QLabel(self)
        self.lb_target2.setFixedSize(100, 100)
        self.lb_target2.move(150, 250)

        # create a text editor to display processing result
        self.textedit = QTextEdit(self)
        self.textedit.setGeometry(400, 150, 500, 200)


    def bt_show_image(self):
        self.textedit.append("Make prediction ...")

        img = load_img(self.image_path).resize((32, 32))
        x = img_to_array(img)
        x = normalize(x.reshape((1,) + x.shape))
        result = np.argmax(self.cnn.single_predict(x))

        self.textedit.append("The food is: {}".format(self.food_list[result]))
        # self.show_index = np.random.randint(3, 1000)
        # for i in range(3):
        self.lb_target2.setPixmap(QPixmap("./{}/2.jpg".format(result)).scaled(100, 100))
        # self.textedit.append("Display 3 similar sample images")

    def bt_open_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Image", "./", 'Image files(*.jpg *.gif *.png)')
        self.lb_target.setPixmap(QPixmap(fname).scaled(self.lb_target.width(), self.lb_target.height()))
        self.textedit.append("Open an image from {}".format(fname))
        self.image_path = fname


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())



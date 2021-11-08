from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout
import sys
import os


class DicePointCounter(QWidget):
    def __init__(self, parent=None):
        super(DicePointCounter, self).__init__(parent)
        self.gui_container = QVBoxLayout()
        self.file_controls = QHBoxLayout()

        self.file_label = QtWidgets.QLabel()
        self.file_label.setText('Fájl: ')
        self.file_controls.addWidget(self.file_label)

        self.file_path = QtWidgets.QTextEdit()
        self.file_path.setDisabled(True)
        self.file_path.setFixedWidth(240)
        self.file_path.setMaximumHeight(32)
        self.file_controls.addWidget(self.file_path)

        self.open_file_button = QtWidgets.QPushButton()
        self.open_file_button.setText('Megnyitás')
        self.open_file_button.clicked.connect(self.open_file_clicked)
        self.file_controls.addWidget(self.open_file_button)

        self.gui_container.addLayout(self.file_controls)

        self.pic_priview = QtWidgets.QLabel()
        self.pic_priview.setMinimumHeight(150)
        self.gui_container.addWidget(self.pic_priview)

        self.count_button = QtWidgets.QPushButton()
        self.count_button.setText('Érték Számolása')
        self.count_button.clicked.connect()
        self.gui_container.addWidget(self.count_button)

        self.setLayout(self.gui_container)
        self.setWindowTitle('Dice Point Count')

    def open_file_clicked(self):
        filename = QFileDialog.getOpenFileName(self, 'Kép megnyitása', os.getcwd(), 'Image files (*.jpg *.gif)')
        pic = QtGui.QPixmap(filename[0])
        pic = pic.scaledToWidth(200)
        self.pic_priview.setPixmap(pic)

    def do_count(self):
        # TODO: funkció implementálása
        pass


def main():
    app = QApplication(sys.argv)
    ex = DicePointCounter()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

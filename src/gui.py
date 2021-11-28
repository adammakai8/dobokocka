""" A felhasználói felület és a program fő belépési pontja """

from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QMessageBox
import sys
import detection_segmentation as detection
from constants import IMG_DIR
from constants import IMG_FORMATS, PIC_PREVIEW_STYLE
from constants import FILE_LABEL_TEXT, OPEN_FILE_BTN_TEXT, USE_CNN_CB_TEXT, COUNT_BTN_TEXT, WAIT_TEXT
from constants import WIND_TITLE, OPEN_IMG_WIN_TITLE, POINT_MSG_TITLE, WAIT_TITLE


class DicePointCounter(QWidget):
    def __init__(self, parent=None):
        super(DicePointCounter, self).__init__(parent)
        self.gui_container = QVBoxLayout()
        self.file_controls = QHBoxLayout()

        self.file_label = QtWidgets.QLabel()
        self.file_label.setText(FILE_LABEL_TEXT)
        self.file_controls.addWidget(self.file_label)

        self.file_path = QtWidgets.QTextEdit()
        self.file_path.setDisabled(True)
        self.file_path.setFixedWidth(320)
        self.file_path.setMaximumHeight(32)
        self.file_controls.addWidget(self.file_path)

        self.open_file_button = QtWidgets.QPushButton()
        self.open_file_button.setText(OPEN_FILE_BTN_TEXT)
        self.open_file_button.clicked.connect(self.open_file_clicked)
        self.file_controls.addWidget(self.open_file_button)

        self.gui_container.addLayout(self.file_controls)

        self.pic_priview = QtWidgets.QLabel()
        self.pic_priview.setMinimumHeight(300)
        self.pic_priview.setStyleSheet(PIC_PREVIEW_STYLE)
        self.gui_container.addWidget(self.pic_priview)

        self.use_cnn_cb = QtWidgets.QCheckBox(USE_CNN_CB_TEXT)
        self.gui_container.addWidget(self.use_cnn_cb)

        self.count_button = QtWidgets.QPushButton()
        self.count_button.setText(COUNT_BTN_TEXT)
        self.count_button.clicked.connect(self.do_count)
        self.gui_container.addWidget(self.count_button)

        self.setLayout(self.gui_container)
        self.setWindowTitle(WIND_TITLE)

        self.filename = ''

    def open_file_clicked(self):
        self.filename = QFileDialog.getOpenFileName(self, OPEN_IMG_WIN_TITLE, IMG_DIR, IMG_FORMATS)
        self.file_path.setText(self.filename[0])
        pic = QtGui.QPixmap(self.filename[0])
        pic = pic.scaledToWidth(450)
        self.pic_priview.setPixmap(pic)

    def do_count(self):
        if self.filename == '':
            return
        else:
            msg = QMessageBox(self)
            if self.use_cnn_cb.isChecked():
                wait_msg = QMessageBox(self)
                wait_msg.setText(WAIT_TEXT)
                wait_msg.setWindowTitle(WAIT_TITLE)
                wait_msg.show()
                try:
                    pts = detection.segment_with_cnn(self.filename[0])
                except Exception as e:
                    warning_msg = QMessageBox(self)
                    warning_msg.setText(e)
                    warning_msg.setWindowTitle(e)
                    warning_msg.show()
                    pts = 0
                wait_msg.close()
            else:
                pts = detection.segment_with_traditional_techniques(self.filename[0])
            msg.setText(f'Points Counted: {pts}')
            msg.setWindowTitle(POINT_MSG_TITLE)
            msg.show()


def main():
    app = QApplication(sys.argv)
    ex = DicePointCounter()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

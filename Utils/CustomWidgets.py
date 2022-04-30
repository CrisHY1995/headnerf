
from PyQt5.QtWidgets import (QPushButton, QWidget, QLabel, QApplication, QHBoxLayout, QVBoxLayout, 
                               QListWidget, QComboBox, QMainWindow, QSlider, QGroupBox, QFileDialog, QSplitter)
from PyQt5.QtCore import Qt, QRect, QPoint
from PyQt5.QtGui import QFont, QPixmap, QImage, QPainter, QPen, QTabletEvent, QColor, QIcon


class CustomQSlider(QWidget):
    
    def __init__(self, name, width, height, min_val, max_val, init_val = 0) -> None:
        super().__init__()
        
        self.name = name
        self.width = width
        self.height = height
        self.min_val = min_val
        self.max_val = max_val
        self.cur_val = init_val
        
        self.build_info()
        self.init_layout()
        
    
    def build_info(self):
        self.slider_w = int(self.width * 0.55)
        self.label_name_w = int(self.width * 0.16)
        self.label_val_w = int(self.width * 0.16)        
        
    
    def update_geometry(self, width, height):
        self.slider_w = int(width * 0.55)
        self.label_name_w = int(width * 0.16)
        self.label_val_w = int(width * 0.16)    

        # self.slider.setFixedWidth(self.slider_w)
        # self.name_label.setFixedWidth(self.label_name_w)
        # self.val_label.setFixedWidth(self.label_val_w)
        self.slider.setFixedHeight(height)
        self.name_label.setFixedHeight(height)
        self.val_label.setFixedHeight(height)
        self.widget.setFixedHeight(height)
    
    
    def slider_set_value(self, cur_val):
        slider_val = int(self.slider_w * (cur_val - self.min_val) / (self.max_val - self.min_val))
        self.slider.setValue(slider_val)
        self.cur_val = cur_val
        self.val_label.setText("%.02f" % self.cur_val)


    def init_layout(self):
        
        self.slider = QSlider()
        self.slider.setObjectName(self.name)
        
        self.slider.setFixedWidth(self.slider_w)
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setRange(0, self.slider_w)
        
        # slider_val = int(self.slider_w * (self.cur_val - self.min_val) / (self.max_val - self.min_val))
        # self.slider.setValue(slider_val)
        
        self.name_label = QLabel()
        self.name_label.setFixedWidth(self.label_name_w)
        self.name_label.setText(self.name)
        self.name_label.setAlignment(Qt.AlignLeft)
        self.name_label.setFont(QFont('Courier New', 13))
        self.name_label.setAlignment(Qt.AlignVCenter)
        
        self.val_label = QLabel()
        self.val_label.setFixedWidth(self.label_val_w)
        # self.val_label.setText("%.02f" % self.cur_val)
        self.val_label.setFont(QFont('Courier New', 13))
        self.val_label.setAlignment(Qt.AlignVCenter)
        
        self.slider_set_value(self.cur_val)
        
        self.widget = QWidget()
        self.widget.setFixedHeight(self.height)
        
        self.h_layout = QHBoxLayout(self.widget)
        
        self.h_layout.addStretch(1)
        self.h_layout.addWidget(self.name_label)
        self.h_layout.addWidget(self.val_label)
        self.h_layout.addWidget(self.slider)
        self.h_layout.addStretch(1)
        self.h_layout.setAlignment(Qt.AlignVCenter)


    def update_labels(self):
        self.cur_val = (self.slider.value() / self.slider_w) * (self.max_val - self.min_val) + self.min_val
        self.val_label.setText("%.02f" % self.cur_val)
        self.update()
        
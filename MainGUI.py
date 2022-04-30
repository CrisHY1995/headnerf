import sys
from glob import glob
from tkinter.font import BOLD
from turtle import width
from webbrowser import get
import cv2
import numpy as np
from PIL import Image

import os
from os.path import join

from PyQt5.QtWidgets import (QPushButton, QWidget, QLabel, QApplication, QHBoxLayout, QVBoxLayout, QGridLayout, 
                               QListWidget, QComboBox, QMainWindow, QSlider, QGroupBox, QFileDialog, QSplitter)
from PyQt5.QtCore import Qt, QRect, QPoint
from PyQt5.QtGui import QFont, QPixmap, QImage, QPainter, QPen, QTabletEvent, QColor, QIcon, QBrush, QFontMetrics,QPalette

from Utils.CustomWidgets import CustomQSlider
import functools

import time
from Utils.AxisUtils import AxisUtils
from Utils.ArcBall import ArcBallUtil
from Utils.HeadNeRFUtils import HeadNeRFUtils
import argparse


class MainWindows(QMainWindow):
    def __init__(self, screen_w, screen_h, model_path):
        super().__init__()
        
        self.model_path = model_path
        self.base_name = os.path.basename(model_path)[:-4]
        self.code_1_path = "LatentCodeSamples/%s/S001_E01_I01_P02.pth" % self.base_name
        self.code_2_path = "LatentCodeSamples/%s/S016_E02_I01_P02.pth" % self.base_name
        assert os.path.exists(self.code_1_path)
        assert os.path.exists(self.code_2_path)
        with open('ConfigFiles/stylesheet.qss', 'r') as f:
            style = f.read()
            self.setStyleSheet(style)
        # self.setStyleSheet(
        #     "QGroupBox{padding-top:20px; margin-top:-20px}"
        # )
        self.init_varibles()
        self.init_window(screen_w, screen_h)
        self.init_widgets()
        self.build_connnect()


    def init_varibles(self):
           
        self.operational_zone_w = 400
        self.render_img_size = 512
        self.axis_img_size = self.render_img_size // 4
        self.st_img_w = 256
        self.border_w = 10
        self.margin_w = 40
        self.margin_h = 40
        self.putton_h = 0

        self.slider_name_list = ["Total: ", 
                            "Iden~: ", 
                            "Expr~: ", 
                            "Albe~: ", 
                            "Illu~: ", 
                            "Shape: ", 
                            "Appea: ", 
                            "Id+Al: ", 
                            "Pitch: ",
                            "Yaw  : ", 
                            "Roll : ", 
                            ]
        
        self.name_2_slider = {
            "Total: ": ["Iden~: ", "Expr~: ", "Albe~: ", "Illu~: "], 
            "Iden~: ": [], 
            "Expr~: ": [], 
            "Albe~: ": [], 
            "Illu~: ": [], 
            "Shape: ": ["Iden~: ", "Expr~: "], 
            "Appea: ": ["Albe~: ", "Illu~: "], 
            "Id+Al: ": ["Iden~: ", "Albe~: "], 
            "Pitch: ": [],
            "Yaw  : ": [], 
            "Roll : ": []
        }
        
        self.fps_info = "--"
        self.calc_fps = False
        
        self.render_img = None
        self.source_img = None
        self.target_img = None
        self.axis_img = None
        
        self.updata_img_available = True
        
        self.headnerf_utils = HeadNeRFUtils(self.model_path)
        
        self.headnerf_utils.update_code_1(self.code_1_path)
        self.source_img = cv2.resize(self.headnerf_utils.source_img, (int(self.st_img_w), int(self.st_img_w)))
        
        self.headnerf_utils.update_code_2(self.code_2_path)
        self.target_img = cv2.resize(self.headnerf_utils.target_img, (int(self.st_img_w), int(self.st_img_w)))
        
        self.axis_utils = AxisUtils(self.axis_img_size)
        self.arcball_utils = ArcBallUtil(int(self.render_img_size), int(self.render_img_size))
        

    def init_window(self, screen_w, screen_h):                
        
        total_width = self.margin_w + self.operational_zone_w + self.border_w + \
                            self.render_img_size + self.border_w + self.st_img_w + self.margin_w
        total_height = self.margin_h + self.render_img_size + self.putton_h + self.margin_h
        
        total_x_start = (screen_w - total_width) // 2
        total_y_start = (screen_h - total_height) // 2
        
        self.setWindowIcon(QIcon('ConfigFiles/logo.png'))
        self.setWindowTitle("HeadNeRF—Demo")
        self.setGeometry(total_x_start, total_y_start, total_width, total_height)
        self.setMinimumWidth(total_width)
        self.setMinimumHeight(total_height)
        
    
    def init_widgets(self):
        
        x_start = self.margin_w
        y_start = self.margin_h
        width = self.operational_zone_w
        height = self.render_img_size
        
        self.menu_v_layout = QVBoxLayout()
        self.menu_v_layout.setGeometry(QRect(x_start, y_start, width, height))
        
        # Silider
        self.editing_group_box = QGroupBox(self)
        self.editing_group_box.setGeometry(x_start, y_start, width, int(height * 0.82))
        v_editing_layout = QVBoxLayout(self.editing_group_box)
        
        v_editing_layout.addStretch(1)
        for cnt, name in enumerate(self.slider_name_list):
            if name in ["Yaw  : ", "Pitch: ", "Roll : "]:
                cur_slider = CustomQSlider(name, int(0.93 * width), int(0.75 * height / len(self.slider_name_list)), 
                                           min_val=-1.0, max_val=1.0, init_val=0.0)
            else:
                cur_slider = CustomQSlider(name, int(0.93 * width), int(0.75 * height / len(self.slider_name_list)), 
                                           min_val=0.0, max_val=1.0, init_val=0.0)
            
            setattr(self, name + "slider", cur_slider)
            v_editing_layout.addWidget(cur_slider.widget)
            v_editing_layout.addStretch(1)


        # PushButton
        self.button_group_box = QGroupBox(self)
        self.button_group_box.setGeometry(x_start, y_start + int(height * 0.823), width, int(height * 0.18))
        button_layout = QGridLayout(self.button_group_box)
        
        self.reset_view_button = QPushButton("Reset View  ")
        self.reset_view_button.setFont(QFont('Courier New', 14))
        button_layout.addWidget(self.reset_view_button, 0, 0)
        self.reset_weight_button = QPushButton("Reset Weight")
        self.reset_weight_button.setFont(QFont('Courier New', 14))
        button_layout.addWidget(self.reset_weight_button, 1, 0)
        
        self.change_source_button = QPushButton("Change Source")
        self.change_source_button.setFont(QFont('Courier New', 14))
        button_layout.addWidget(self.change_source_button, 0, 1)
        self.change_target_button = QPushButton("Change Target")
        self.change_target_button.setFont(QFont('Courier New', 14))
        button_layout.addWidget(self.change_target_button, 1, 1)
        
        self.menu_v_layout.addWidget(self.editing_group_box)
        self.menu_v_layout.addWidget(self.button_group_box)


    def resizeEvent(self, event):
        
        scale_w = event.size().width() / event.oldSize().width()
        scale_h = event.size().height() / event.oldSize().height()
        
        if scale_w > 0: 
            self.render_img_size = (self.render_img_size * scale_w)
            # self.margin_w = (self.margin_w * scale_w)

            self.margin_h = (self.margin_h * scale_w)
            self.border_w = (self.border_w * scale_w)
            self.st_img_w = (self.st_img_w * scale_w)
            self.margin_w = (event.size().width() - self.operational_zone_w - self.border_w - \
                            self.render_img_size - self.border_w - self.st_img_w) * 0.5
            # self.operational_zone_w = (self.operational_zone_w * scale_w)
            
        
        if scale_h > 0: 
            self.margin_h = (event.size().height() - self.render_img_size - self.putton_h) * 0.5
            self.editing_group_box.setGeometry(int(self.margin_w), int(self.margin_h), 
                                               int(self.operational_zone_w), int(self.render_img_size * 0.82))
            self.button_group_box.setGeometry(int(self.margin_w), int(self.margin_h) + int(self.render_img_size * 0.823), 
                                              int(self.operational_zone_w), int(self.render_img_size * 0.18))

        self.arcball_utils.setBounds(int(self.render_img_size), int(self.render_img_size))
        self.axis_utils.set_img_size(int(self.render_img_size / 4))
        self.update_imgs()
        self.update()
        

    def build_connnect(self):
        for name in self.slider_name_list:
            cur_slider = getattr(self, name + "slider")
            cur_slider.slider.valueChanged.connect(self.slider_slot_func)   
    
        self.change_source_button.clicked.connect(self.update_source_code)
        self.change_target_button.clicked.connect(self.update_target_code)
        self.reset_view_button.clicked.connect(self.reset_view)
        self.reset_weight_button.clicked.connect(self.reset_weight)
        
        
    def update_source_code(self):

        dialog = QFileDialog()
        # dialog.setStyleSheet("QWidget{background-color:rgb(0, 0, 0);color: rgb(0, 0, 0);\
        #                         border-color: rgb(58, 58, 58)}")
        # with open('ConfigFiles/qdialog.qss', 'r') as f:
        #     style = f.read()
        #     dialog.setStyleSheet(style)
            
        file_path_1, filetype = dialog.getOpenFileName(self, "Select Sample", "./LatentCodeSamples/%s" % self.base_name, 
                                                                            "All Files (*);;Text Files (*.pth)") 
        # file_path_1, filetype = QFileDialog.getOpenFileName(self, "Select Sample", "./LatentCodeSamples/%s" % self.base_name, 
        #                                                                     "All Files (*);;Text Files (*.pth)")  
        if os.path.exists(file_path_1):
            self.headnerf_utils.update_code_1(file_path_1)
            self.source_img = cv2.resize(self.headnerf_utils.source_img, (int(self.st_img_w), int(self.st_img_w)))
        self.update_imgs()
        self.update() 
    

    def update_target_code(self):
        file_path_1, filetype = QFileDialog.getOpenFileName(self, "Select Sample", "./LatentCodeSamples/%s" % self.base_name, 
                                                                            "All Files (*);;Text Files (*.pth)")  
        if os.path.exists(file_path_1):
            self.headnerf_utils.update_code_2(file_path_1)
            self.target_img = cv2.resize(self.headnerf_utils.target_img, (int(self.st_img_w), int(self.st_img_w)))
        self.update_imgs()
        self.update() 
    
    
    def reset_view(self):
        self.updata_img_available = False
        getattr(self, "Pitch: " + "slider").slider_set_value(0.0)
        getattr(self, "Yaw  : " + "slider").slider_set_value(0.0)
        getattr(self, "Roll : " + "slider").slider_set_value(0.0)
        self.updata_img_available = True
        self.update_imgs()
        self.arcball_utils.resetRotation()


    def reset_weight(self):
        self.updata_img_available = False
        for name in self.name_2_slider:
            if name not in ["Pitch: ", "Yaw  : ", "Roll : "]:
                getattr(self, name + "slider").slider_set_value(0.0)
        self.updata_img_available = True
        self.update_imgs()


    def slider_slot_func(self):
        
        base_name = self.sender().objectName()
        if self.updata_img_available:
            self.updata_img_available = False
            self.update_slider(base_name)
            self.updata_img_available = True
            self.update_imgs()
        else:
            self.update_slider(base_name)
            
        self.update()
        

    def update_slider(self, base_name):
        base_slider = getattr(self, base_name + "slider")
        base_slider.update_labels()
        base_val = base_slider.cur_val
        name_list = self.name_2_slider[base_name]
        for temp_name in name_list:
            cur_slider = getattr(self, temp_name + "slider")
            cur_slider.slider_set_value(base_val)
    
    
    def update_imgs(self):
        if not self.updata_img_available:
            return
        
        pitch = getattr(self, "Pitch: " + "slider").cur_val
        yaw = getattr(self, "Yaw  : " + "slider").cur_val
        roll = getattr(self, "Roll : " + "slider").cur_val
        
        # render img
        if self.calc_fps: 
            time_s = time.time()
            
        self.render_img = self.headnerf_utils.gen_image(getattr(self, "Iden~: " + "slider").cur_val, 
                                getattr(self, "Expr~: " + "slider").cur_val,
                                getattr(self, "Albe~: " + "slider").cur_val,
                                getattr(self, "Illu~: " + "slider").cur_val,
                                pitch, 
                                yaw,
                                roll
                                )
        if self.calc_fps:
            time_e = time.time()
            fps = int(1.0 / (time_e - time_s))

            self.fps_info = fps

        self.calc_fps = True
        
        self.render_img = cv2.resize(self.render_img, (int(self.render_img_size), int(self.render_img_size)))
        self.render_img = self.axis_utils.generate_img(pitch, yaw, roll, self.render_img)


    def _map2imageRegion(self, pts_x, pts_y):
        x = pts_x - int(self.margin_w + self.operational_zone_w + self.border_w)
        y = pts_y - int(self.margin_h)
        return x, y


    def mousePressEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            img_x, img_y = self._map2imageRegion(event.x() , event.y())
            
            if img_x > 0 and img_x < self.render_img_size and img_y > 0 and img_y < self.render_img_size:
                self.arcball_utils.onClickLeftDown(float(img_x), float(img_y))
            self.update()


    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            img_x, img_y = self._map2imageRegion(event.x() , event.y())
            self.arcball_utils.onDrag(img_x, img_y)
            cur_eulur = self.arcball_utils.eulur_angle
            
            self.updata_img_available = False
            getattr(self, "Pitch: " + "slider").slider_set_value(cur_eulur[0])
            getattr(self, "Yaw  : " + "slider").slider_set_value(cur_eulur[1])
            getattr(self, "Roll : " + "slider").slider_set_value(cur_eulur[2])
            self.updata_img_available = True
            self.update_imgs()


    def mouseReleaseEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.arcball_utils.onClickLeftUp()


    def paintEvent(self, e):

        img_x = int(self.margin_w + self.operational_zone_w + self.border_w)
        img_y = int(self.margin_h)
        render_img_size = int(self.render_img_size)
        border_w = int(self.border_w)
        st_img_w = int(self.st_img_w)
        
        # axis_img_rect = QRect(img_x, img_y, int(self.axis_img_size), int(self.axis_img_size))
        
        img_rect = QRect(img_x, img_y, render_img_size, render_img_size)
        
        source_img_rect = QRect(img_x + render_img_size + border_w, img_y, st_img_w, st_img_w)
        target_img_rect = QRect(img_x + render_img_size + border_w, img_y + st_img_w, st_img_w, st_img_w)

        qp = QPainter()        
        qp.begin(self)

        # if self.axis_img is not None:
        #     # axis_img = cv2.resize(self.axis_img, (render_img_size, render_img_size))
        #     qp.drawImage(axis_img_rect, Image.fromarray(self.axis_img).toqimage())

        if self.render_img is not None:
            qp.drawImage(img_rect, Image.fromarray(self.render_img).toqimage())
        
        if self.source_img is not None:
            qp.drawImage(source_img_rect, Image.fromarray(self.source_img).toqimage())

        if self.target_img is not None:
            qp.drawImage(target_img_rect, Image.fromarray(self.target_img).toqimage())

        qp.setPen(QColor(220, 220, 220, 255))
        qp.drawRect(img_rect)
        qp.drawRect(source_img_rect)
        qp.drawRect(target_img_rect)


        # f.setPixelSize(30)
        qp.setFont(QFont('Courier New', 14))
        qp.setPen(Qt.black)

        qp.font().setBold(True)
        fm = QFontMetrics(qp.font())
        fw = fm.width("Source")
        fh = fm.height()
        
        qp.translate(QPoint(img_x + render_img_size + border_w + st_img_w - (fh // 4), img_y + ((st_img_w - fw)//2)))
        qp.rotate(90)
        qp.fillRect(QRect(0, -fh, fw, fh), QBrush(QColor(255, 255, 255, 255)))
        qp.drawText(QPoint(0, 0), "Source")
        
        qp.fillRect(QRect(st_img_w, -fh, fw, fh), QBrush(QColor(255, 255, 255, 255)))
        qp.drawText(QPoint(st_img_w, 0), "Target")
        qp.rotate(-90)
        qp.translate(-QPoint(img_x + render_img_size + border_w + st_img_w - (fh // 4), img_y + ((st_img_w - fw)//2)))
        
        
        #FPS info
        f = self.font()
        f.setFamily('Courier New')
        f.setPointSize(20)
        # f.setBold(True)
        qp.setFont(f)
        qp.setPen(Qt.black)

        fm = QFontMetrics(f)
        fw = fm.width("FPS %s" % self.fps_info)
        fh = fm.height()
        
        qp.drawText(QPoint(img_x + render_img_size - fw - 20, img_y + render_img_size - 10), "FPS %s" % self.fps_info)
        qp.end()



if __name__ == "__main__":
    # opt = parser.parse_args()

    # model_path = "Models/model_all_Reso32HR.pth"
    # para_file_path = "Models/mode_all_32x32_HR.json"
    # base_name = "mode_all__version_Reso32x32_HR_v1_full"

    parser = argparse.ArgumentParser(description='HeadNeRF-GUI')
    parser.add_argument("--model_path", type=str, required=True)

    args = parser.parse_args()
    # model_path = "TrainedModels/model_Reso32.pth"
    model_path = args.model_path

    # model_path = "Models/model_all_Reso64.pth"
    # para_file_path = "Models/mode_all_64x64.json"
    # base_name = "mode_all__version_new_v1_full"
    
    app = QApplication(sys.argv)
    # app.setStyle('Windows')
    screen = app.primaryScreen()
    size = screen.size()
    rect = screen.availableGeometry()
    ex = MainWindows(rect.width(), rect.height(), model_path)
    ex.show()
    sys.exit(app.exec_())
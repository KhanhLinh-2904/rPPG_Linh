import cv2
import numpy as np
import pyqtgraph as pg
import webbrowser
from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from run_all_models import RunAlModels
from webcam import Camera_RGB
import sys
import signal
from PyQt5.QtCore import pyqtSignal
from multiprocessing import Queue, Process, freeze_support

class Communicate(QObject):
    closeApp = pyqtSignal()


def model_worker(input_queue, output_queue):
    """Worker process to run models."""
    run_all_models = RunAlModels()
    while True:
        task = input_queue.get()
        if task == "STOP":
            break
        # Perform model processing
        frame = task["frame"]
        result = run_all_models.run(frame)
        output_queue.put(result)


class GUI(QMainWindow, QThread, QApplication):
    def __init__(self):
        super().__init__()
        self.initUI()  # start the UI when run
        
        self.timer = QTimer(self)  # Initialize QTimer
        self.timer.timeout.connect(self.update_time)  # Connect timer timeout to update method
        self.start_time = None  # Record when the timer starts
        self.elapsed_time = 0  # Store elapsed time in seconds
        
        self.input_rgb_camera = Camera_RGB()
        self.input = self.input_rgb_camera  # input of the app is cameras
      
        # Queues for multiprocessing communication
        self.input_queue = Queue()
        self.output_queue = Queue()
        
        # Worker process for running models
        self.worker_process = Process(target=model_worker, args=(self.input_queue, self.output_queue))
        self.worker_process.start()
        # self.runAllModels = RunAlModels()
        self.status = False  # If false, not running, if true, running
        self.length = 10 #Estimator = 10 second
        self.running = False
        self.avg_bpms = []
        self.smooth_bpms = []
        self.bpm_count = 0
        self.countFrame = 0
        
        # Add a list to track RR intervals (in seconds)
        self.rr_intervals = []

    def initUI(self):
        # set font
        font = QFont()
        font.setFamily('Inter')             # Set font family to Inter
        font.setPointSize(3)                # Set font size to 24
        font.setWeight(QFont.ExtraBold)      # Set font weight to extra bold

        # display face online (the biggest window)
        self.lblDisplay = QLabel(self)
        self.lblDisplay.setGeometry(113, 90, 568, 396)
        self.lblDisplay.setStyleSheet("background-color: #272626")
        self.lblDisplay.setAlignment(QtCore.Qt.AlignCenter)

        # display mean frame (the biggest window)
        self.meanDisplay = QLabel(self)
        self.meanDisplay.setGeometry(778, 46, 245, 203)
        self.meanDisplay.setStyleSheet("background-color: #272626")
        self.meanDisplay.setAlignment(QtCore.Qt.AlignCenter)

        # display different frame (the biggest window)
        self.diffDisplay = QLabel(self)
        self.diffDisplay.setGeometry(1068, 46, 245, 203)
        self.diffDisplay.setStyleSheet("background-color: #272626")
        self.diffDisplay.setAlignment(QtCore.Qt.AlignCenter)

        # display ROI frame (the biggest window)
        self.roiDisplay = QLabel(self)
        self.roiDisplay.setGeometry(778, 303, 245, 203)
        self.roiDisplay.setStyleSheet("background-color: #272626")
        self.roiDisplay.setAlignment(QtCore.Qt.AlignCenter)
        
        # dynamic plot # Processed Signal 圖表
        self.signal_Plt = pg.PlotWidget(self)
        self.signal_Plt.setGeometry(113, 591, 1206, 361)
        self.signal_Plt.setBackground('#ffffff')

        # Create a bold font with padding
        font = QFont()
        font.setBold(True)
        font.setPointSize(9)  

        # Set the top, bottom, and left labels with font and padding
        self.signal_Plt.setLabel('top', "Photoplethysmographic Estimate HR", font=font)
        self.signal_Plt.setLabel('bottom', "", font=font)
        self.signal_Plt.setLabel('left', "rPPG Signal", font=font)

        self.signal_Plt.getAxis('top').setHeight(60)  # Add padding to the top axis
        self.signal_Plt.getAxis('bottom').setHeight(60)  # Add padding to the bottom axis
        self.signal_Plt.getAxis('left').setWidth(60)  # Add padding to the left axis

        # display face online (the biggest window)
        self.lblInformation = QLabel(self)
        self.lblInformation.setGeometry(1068, 329, 309, 183)
        self.lblInformation.setStyleSheet("background-color: #272626")
        self.lblInformation.setAlignment(QtCore.Qt.AlignCenter)

        # Time Label
        #now = QDateTime.currentDateTime()
        self.Duration = QLabel(self)
        self.Duration.setGeometry(1083, 344, 86, 24)
        self.Duration.setFont(font)
        self.Duration.setAlignment(Qt.AlignLeft)
        self.Duration.setStyleSheet("color:#999797")
        self.Duration.setText("Duration")
        
        self.lblTime = QLabel(self)
        self.lblTime.setGeometry(1239, 344, 114, 24)
        self.lblTime.setFont(font)
        self.lblTime.setAlignment(Qt.AlignLeft)
        self.lblTime.setStyleSheet("color:#999797")
        self.lblTime.setText("00 : 00 : 00")
        
      
        # self.Output = QLabel(self)
        # self.Output.setGeometry(1083, 369, 98, 29)
        # self.Output.setFont(font)
        # self.Output.setAlignment(Qt.AlignLeft)
        # self.Output.setStyleSheet("color:#999797")
        # self.Output.setText("Output")
        
        # self.lblOutput = QLabel(self)
        # self.lblOutput.setGeometry(1239, 369, 114, 29)
        # self.lblOutput.setFont(font)
        # self.lblOutput.setAlignment(Qt.AlignLeft)
        # self.lblOutput.setStyleSheet("color:#999797")
        # self.lblOutput.setText("0")

        # HR Label 
        # Display the heart rate  
        self.HR = QLabel(self)
        self.HR.setGeometry(1083, 394, 200, 29)
        self.HR.setFont(font)
        self.HR.setAlignment(Qt.AlignLeft)
        self.HR.setStyleSheet("color:#999797")
        self.HR.setText("HR (bpm)")
        
        self.lblHR = QLabel(self)
        self.lblHR.setGeometry(1239, 394, 200, 29)
        self.lblHR.setFont(font)
        self.lblHR.setAlignment(Qt.AlignLeft)
        self.lblHR.setStyleSheet("color:#999797")
        self.lblHR.setText("0")
       
        # Frequency Label 
        # Display the Frequency 
        self.Frequency = QLabel(self)
        self.Frequency.setGeometry(1083, 419, 200, 29)
        self.Frequency.setFont(font)
        self.Frequency.setAlignment(Qt.AlignLeft)
        self.Frequency.setStyleSheet("color:#999797")
        self.Frequency.setText("Frequency (Hz)")
        
        self.lblFrequency = QLabel(self)
        self.lblFrequency.setGeometry(1239, 419, 200, 29)
        self.lblFrequency.setFont(font)
        self.lblFrequency.setAlignment(Qt.AlignLeft)
        self.lblFrequency.setStyleSheet("color:#999797")
        self.lblFrequency.setText("0")
        
        # Heart Arrythmia 
        # Display the Frequency 
        self.HeartArrythmia = QLabel(self)
        self.HeartArrythmia.setGeometry(1083, 444, 250, 29)
        self.HeartArrythmia.setFont(font)
        self.HeartArrythmia.setAlignment(Qt.AlignLeft)
        self.HeartArrythmia.setStyleSheet("color:#999797")
        self.HeartArrythmia.setText("Heart Arrythmia")
        
        self.lblHeartArrythmia = QLabel(self)
        self.lblHeartArrythmia.setGeometry(1239, 444, 250, 29)
        self.lblHeartArrythmia.setFont(font)
        self.lblHeartArrythmia.setAlignment(Qt.AlignLeft)
        self.lblHeartArrythmia.setStyleSheet("color:#821515")
        self.lblHeartArrythmia.setText("Not Detected")


        # Estimated HR Label 
        # Display the estimated heart rate 
        self.EstimatedHR = QLabel(self)
        self.EstimatedHR.setGeometry(1083, 469, 216, 90)
        self.EstimatedHR.setFont(font)
        self.EstimatedHR.setAlignment(Qt.AlignLeft)
        self.EstimatedHR.setStyleSheet("color:#FFFFFF")
        self.EstimatedHR.setText("Average HR")
        
        self.lblEstimatedHR = QLabel(self)
        self.lblEstimatedHR.setGeometry(1239, 469, 114, 29)
        self.lblEstimatedHR.setFont(font)
        self.lblEstimatedHR.setAlignment(Qt.AlignLeft)
        self.lblEstimatedHR.setStyleSheet("color:#F94868")
        self.lblEstimatedHR.setText("0")

        # self.lblEstimatedHRtext = QLabel(self)
        # self.lblEstimatedHRtext.setGeometry(1069, 539, 60, 29)
        # self.lblEstimatedHRtext.setFont(font)
        # self.lblEstimatedHRtext.setAlignment(Qt.AlignLeft)
        # self.lblEstimatedHRtext.setStyleSheet("color:#F94868")
        # self.lblEstimatedHRtext.setText("BPM")

        # self.lblCCU_Logo1 = QLabel(self)
        # self.lblCCU_Logo1.setGeometry(1140, 540, 20, 20)
        # self.lblCCU_Logo1.setStyleSheet(
        #     "QLabel{border-image: url(./IMG_Source/Heart_icon.png);}")


        # Infor GUI show
        
        self.textMF = QLabel(self)
        self.textMF.setGeometry(830, 262, 146, 26)
        self.textMF.setFont(font)
        self.textMF.setAlignment(Qt.AlignLeft)
        self.textMF.setStyleSheet("color:#FFFFFF")
        self.textMF.setText("Mean frame")
        
        self.textDF = QLabel(self)
        self.textDF.setGeometry(1091, 262, 194, 26)
        self.textDF.setFont(font)
        self.textDF.setAlignment(Qt.AlignLeft)
        self.textDF.setStyleSheet("color:#FFFFFF")
        self.textDF.setText("ROI 36x36")
        
        self.textROI = QLabel(self)
        self.textROI.setGeometry(877, 521, 146, 26)
        self.textROI.setFont(font)
        self.textROI.setAlignment(Qt.AlignLeft)
        self.textROI.setStyleSheet("color:#FFFFFF")
        self.textROI.setText("ROI")
        # CCU Logo1 button
        # National Chung Cheng University
        self.lblCCU_Logo1 = QLabel(self)
        self.lblCCU_Logo1.setGeometry(70, 10, 270, 50)
        self.lblCCU_Logo1.setStyleSheet(
            "QLabel{border-image: url(./IMG_Source/CCU_Logo.png); color: #3683BC;}")


        # CCU Logo3 button
        # Display text: 電機工程 研究所
        self.lblCCU_Logo3 = QLabel(self)
        self.lblCCU_Logo3.setGeometry(350, 15, 270, 50)
        self.lblCCU_Logo3.setStyleSheet("color:#204C8F")
        self.lblCCU_Logo3.setFont(QFont("Adobe 宋体 Std L", 10, QFont.Bold))
        self.lblCCU_Logo3.setText("電機工程 研究所")

      
        buttonFont = QFont()
        buttonFont.setBold(True)
        buttonFont.setPointSize(10)  
        # START button
        self.btnStart = QPushButton("START", self)
        self.btnStart.setGeometry(184, 526, 80, 30)
        self.btnStart.setFont(buttonFont)
        self.btnStart.setStyleSheet("QPushButton{color: #230CF2 ; background-color: #A09A9A; border-radius: 10px; border: 2px groove gray;border-style: outset;}"
                                    "QPushButton:hover{color: #110388;}"
                                    "QPushButton:pressed{color: #110388;}")
        self.btnStart.clicked.connect(self.run)
        

        # STOP button
        self.btnStop = QPushButton("STOP", self)
        self.btnStop.setGeometry(344, 526, 80, 30)
        self.btnStop.setFont(buttonFont)
        self.btnStop.setStyleSheet("QPushButton{color: #FF0606 ;background-color: #A09A9A;  border-radius: 10px; border: 2px groove gray;border-style: outset;}"
                                   "QPushButton:hover{color: #873131;}"
                                   "QPushButton:pressed{color: #873131;}")
        # Connect STOP button to stop the timer
        self.btnStop.clicked.connect(self.stop)
        
        # RESET button
        self.btnReset = QPushButton("RESET", self)
        self.btnReset.setGeometry(504, 526, 80, 30)
        self.btnReset.setFont(buttonFont)
        self.btnReset.setStyleSheet("QPushButton{color: #E16D07 ;background-color: #A09A9A;  border-radius: 10px; border: 2px groove gray;border-style: outset;}"
                                   "QPushButton:hover{color: #964E0E;}"
                                   "QPushButton:pressed{color: #964E0E;}")
         # Connect RESET button to reset the timer
        self.btnReset.clicked.connect(self.reset)

        
        # Information button
        self.btnInformation = QPushButton(self)
        self.btnInformation.setGeometry(20, 10, 40, 40)
        self.btnInformation.setStyleSheet("QPushButton{border-image: url(./IMG_Source/Information_Button.png)}"
                                          "QPushButton:hover{background-color: #6F6B6B;}"
                                          "QPushButton:pressed{background-color: #6F6B6B;}")
        self.btnInformation.clicked.connect(self.btnInformation_clicked)

        # event close
        self.c = Communicate()
        self.c.closeApp.connect(self.closeEvent)

        # config main window # 視窗大小
        # Name of program
        self.setWindowTitle("Heart Rate Monitor")
        self.setGeometry(0, 0, 1440, 1024)

       
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor("#373B54"))
        self.setPalette(palette)

        self.statusBar = QStatusBar()
        self.statusBar.setStyleSheet("color:white")
        self.statusBar.setFont(QFont("OCR A Std", 13, QFont.Bold))
        self.setStatusBar(self.statusBar)

        # event close
        self.c = Communicate()
        self.c.closeApp.connect(self.closeEvent)

        self.center()
        self.show()

    def start_timer(self):
            """Start the timer."""
            if not self.timer.isActive():
                self.start_time = QTime.currentTime()
                self.timer.start(1000)  # Trigger every 1 second
                self.update_time()

    def stop_timer(self):
        """Stop the timer."""
        if self.timer.isActive():
            self.timer.stop()

    def stop(self):
        self.stop_timer()
        self.running = False
        
    def reset(self):
        """Reset the timer to 00:00:00."""
        self.stop_timer()
        self.elapsed_time = 0
        self.update_timer_display()
        self.input_queue.put("STOP")  # Signal worker to stop
        # self.runAllModels.reset()
        self.lblDisplay.clear()
        self.signal_Plt.clear()
        self.input.stop()
        self.running = False
        # Reset all labels to initial state
        self.lblHR.setText("0")
        QApplication.processEvents()
        
        # self.lblHRV.setText("0")
        # QApplication.processEvents()
        
        self.lblFrequency.setText("0")
        QApplication.processEvents()
        
        self.lblHeartArrythmia.setStyleSheet("color:#821515")
        QApplication.processEvents()
        
        self.lblHeartArrythmia.setText("Not Detected")
        QApplication.processEvents()
        
        self.lblEstimatedHR.setText("0")
        QApplication.processEvents()
        
        

    def update_time(self):
        """Update elapsed time and display."""
        self.elapsed_time += 1
        self.update_timer_display()

    def update_timer_display(self):
        """Update the lblTime label with the formatted time."""
        hours, remainder = divmod(self.elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_time = f"{hours:02}:{minutes:02}:{seconds:02}"
        self.lblTime.setText(formatted_time)
        QApplication.processEvents()
        
    def btnInformation_clicked(self):
        webbrowser.open('https://ee.ccu.edu.tw/p/412-1097-559.php?Lang=en')


    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def closeEvent(self, event):
        reply = QMessageBox.question(self, "Message", "Are you sure want to quit",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            event.accept()
            self.input.stop()
            self.running = False
            cv2.destroyAllWindows()
            app.quit()
            # sys.exit(app.exec_())
        else:
            event.ignore()

   
   

    def key_handler(self):
        """
        cv2 window must be focused for keypresses to be detected.
        """
        self.pressed = cv2.waitKey(1) & 255  # wait for keypress for 10 ms
        if self.pressed == 27:  # exit program on 'esc'
            print("[INFO] Exiting")
            self.input.stop()
  

    def update_bpm_and_fre(self, bpm):
            """This method updates the BPM and calculates HRV."""
            # Calculate RR interval from BPM (in seconds)
            if bpm == 0 :
                return
            rr_interval = 60 / bpm
            frequency = 1/rr_interval if rr_interval != 0 else 0
            # Append the new RR interval to the list
            self.rr_intervals.append(rr_interval)
            
            self.lblHR.setText(f"{bpm:.2f}")
            QApplication.processEvents()
          
            self.lblFrequency.setText(f"{frequency:.2f}") 
            QApplication.processEvents()
            
            
    @QtCore.pyqtSlot()
    def main_loop(self):
        color_frame = self.input.get_frame() #Capture 1 frame from Camera
        if color_frame is not None:
            gui_img = QImage(color_frame, color_frame.shape[1], color_frame.shape[0], color_frame.strides[0],
                            QImage.Format_RGB888)
            color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
            # print("shape of color frame: ", color_frame.shape)
            self.input_queue.put({"frame": color_frame})
            
            self.lblDisplay.setPixmap(QPixmap(gui_img))  # show frame on GUI
            QApplication.processEvents()
            while not self.output_queue.empty():
                (color_face, dif_frame, mean_frame, outputs, bpm, idx, RGB_signal_buffer, bpms) = self.output_queue.get()
                if color_face is not None:
                    color_face = cv2.cvtColor(color_face, cv2.COLOR_BGR2RGB)
                    # print("type color_face: ", type(color_face))
                    color_face = cv2.resize(color_face, (180, 180), interpolation=cv2.INTER_CUBIC)
                    # print("shape color_face:", color_face.shape)
                    
                    # reshape color_face (240, 240)
                    gui_face = QImage(color_face, color_face.shape[1], color_face.shape[0], color_face.strides[0],
                                    QImage.Format_RGB888)
                    self.roiDisplay.setPixmap(QPixmap(gui_face))
                    QApplication.processEvents()
                    
                # if isinstance(outputs, np.ndarray):
                #     for inum in outputs[0]:
                #         # self.countFrame += 1
                #         # print("Frame ", self.countFrame ,inum)
                #         self.lblOutput.setText(f"{inum:.2f}")
                #         QApplication.processEvents()
                        
                #         # self.Output.setText(f"Output {self.countFrame}")
                #         # QApplication.processEvents()
                        
                if color_face is not None:
                    
                    color_face = cv2.resize(color_face, (36, 36), interpolation=cv2.INTER_CUBIC)
                    # print("type of dif_frame: ", type(dif_frame))
                    # print("shape dif_frame: ",dif_frame.shape )
                    gui_dif_face = QImage(color_face, color_face.shape[1], color_face.shape[0], color_face.strides[0],
                                    QImage.Format_RGB888)
                    self.diffDisplay.setPixmap(QPixmap(gui_dif_face))
                    
                    QApplication.processEvents()
                # if dif_frame is not None:
                    
                #     # dif_frame = cv2.cvtColor(dif_frame, cv2.COLOR_BGR2RGB)
                #     dif_frame = cv2.resize(dif_frame, (180, 180), interpolation=cv2.INTER_CUBIC)
                #     # print("type of dif_frame: ", type(dif_frame))
                #     # print("shape dif_frame: ",dif_frame.shape )
                #     gui_dif_face = QImage(dif_frame, dif_frame.shape[1], dif_frame.shape[0], dif_frame.strides[0],
                #                     QImage.Format_RGB888)
                #     self.diffDisplay.setPixmap(QPixmap(gui_dif_face))
                    
                #     QApplication.processEvents()
                    
                if mean_frame is not None:
                    mean_frame = cv2.cvtColor(mean_frame, cv2.COLOR_BGR2RGB)
                    mean_frame = cv2.resize(mean_frame, (180, 180), interpolation=cv2.INTER_CUBIC)
                    gui_mean_face = QImage(mean_frame, mean_frame.shape[1], mean_frame.shape[0], mean_frame.strides[0],
                                    QImage.Format_RGB888)
                    self.meanDisplay.setPixmap(QPixmap(gui_mean_face))
                    
                self.update_bpm_and_fre(bpm)
                
                if len(bpms) > 15:
                    # print("===========: ",len(self.runAllModels.bpms))
                    # print("self.runAllModels.bpms: ",self.runAllModels.bpms)
                    
                    for i in range(3, 0, -1):
                        try:
                            if(len(bpms[-5 * i:-5 * (i - 1)])==0):
                                continue
                            self.smooth_bpms.append(np.mean(bpms[-5 * i:-5 * (i - 1)]))
                            # print("self.smooth_bpms: ",self.smooth_bpms)

                        except:
                            print("lblHR: eror in mean ")
                    self.avg_bpms = np.mean(self.smooth_bpms)
                    # self.update_bpm_and_hrv_and_fre(self.avg_bpms)
                    self.estimatedHR_and_arrhythmia(self.avg_bpms)

                # if self.runAllModels.bpms.__len__() > 1:
                #     self.estimatedHR_and_arrhythmia(self.runAllModels.bpms)
                
                    # self.avg_bpms.append(np.mean(self.runAllModels.bpms))

                self.key_handler()  # if not the GUI cant show anything, to make the gui refresh after the end of loop
                self.signal_Plt.clear()
                self.signal_Plt.plot(idx, RGB_signal_buffer, pen='r')  # Plot green signal
        
    def estimatedHR_and_arrhythmia(self, processed_bpm):
        estimated_HR = round(np.mean(processed_bpm))
        self.lblEstimatedHR.setText(str(estimated_HR))
        QApplication.processEvents()
        strHA = ''
        if estimated_HR == 0:
            strHA = 'Not Detected'
            self.lblHeartArrythmia.setStyleSheet("color:#821515")
            QApplication.processEvents()
            
        if estimated_HR > 100:
            strHA = 'Possible Tachycardia'
            self.lblHeartArrythmia.setStyleSheet("color:#821515")
            QApplication.processEvents()
            
        elif estimated_HR < 60:
            strHA = 'Possible Bradycardia'
            self.lblHeartArrythmia.setStyleSheet("color:#821515")
            QApplication.processEvents()
            
        else:
            strHA = 'Normal'
            self.lblHeartArrythmia.setStyleSheet("color:#10F33A")
            QApplication.processEvents()
            
        self.lblHeartArrythmia.setText(strHA)
        QApplication.processEvents()

   
    def run(self):
        self.running = True
        self.input.start()
        self.start_timer()
        while self.running:
            self.main_loop()
            


def signal_handler(sig, frame):
    print("ctrl c")
    app.quit()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    freeze_support()
    app = QApplication(sys.argv)
    ex = GUI()
    sys.exit(app.exec_())

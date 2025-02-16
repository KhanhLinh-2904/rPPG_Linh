import time
import numpy as np
import scipy
from face_detection import FaceDetection
from predict_bpm import Prediction_bpm
import scipy.signal
from scipy.signal import butter
from face_segment import FaceSegment
class RunAlModels(object):
    def __init__(self):
        self.sampling_rate = 30  # Frame rate of the video input
        self.order = 10  # order of butterworth filter
        self.length = 10
        self.buffer_size = self.sampling_rate * self.length
        self.RGB_signal_buffer = []
        self.fd = FaceDetection()
        self.fs = FaceSegment()
        self.count = 0  # The second condition to stop the app
        self.MTTS_CSTM = Prediction_bpm()
        self.bpms = []
        self.T = 10
        self.motion_frames = []
        self.app_frames = []
        self.list_infer_arr = []
        self.new_list_infer_arr = []
        self.indx = []
        self.bpm = 0
        self.outputs = 0
        

    def generate_motion_difference(self, prev_frame, cur_frame):
        prev_frame = prev_frame.astype(np.float32)
        cur_frame = cur_frame.astype(np.float32)
        dif_frame = (cur_frame - prev_frame) 
        return dif_frame

    def process_model(self, arr_motion, arr_appearance):
        outputs = self.MTTS_CSTM.predict_bpm(arr_appearance, arr_motion)
        # print("Outputs: ", outputs)
            
        inference_array = np.reshape(outputs.cpu().detach().numpy(), (1, -1))
        # print("type inference_array: ", type(inference_array))
        # print("shape inference_array: ", inference_array.shape) (=10)
        # print("inference_array: ", inference_array)
        self.list_infer_arr = np. append(self.list_infer_arr, inference_array)
        # print("list_infer_arr: ", self.list_infer_arr)
        length = len(self.list_infer_arr)
        self.indx = np.arange(length - self.buffer_size, length)
        print("length: ", length)
        if length >= self.buffer_size:
            # =======
            self.new_list_infer_arr = self.list_infer_arr[-self.buffer_size:]
            self.RGB_signal_buffer  = self.BPF_dict(self.new_list_infer_arr, self.sampling_rate)
            # print("self.RGB_signal_buffer: ", len(self.RGB_signal_buffer)) = 300
            self.freqOuput = np.abs(scipy.fft.rfft(self.RGB_signal_buffer, n=5*len(self.RGB_signal_buffer))) / len(self.RGB_signal_buffer)
            # print("self.freqOuput: ", self.freqOuput)
            self.FREQUENCY = scipy.fft.rfftfreq(n=5*self.buffer_size, d=(1 / self.sampling_rate))
            # print("self.FREQUENCY: ", self.FREQUENCY)
            self.FREQUENCY *= 60
            idx = np.argmax(self.freqOuput)
            if not np.isnan(self.bpm):
                self.bpm = self.limit_bpm(self.freqOuput, self.FREQUENCY, 144)
                print("bpm: ",self. bpm)
            else:
                print("nan value")
            self.bpms.append(self.bpm)
            # print("bpms len: ", len(self.bpms))
            # =======
        return inference_array
        
    
    def run(self, rgb_frame):
        dif_frame = None
        mean_frame = None
        color_face = self.fd.face_detect(rgb_frame)
        # print("self.count: ",self.count)
        if color_face is not None:
            # color_face = self.fs.face_segment(color_face)
            if self.count % (self.T +1) == 0:
                if self.count != 0:
                    self.app_frames = self.app_frames[-10:]
                    arr_motion = np.array(self.motion_frames)
                    arr_appearance = np.array(self.app_frames)
                    mean_frame = np.mean(np.array(self.app_frames), axis=0).astype(np.uint8)
                    # print("==========")
                    # print("arr_motion: ", arr_motion.shape)
                    # print("arr_appearance: ", arr_appearance.shape)
                    self.outputs = self.process_model( arr_motion,arr_appearance)
                self.app_frames  = []
                self.motion_frames = []
                self.app_frames.append(color_face)
            else:
                prev_frame = self.app_frames[-1]
                cur_frame = color_face
                self.app_frames.append(color_face)
                # mean frame
                
                dif_frame = self.generate_motion_difference(prev_frame, cur_frame)
                self.motion_frames.append(dif_frame)
            self.count += 1
        # print("type self.RGB_signal_buffer: ", type(self.RGB_signal_buffer))
        return (color_face, dif_frame,mean_frame, self.outputs, self.bpm, self.indx, self.RGB_signal_buffer, self.bpms)

    def reset(self):
        self.RGB_signal_buffer = []
        self.SNR = []
        self.bpms = []
        self.fd = FaceDetection()
        self.count = 0

    def limit_bpm(self, PSD, Frequency, limit):
        idx = np.argmax(PSD)
        bpm = Frequency[idx]
        if bpm > limit:
            wo_max_PSD = PSD.copy()
            wo_max_PSD[idx] = np.min(PSD)
            bpm = self.limit_bpm(wo_max_PSD, Frequency, limit)
        return bpm
    
    def BPF_dict(self, input_val, fs):
        signal = np.squeeze(input_val)
        low = 0.67 / (0.5 * fs)  # Version1
        high = 2.4 / (0.5 * fs)
        sos = butter(10, [low, high], btype='bandpass', output='sos')# Design an Nth-order digital or analog Butterworth filter and return the filter coefficients
        signal = scipy.signal.sosfiltfilt(sos, signal) #A forward-backward digital filter using cascaded second-order sections.
        return signal

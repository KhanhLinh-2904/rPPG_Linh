import time
import numpy as np
import scipy
from face_detection import FaceDetection
from predict_bpm import Prediction_bpm
import scipy.signal
from scipy.signal import butter

class RunAlModels(object):
    def __init__(self):
        self.sampling_rate = 30  # Frame rate of the video input
        self.order = 10  # order of butterworth filter
        self.length = 10
        self.buffer_size = self.sampling_rate * self.length
        self.RGB_signal_buffer = []
        self.fd = FaceDetection()
        self.count = 0  # The second condition to stop the app
        self.MTTS_CSTM = Prediction_bpm()
        self.bpms = []
        self.T = 10
        self.motion_frames = []
        self.app_frames = []
        self.list_infer_arr = []

    def generate_motion_difference(self, prev_frame, cur_frame):
        prev_frame = prev_frame.astype(np.float32)
        cur_frame = cur_frame.astype(np.float32)
        dif_frame = (cur_frame - prev_frame) / (cur_frame + prev_frame + 1)
        return dif_frame

    def run(self, rgb_frame):
        t0 = time.time()
        bpm = 0.
        # first frame detect
        if self.count == 0:
            color_face = self.fd.face_detect(rgb_frame)
            if color_face is not None:
                self.count += 1
                self.app_frames.append(color_face)
        elif self.count % (self.T + 1) != 0:
            color_face = self.fd.face_track(rgb_frame)
            prev_frame = self.app_frames[-1]
            cur_frame = color_face
            self.app_frames.append(color_face)
            dif_frame = self.generate_motion_difference(prev_frame, cur_frame)
            self.motion_frames.append(dif_frame)
            self.count += 1
        else:
            arr_motion = np.array(self.motion_frames)
            arr_appearance = np.array(self.app_frames)
            # print("self.app_frames: ", arr_appearance.shape)
            # print("self.motion_frames: ", arr_motion.shape)
            
            outputs = self.MTTS_CSTM.predict_bpm(arr_appearance, arr_motion)
            # print("Outputs: ", outputs)
            
            inference_array = np.reshape(outputs.cpu().detach().numpy(), (1, -1))
            # print("type inference_array: ", type(inference_array))
            # print("shape inference_array: ", inference_array.shape)
            self.list_infer_arr = np. append(self.list_infer_arr, inference_array)
            length = len(self.list_infer_arr)
            # print("length: ", length)
            if length > self.buffer_size:
                results = {}
                self.list_infer_arr = self.list_infer_arr[-self.buffer_size:]
                results[0] = self.list_infer_arr
                result = self.BPF_dict(results, self.sampling_rate)
                # print("Results: ", result)
                self.RGB_signal_buffer = result[0]
             
                self.freqOuput = np.abs(scipy.fft.rfft(self.RGB_signal_buffer, n=5*len(self.RGB_signal_buffer))) / len(self.RGB_signal_buffer)
                # print("self.freqOuput: ", self.freqOuput)
                self.FREQUENCY = scipy.fft.rfftfreq(n=5*self.buffer_size, d=(1 / self.sampling_rate))
                # print("self.FREQUENCY: ", self.FREQUENCY)
                self.FREQUENCY *= 60
                idx = np.argmax(self.freqOuput)
                if not np.isnan(bpm):
                    bpm = self.limit_bpm(self.freqOuput, self.FREQUENCY, 144)
                else:
                    print("nan value")
                self.bpms.append(bpm)
            # print("self.list_infer_arr: ", self.list_infer_arr)
            last_item = self.app_frames[-1]
            self.app_frames = []
            self.motion_frames = []
            self.app_frames.append(last_item)
            self.count += 1
            
        return (bpm, self.RGB_signal_buffer, self.bpms)

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
        for index, signal in input_val.items():
            signal = np.squeeze(signal)

            if type(fs) == list:
                low = 0.67 / (0.5 * fs[index])  # Frequency range: 0.67-2.4 Hz
                high = 2.4 / (0.5 * fs[index])
            else:
                low = 0.67 / (0.5 * fs)  # Version1
                high = 2.4 / (0.5 * fs)

            sos = butter(10, [low, high], btype='bandpass', output='sos')# Design an Nth-order digital or analog Butterworth filter and return the filter coefficients
            signal = scipy.signal.sosfiltfilt(sos, signal) #A forward-backward digital filter using cascaded second-order sections.
            input_val[index] = signal
        return input_val

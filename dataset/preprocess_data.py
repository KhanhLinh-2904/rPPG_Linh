import numpy as np
import scipy.signal
import json
from scipy.signal import butter
def PURE_preprocess_label(path, frame_total):
    file = open(path)
    data = json.loads(file.read())
    data = data['/FullPackage']
    label = []
    for p in data:
        label.append(p['Value']['waveform'])
    file.close()

    label = np.asarray(label)
    label = BPF_signal(label, 60, 0.4, 4)       
    label = signal_normalization(label)
    label = scipy.signal.resample(label, frame_total)
    return label

def signal_normalization(signal):
    signal = (signal - np.mean(signal)) / np.std(signal)
    return signal

def BPF_signal(input_val, fs, low, high): # Butterworth band-pass filter, fs: frame rate
    input_val = np.squeeze(input_val)
    low = low / (0.5 * fs)  # Frequency range: 0.67-2.5 Hz
    high = high / (0.5 * fs)
    sos = butter(10, [low, high], btype='bandpass', output='sos')
    input_val = scipy.signal.sosfiltfilt(sos, input_val)
    return input_val

if __name__ == "__main__":
    label = PURE_preprocess_label("dataset/01-01.json", 2026)
    np.save("dataset/label_PURE.npy", label)
    # print("label: ", label)
    # Find max and min values
    max_value = np.max(label)
    min_value = np.min(label)

    # print("Max value:", max_value)
    # print("Min value:", min_value)

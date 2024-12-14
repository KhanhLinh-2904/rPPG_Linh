import numpy as np
import matplotlib.pyplot as plt
import scipy


def amplitudeSelectiveFiltering(C_input, sampling_rate=30, red_max=0.0025, nir_max=0.0025, delta=0.0001):
    '''
    Input: Raw RGB signals with dimensions NxL, where the R channel is column 0
    Output:
    C = Filtered RGB-signals with added global mean,
    raw = Filtered RGB signals
    '''

    N = C_input.shape[0]
    L = C_input.shape[1]
    C_mean = np.tile((np.mean(C_input, axis=1) - 1).reshape(C_input.shape[0], 1), L)
    if C_mean is None or len(C_mean) == 0:
        print("C_mean is None or 0")
    C = C_input/C_mean - 1
    F = scipy.fft.rfft(C, n=L, axis=1)
    if L is None or L == 0:
        print("L is None or 0")
    F_mag = np.abs(F) / L
    F_phase = np.exp(1.0j * np.angle(F))
    freqs = 60*scipy.fft.rfftfreq(L, 1/sampling_rate)
    W = np.ones((1, len(freqs)))

    for i in range(1, int(L/2) + 1):
        if np.sum(F_mag[2][i]) >= 1.25 * (F_mag[0][i] + F_mag[1][i]):
            W[0][i] = delta * F_mag[0][i]
            # W[0][i] = 0
        if F_mag.shape[0] > 3:
            if F_mag[0][i] >= red_max or F_mag[3][i] >= nir_max:
                #print("in masf")
                W[0][i] = delta * F_mag[0][i]
        else:
            if F_mag[0][i] >= red_max:
                W[0][i] = delta * F_mag[0][i]
      
    W = np.tile(W, (N, 1))
    F_mag = np.multiply(F_mag, W)

    # line 6
    F_mag = F_mag * L
    F = np.multiply(F_mag, F_phase)
    C = scipy.fft.irfft(F, n=L, axis=1) + 1
    C = C*C_mean

    del F_mag, F, W
    return C

import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def get_inertial_freq(lat):
    rot_rate = 7.2921e-5
    return rot_rate * 2 * np.sin(np.deg2rad(abs(lat)))


from scipy.signal import butter, lfilter, freqz


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def get_cut_off(lat):
    day_hertz = 1 / (60 ** 2 * 24)

    # set it the 1.5*intertial period
    # Divide by 1.5 as perioid = 1/freq
    lp_freq = get_inertial_freq(lat) / 1.5 / (2 * np.pi)
    # lp_freq = 2/get_inertial_freq(lat)/1.5
    if lp_freq > day_hertz:
        cut_off = day_hertz
    elif lp_freq < (day_hertz / 5):
        cut_off = day_hertz / 5
    else:
        cut_off = lp_freq
    return cut_off


def filter_from_lat(x_list, y_list, lat, plot=False):
    ### 60 seconds/minutes 24 hours
    day_hertz = 1 / (60 ** 2 * 24)
    # 4 times per day
    sampling_freq = day_hertz * 4
    cut_off = get_cut_off(lat)
    ## divide by 2pi to get it to hertz
    return hard_filter(x_list, y_list, cut_off, sampling_freq, plot=plot)


def hard_filter(x_list, y_list, cutoff, fs, plot=False):
    freq_domain_data, freqs = fft_and_freq(x_list, y_list, 1 / fs)
    mask = np.abs(freqs) > cutoff
    freq_domain_data[mask] = 0

    x_ret, y_ret = invfft_to_2d(freq_domain_data)
    if plot:
        fig = plt.figure(constrained_layout=True)
        spec2 = gridspec.GridSpec(2, 2, fig)
        ax_x_data = fig.add_subplot(spec2[0, 0])
        ax_y_data = fig.add_subplot(spec2[0, 1])
        ax_fft = fig.add_subplot(spec2[1, :])
        x_axis = list(range(len(x_list)))
        ax_x_data.plot(x_axis, x_list)
        ax_x_data.plot(x_axis, x_ret)
        ax_y_data.plot(x_axis, y_list)
        ax_y_data.plot(x_axis, y_ret)
        ax_fft.plot(freqs, np.abs(freq_domain_data))
        ax_fft.axvline(cutoff)
        ax_fft.axvline(-cutoff)
    return x_ret, y_ret
    # return [(x_ret[i], y_ret[i]) for i in range(x_ret.shape[0])]


def fft_and_freq(x_list, y_list, sample_spacing=1, plot=False, sort_freqs=False):
    cplex = np.array([complex(x, y) for x, y in zip(x_list, y_list)])
    freq_domain_data = fft.fft(cplex)
    freqs = fft.fftfreq(cplex.shape[0], sample_spacing)
    if sort_freqs:
        order = np.argsort(freqs)
        freqs = freqs[order]
        freq_domain_data = freq_domain_data[order]
    return freq_domain_data, freqs


def invfft_to_2d(fft_data):
    time_domain_data = fft.ifft(fft_data)
    x = np.real(time_domain_data)
    y = np.imag(time_domain_data)
    return x, y

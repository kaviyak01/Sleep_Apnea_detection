# visualize_ecg.py
import matplotlib.pyplot as plt
import numpy as np

def plot_ecg_with_apnea(ecg_signal, labels_bin, segment_len=6000, downsample_factor=100):
    time_ds = np.arange(0, len(ecg_signal), downsample_factor)
    ecg_ds = ecg_signal[::downsample_factor]

    plt.figure(figsize=(18,6))
    plt.plot(time_ds, ecg_ds, color='blue', linewidth=0.8, label='ECG')

    last_marked = -segment_len
    for i, label in enumerate(labels_bin):
        if label == 1 and i*segment_len - last_marked > segment_len:
            mid = i*segment_len + segment_len//2
            plt.scatter(mid, ecg_signal[mid], s=1500, facecolors='none', edgecolors='red', linewidths=3, zorder=2)
            last_marked = i*segment_len

    plt.title("ECG with Apnea Events", fontsize=16)
    plt.xlabel("Samples", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.legend()
    plt.show()

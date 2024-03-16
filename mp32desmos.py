import librosa
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def replace_below_threshold(arr, threshold):
    for i in range(len(arr)):
        if arr[i] < threshold:
            arr[i] = 0
    return arr
            
            

def get_dominant_freqs(audio_file, n_freqs=40, n_bands=40):
    # Load audio file
    y, sr = librosa.load(audio_file)

    # Divide audio into time slices
    n_slices = int(np.round(librosa.get_duration(y=y, sr=sr) * 20))
    hop_length = int(len(y) // n_slices)
    slices = [y[i:i+hop_length] for i in range(0, int(len(y)), int(hop_length))]

    # Get dominant frequencies and their intensities for each slice
    freqs = []
    amps = []
    for slice in slices:
        D = np.abs(librosa.stft(slice))
        n_fft = 2 * (D.shape[0] - 1)
        fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        freqs_slice, amps_slice = [], []

        # Divide frequency range into logarithmically-spaced bands
        band_edges = np.logspace(np.log10(20), np.log10(sr // 2), num=n_bands + 1)
        for i in range(n_bands):
            band_start, band_end = band_edges[i], band_edges[i + 1]
            band_mask = np.logical_and(fft_freqs >= band_start, fft_freqs < band_end)
            band_values = D[band_mask, :]

            # Find dominant frequency and amplitude in the band
            if len(band_values) > 0:
                max_idx = np.argmax(band_values)
                max_freq_bin = np.unravel_index(max_idx, band_values.shape)[0]
                max_freq = np.round(fft_freqs[band_mask][max_freq_bin], 1)
                max_amp = np.round(band_values.flatten()[max_idx], 4)

                # Add dominant frequency and amplitude to the output
                freqs_slice.append(max_freq)
                amps_slice.append(max_amp)

        freqs.append(replace_below_threshold(freqs_slice, 100))
        amps.append(replace_below_threshold(amps_slice, 0.0001))

    # Transpose frequency and amplitude matrices
    freqs = list(map(list, zip(*freqs)))
    amps = list(map(list, zip(*amps)))

    # Normalize amplitudes
    max_amp = np.max([np.max(amp) for amp in amps])
    if max_amp > 0:
        amps = [2 * np.array(amp) / max_amp for amp in amps]
    else:
        amps = [np.zeros_like(amp) for amp in amps]

    # Format output
    freqs_output = ",".join([f"\\left[{','.join([str(freq) for freq in freqs[i]])}\\right]\\left[x\\right]" for i in range(len(freqs))])
    amps_output = ",".join([f"\\left[{','.join([str('{:.4f}'.format(amp)) for amp in amps[i]])}\\right]\\left[x\\right]" for i in range(len(amps))])

    return freqs_output, amps_output



# Usage

Tk().withdraw()
audio_file = askopenfilename()
freqs, amps = get_dominant_freqs(audio_file, n_freqs=80)
print("Frequencies(put inside 'F(x) = ['):\n", freqs)
print("\nAmplitudes(put inside 'A(x) = ['):\n", amps)

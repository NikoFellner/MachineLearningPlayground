from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

def load_sample(filename, duration=4*44100, offset=44100//10):
    # Complete this function
    directory = os.path.join("C:/10_Studium/Masterstudium/20_StudienSemester/04_SoSe23/Introduction_to_Machine_Learning/Exercise/introml_ex1/introml_ex1/", filename)
    piano_data = np.load(directory)
    max_value = np.argmax(piano_data) + offset
    piano_data = piano_data[max_value:max_value+duration]
    return piano_data

def compute_frequency(signal, min_freq=20):
    frequency = np.fft.fft(signal)
    frequency = abs(frequency)
    threshold = int(min_freq * (len(signal)*(1/44100)))
    frequency[0:threshold] = 0
    #plt.plot(frequency)
    #plt.show()
    highestPeak2 = np.argmax(frequency[0:int(len(signal)/2)])
    peakFrequency = highestPeak2/(len(signal)*(1/44100))
    return peakFrequency

if __name__ == '__main__':
    # Implement the code to answer the questions here
    files = os.listdir("sounds")
    pianoFrequency = np.empty(len(files))
    for f in files:
        signal = load_sample(os.path.join("sounds",f),duration=4*44100, offset=44100//10)
        note = compute_frequency(signal)
        pianoFrequency[list.index(files,f)] = note
    #plt.plot(pianoFrequency)
    #plt.show()
    print(pianoFrequency)

# This will be helpful:
# https://en.wikipedia.org/wiki/Piano_key_frequencies
# A2 (110.0000 Hz   -> 109.5866767 Hz)
# A3 (220.0000 Hz   -> 220.3422339 Hz)
# A4 (440.0000 Hz   -> 440.7500000 Hz)
# A5 (880.0000 Hz   -> 883.5000000 Hz)
# A6 (1760.0000 Hz  -> 1776.26640886 Hz)
# A7 (3520.0000 Hz  -> 3610.25124768 Hz)
# 1179.87801675 Hz (Könnte D6 sein mit 1174.659 Hz)
# Das Gehör von Alice lässt mit steigender Tonhöhe (Frequenz) nach.
# Je höher der Ton, desto größer die Abweichungen

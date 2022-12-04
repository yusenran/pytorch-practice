from typing import Callable

import pyaudio
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def audio_io(freq:int, frame_size:int, filter: Callable[[bytes], bytes]):
    p=pyaudio.PyAudio()
    fr = freq
    fn=51200*N/50
    fs=fn/fr

    stream=p.open( format = pyaudio.paInt16,
            channels = 1,
            rate = freq,
            frames_per_buffer = frame_size,
            input = True,
            output = True) # inputとoutputを同時にTrueにする
    # Figureの初期化
    fig = plt.figure(figsize=(16, 8)) #...1
    # Figure内にAxesを追加()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    while stream.is_active():
        input = stream.read(frame_size)
        output = stream.write(filter(input))
        # sig =[]
        # sig = np.frombuffer(input, dtype="int16")  /32768.0

        # nperseg = 1024*N

        # f, t, Zxx = signal.stft(sig, fs=fn, nperseg=nperseg)
        # ax2.pcolormesh(fs*t, f/fs, np.abs(Zxx), cmap='hsv')
        # ax2.set_xlim(0,fs)
        # ax2.set_ylim(2,20000)
        # ax2.set_yscale('log')
        # ax2.set_axis_off()
        # x = np.linspace(0, 100, nperseg)
        # ax1.plot(x,sig)
        # ax1.set_ylim(-0.5,0.5)
        # plt.pause(0.01)
        # plt.clf()
        # ax1 = fig.add_subplot(211)
        # ax2 = fig.add_subplot(212)

    stream.stop_stream()
    stream.close()
    print( "Stop Streaming")

def pass_through(input: bytes) -> bytes:
    return input

if __name__ == "__main__":
    RATE=44100
    N=1
    # CHUNK=1024*N
    CHUNK=32*N
    audio_io(RATE, CHUNK, pass_through)

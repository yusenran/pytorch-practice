
import pyaudio
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

CHUNK=1024
RATE=44100
p=pyaudio.PyAudio()
N=10
CHUNK=1024*N
p=pyaudio.PyAudio()
fr = RATE
fn=51200*N/50
fs=fn/fr

stream=p.open(  format = pyaudio.paInt16,
        channels = 1,
        rate = RATE,
        frames_per_buffer = CHUNK,
        input = True,
        output = True) # inputとoutputを同時にTrueにする
# Figureの初期化
fig = plt.figure(figsize=(16, 8)) #...1
# Figure内にAxesを追加()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
while stream.is_active():
    input = stream.read(CHUNK)
    output = stream.write(input)
    sig =[]
    sig = np.frombuffer(input, dtype="int16")  /32768.0

    nperseg = 1024*N

    f, t, Zxx = signal.stft(sig, fs=fn, nperseg=nperseg)
    ax2.pcolormesh(fs*t, f/fs, np.abs(Zxx), cmap='hsv')
    ax2.set_xlim(0,fs)
    ax2.set_ylim(2,20000)
    ax2.set_yscale('log')
    ax2.set_axis_off()
    x = np.linspace(0, 100, nperseg)
    ax1.plot(x,sig)
    ax1.set_ylim(-0.5,0.5)
    plt.pause(0.01)
    plt.clf()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

stream.stop_stream()
stream.close()
print( "Stop Streaming")

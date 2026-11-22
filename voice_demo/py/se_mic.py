#!/usr/bin/env python3
"""Live denoising demo: record from the microphone (or take a file), enhance with the CPU binary, play both.

    python3 se_mic.py --bin ../../bin_cpu/se_infer --model ../../models/se/s1_e25.mod [--seconds 4] [--file noisy.wav]

Records --seconds of audio at 16 kHz (speak with some background noise: a fan, music, tapping), writes
noisy.wav and enhanced.wav next to each other in a temp directory, plays the noisy then the enhanced version,
and prints the RMS of both. With --file the recording step is skipped. Needs: pip install sounddevice numpy.
"""
import argparse
import os
import subprocess
import tempfile
import time
import wave

import numpy as np
import sounddevice as sd

RATE = 16000


def read_wav(path):
    with wave.open(path, 'rb') as w:
        assert w.getsampwidth() == 2 and w.getframerate() == RATE, (w.getsampwidth(), w.getframerate())
        x = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2')
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels())[:, 0]
    return x.copy()


def write_wav(path, x):
    with wave.open(path, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE); w.writeframes(x.astype('<i2').tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--seconds', type=float, default=4.0)
    ap.add_argument('--file', default='')
    ap.add_argument('--no-play', action='store_true')
    ap.add_argument('--extra', default='', help='extra flags for se_infer')
    a = ap.parse_args()
    d = tempfile.mkdtemp(prefix='se_demo_')
    noisy_path, enh_path = os.path.join(d, 'noisy.wav'), os.path.join(d, 'enhanced.wav')
    if a.file:
        x = read_wav(a.file)
        write_wav(noisy_path, x)
    else:
        print('input device:', sd.query_devices(kind='input')['name'])
        input('Enter = record %.1f s (speak over some background noise): ' % a.seconds)
        rec = sd.rec(int(a.seconds * RATE), samplerate=RATE, channels=1, dtype='int16')
        sd.wait()
        x = rec[:, 0].copy()
        write_wav(noisy_path, x)
    cmd = [a.bin, '-model', a.model, '-wav', noisy_path, '-out', enh_path] + a.extra.split()
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stderr); return
    print(out.stderr.strip(), '(%.2f s wall)' % (time.time() - t0))
    y = read_wav(enh_path)
    rms = lambda v: float(np.sqrt(np.mean(v.astype(np.float64) ** 2))) / 32768
    print('rms noisy %.4f  enhanced %.4f  (%.1f dB)' % (rms(x), rms(y), 20 * np.log10((rms(y) + 1e-9) / (rms(x) + 1e-9))))
    print('files:', noisy_path, enh_path)
    if not a.no_play:
        for name, v in (('noisy', x), ('enhanced', y)):
            print('playing', name)
            sd.play(v, RATE); sd.wait()


if __name__ == '__main__':
    main()

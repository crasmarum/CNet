#!/usr/bin/env python3
"""Live keyword-spotting demo: record from the microphone, classify with the CPU inference binary.

    python3 kws_mic.py --bin ../../bin_cpu/kws_infer --model ../../models/kws/r19.mod --labels ../../models/kws/labels.txt

Press Enter, say one of the 35 words within the next 1.5 s, and the top predictions are printed. The loudest
one-second window of the recording is used (the trainer saw one-second clips with the word roughly centred).
Needs: pip install sounddevice numpy. 'q' + Enter quits; --loop 0 records once.
"""
import argparse
import os
import subprocess
import sys
import tempfile
import wave

import numpy as np
import sounddevice as sd

RATE = 16000


def loudest_second(x, step=160):
    n = len(x)
    if n <= RATE:
        return np.pad(x, (0, RATE - n))
    best, best_e = 0, -1.0
    sq = x.astype(np.float64) ** 2
    cs = np.concatenate([[0.0], np.cumsum(sq)])
    for s in range(0, n - RATE + 1, step):
        e = cs[s + RATE] - cs[s]
        if e > best_e:
            best, best_e = s, e
    return x[best:best + RATE]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--seconds', type=float, default=1.5)
    ap.add_argument('--loop', type=int, default=1)
    ap.add_argument('--extra', default='', help='extra flags for kws_infer, e.g. "-stft_window 512 -stft_hop 160"')
    a = ap.parse_args()
    print('input device:', sd.query_devices(kind='input')['name'])
    tmp = os.path.join(tempfile.gettempdir(), 'kws_clip.wav')
    while True:
        s = input('Enter = record %.1f s (q = quit): ' % a.seconds)
        if s.strip().lower() == 'q':
            break
        rec = sd.rec(int(a.seconds * RATE), samplerate=RATE, channels=1, dtype='int16')
        sd.wait()
        x = loudest_second(rec[:, 0])
        rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) / 32768
        with wave.open(tmp, 'wb') as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE); w.writeframes(x.tobytes())
        cmd = [a.bin, '-model', a.model, '-labels', a.labels, '-wav', tmp] + a.extra.split()
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            print(out.stderr); break
        print('  rms %.3f  %s' % (rms, out.stderr.strip()))
        for line in out.stdout.strip().splitlines():
            word, prob = line.split('\t')
            print('  %-10s %5.1f%%  %s' % (word, 100 * float(prob), '#' * int(40 * float(prob))))
        if not a.loop:
            break


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Continuous keyword listener: the microphone is streamed into one long-lived kws_stream process (constant-state
Born attention, one posterior per 10 ms hop), and its detections are printed as they happen.

    python3 kws_listen_stream.py --bin bin_cpu/kws_stream --model models/kws/r21_e24.mod --labels models/kws/labels17.txt \
        --words stop,go,left,right,forward,backward,follow --min-prob 0.65

Compared with kws_listen.py there is no RMS trigger, no "loudest second" and no process per decision: every hop is
classified with the model's full causal context, and the decision rule (posterior averaging, threshold, cooldown)
lives in kws_stream. Needs the `sounddevice` package (the demo venv has it).
"""
import argparse
import subprocess
import sys
import threading

import numpy as np
import sounddevice as sd

RATE = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', required=True, help='path to kws_stream')
    ap.add_argument('--model', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--words', default='', help='comma-separated command set (empty: every class except unknown/silence)')
    ap.add_argument('--min-prob', type=float, default=0.65)
    ap.add_argument('--high-prob', type=float, default=0.95, help='a word at or above this posterior is printed in CAPITALS')
    ap.add_argument('--smooth', type=int, default=5, help='hops averaged before a decision (5 = 50 ms)')
    ap.add_argument('--cooldown-ms', type=int, default=800)
    ap.add_argument('--print-every', type=int, default=0, help='also print the running arg-max every N hops')
    ap.add_argument('--extra', default='', help='extra kws_stream flags, e.g. "-input_gain 50 -stft_hop 160"')
    ap.add_argument('--device', default=None, help='sounddevice input device (name or index)')
    ap.add_argument('--seconds', type=float, default=0, help='stop after this many seconds (0 = until Ctrl-C)')
    args = ap.parse_args()

    cmd = [args.bin, '-model', args.model, '-labels', args.labels, '-stdin_pcm', 'true', '-min_prob', str(args.min_prob),
           '-smooth', str(args.smooth), '-cooldown_ms', str(args.cooldown_ms), '-print_every', str(args.print_every),
           '-high_prob', str(args.high_prob)]
    if args.words:
        cmd += ['-words', args.words]
    if args.extra:
        cmd += args.extra.split()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)

    def pump(stream, prefix=''):
        for line in iter(stream.readline, b''):
            sys.stdout.write(prefix + line.decode('utf-8', 'replace'))
            sys.stdout.flush()

    threading.Thread(target=pump, args=(proc.stdout,), daemon=True).start()
    threading.Thread(target=pump, args=(proc.stderr, '[kws_stream] '), daemon=True).start()

    hop = 160                      # 10 ms blocks: one hop per callback

    def callback(indata, frames, time_info, status):
        if status:
            sys.stderr.write(str(status) + '\n')
        pcm = np.clip(indata[:, 0] * 32768.0, -32768, 32767).astype('<i2')
        try:
            proc.stdin.write(pcm.tobytes())
        except BrokenPipeError:
            raise sd.CallbackStop()

    print('listening (Ctrl-C to stop) ...', flush=True)
    try:
        with sd.InputStream(samplerate=RATE, channels=1, dtype='float32', blocksize=hop, device=args.device, callback=callback):
            if args.seconds > 0:
                sd.sleep(int(args.seconds * 1000))
            else:
                while proc.poll() is None:
                    sd.sleep(200)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait(timeout=5)


if __name__ == '__main__':
    main()

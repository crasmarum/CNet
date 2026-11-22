#!/bin/sh
# Hear the denoiser on the bundled clip (seven seconds of a VoiceBank-DEMAND test recording): the noisy original,
# a one-second pause, then the version cleaned by the streaming s8 twin. No microphone needed.
#
#   ./play_demo.sh              clean and play
#   ./play_demo.sh --no-play    clean only (writes clips/clip2_enhanced.wav)
cd "$(dirname "$0")" || exit 1
[ -x bin/se_stream ] || { echo "run ./build.sh first"; exit 1; }
IN=clips/clip2_noisy.wav
OUT=clips/clip2_enhanced.wav
rm -f "$OUT"                                      # never play a stale result
if ! bin/se_stream -model models/s8_e49.mod -wav "$IN" -out "$OUT" -gain running > clips/se_stream.log 2>&1 \
   || [ ! -s "$OUT" ]; then
	echo "denoising failed:"; tail -5 clips/se_stream.log; exit 1
fi
tail -1 clips/se_stream.log
[ "$1" = "--no-play" ] && { echo "written $OUT"; exit 0; }

play() {
	if command -v afplay > /dev/null; then afplay "$1"
	elif command -v paplay > /dev/null; then paplay "$1"
	elif command -v aplay > /dev/null; then aplay -q "$1"
	elif [ -x .venv/bin/python ]; then .venv/bin/python -c "import sys, wave, numpy as np, sounddevice as sd
w = wave.open(sys.argv[1]); x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16); sd.play(x, w.getframerate()); sd.wait()" "$1"
	else echo "no audio player found (afplay, paplay, aplay or the .venv)"; exit 1
	fi
}
echo "noisy ..."; play "$IN"
sleep 1
echo "cleaned ..."; play "$OUT"

#!/bin/sh
# Record a few seconds from the microphone with some background noise, clean it with the streaming twin of the
# s8 denoiser (one 16 ms frame at a time, causal gain, as it would run on a device), then play the recording and
# the cleaned version one after the other.
#
#   ./denoise.sh                    record 5 s, then play both
#   ./denoise.sh --seconds 8        record longer
#   ./denoise.sh --file noisy.wav   clean a 16 kHz mono WAV file instead of recording
#   ./denoise.sh --offline          use the offline path (overlapping two-second windows) instead of the stream
#   ./denoise.sh --no-play          do not play, just write the files and print their levels
cd "$(dirname "$0")" || exit 1
[ -x bin/se_stream ] && [ -x .venv/bin/python ] || { echo "run ./build.sh first"; exit 1; }
BIN=bin/se_stream
EXTRA="-gain running"
SECONDS_OPT="--seconds 5"
for a in "$@"; do
	[ "$a" = "--offline" ] && { BIN=bin/se_infer; EXTRA=; }
	[ "$a" = "--seconds" ] && SECONDS_OPT=
done
# drop our own --offline flag before passing the rest to the script
ARGS=
for a in "$@"; do [ "$a" != "--offline" ] && ARGS="$ARGS $a"; done
# shellcheck disable=SC2086
exec .venv/bin/python py/se_mic.py --bin $BIN --model models/s8_e49.mod $SECONDS_OPT --extra "$EXTRA" $ARGS

#!/bin/sh
# Record a few seconds from the microphone with some background noise, clean it with the streaming twin of the
# s8 denoiser (one 16 ms frame at a time, causal running gain, as it would run on a device), then play the
# recording and the cleaned version one after the other.
#
#   ./denoise.sh                    record 5 s, then play both
#   ./denoise.sh --seconds 8        record longer
#   ./denoise.sh --file noisy.wav   clean a 16 kHz mono WAV file instead of recording (any path, spaces allowed)
#   ./denoise.sh --offline          the offline path (overlapping two-second windows, per-window gain: the
#                                   setting of the published scores) instead of the stream
#   ./denoise.sh --no-play          do not play, just write the files and print their levels
CALLER=$(pwd)
cd "$(dirname "$0")" || exit 1
[ -x bin/se_stream ] && [ -x .venv/bin/python ] || { echo "run ./build.sh first"; exit 1; }
BIN=bin/se_stream
EXTRA="-gain running"
HAVE_SECONDS=0
NEXT_IS_FILE=0
# rebuild the argument list: drop --offline, resolve a relative --file against the caller's directory, keep
# every other argument intact (one shift per original argument, kept ones appended at the end)
n=$#; i=0
while [ "$i" -lt "$n" ]; do
	a=$1; shift; i=$((i + 1))
	if [ "$NEXT_IS_FILE" = 1 ]; then
		case "$a" in /*) ;; *) a="$CALLER/$a" ;; esac
		[ -f "$a" ] || { echo "denoise.sh: no such file: $a"; exit 1; }
		NEXT_IS_FILE=0
		set -- "$@" "$a"; continue
	fi
	case "$a" in
	--offline) BIN=bin/se_infer; EXTRA= ;;
	--file) NEXT_IS_FILE=1; set -- "$@" "$a" ;;
	--seconds) HAVE_SECONDS=1; set -- "$@" "$a" ;;
	*) set -- "$@" "$a" ;;
	esac
done
[ "$HAVE_SECONDS" = 0 ] && set -- --seconds 5 "$@"
exec .venv/bin/python py/se_mic.py --bin "$BIN" --model models/s8_e49.mod --extra "$EXTRA" "$@"

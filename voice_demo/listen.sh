#!/bin/sh
# Hands-free command spotting from the microphone with the streaming twin of the r28 model: one posterior every
# 10 ms, a word is printed when it fires, in CAPITALS when the model is sure. Ctrl-C stops it.
#
#   ./listen.sh                    the robot's seven commands: stop, go, left, right, forward, backward, follow
#   ./listen.sh --all              any of the 35 words of the benchmark
#   ./listen.sh --ptt              push-to-talk: press Enter, say a word, see the top guesses (clip classifier)
#   ./listen.sh --words yes,no     your own subset; other kws_listen_stream.py options pass through
cd "$(dirname "$0")" || exit 1
[ -x bin/kws_stream ] && [ -x .venv/bin/python ] || { echo "run ./build.sh first"; exit 1; }
MODEL=models/r28_e37.mod
LABELS=models/labels.txt
WORDS=stop,go,left,right,forward,backward,follow
case "$1" in
--ptt)
	shift
	exec .venv/bin/python py/kws_mic.py --bin bin/kws_infer --model $MODEL --labels $LABELS "$@" ;;
--all)
	shift
	WORDS= ;;
esac
if [ -n "$WORDS" ]; then
	set -- --words "$WORDS" "$@"
fi
exec .venv/bin/python py/kws_listen_stream.py --bin bin/kws_stream --model $MODEL --labels $LABELS \
	--min-prob 0.8 --high-prob 0.8 --print-every 50 "$@"

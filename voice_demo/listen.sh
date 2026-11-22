#!/bin/sh
# Hands-free command spotting from the microphone with a streaming twin: one posterior every 10 ms, a word is
# printed when it fires, in CAPITALS when the model is sure. Ctrl-C stops it.
#
#   ./listen.sh                    r28, the 35-word benchmark model (95.1% clip accuracy), listening for the
#                                  robot's seven commands: stop, go, left, right, forward, backward, follow
#   ./listen.sh --robot            r21, the 17-class model trained with unknown and silence classes: the one
#                                  behind the continuous-audio numbers (80% recall, 6 false accepts in 15 min)
#   ./listen.sh --all              r28 on any of its 35 words
#   ./listen.sh --ptt              push-to-talk: press Enter, say a word, see r28's top guesses (clip classifier)
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
--robot)
	shift
	MODEL=models/r21_e24.mod
	LABELS=models/labels17.txt ;;
esac
if [ -n "$WORDS" ]; then
	set -- --words "$WORDS" "$@"
fi
exec .venv/bin/python py/kws_listen_stream.py --bin bin/kws_stream --model $MODEL --labels $LABELS \
	--min-prob 0.8 --high-prob 0.8 --print-every 50 "$@"

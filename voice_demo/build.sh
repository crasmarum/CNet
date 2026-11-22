#!/bin/sh
# Build the voice demo: the four CPU programs (command spotter and denoiser, streaming twin and offline) into
# voice_demo/bin, and a Python environment with the microphone package into voice_demo/.venv.
#
#   ./build.sh            build everything (about two minutes the first time)
#   ./build.sh --clean    remove bin/, obj/ and .venv/ first
#
# Works from the public CNet repository (voice_demo/ next to the Makefile) and from the research tree
# (voice_demo/ next to src/). macOS needs the Xcode command-line tools and `brew install libomp`;
# Linux needs g++, make, python3 with venv, and the PortAudio library for the microphone
# (Debian/Ubuntu: sudo apt install g++ make python3-venv libportaudio2).
set -e
cd "$(dirname "$0")"
DEMO=$(pwd)

if [ "$1" = "--clean" ]; then
	rm -rf bin obj .venv
fi

# where is the CNet Makefile?
if [ -f ../Makefile ]; then SRC=..
elif [ -f ../src/Makefile ]; then SRC=../src
else echo "build.sh: cannot find the CNet Makefile (expected ../Makefile or ../src/Makefile)"; exit 1
fi

# compiler flags: Apple clang with Homebrew's libomp on macOS, g++ with its own OpenMP on Linux
case "$(uname -s)" in
Darwin)
	OMP=$(brew --prefix libomp 2>/dev/null || echo /opt/homebrew/opt/libomp)
	if [ ! -d "$OMP/include" ]; then
		echo "build.sh: OpenMP runtime not found; run: brew install libomp"; exit 1
	fi
	CXX=g++
	CXXFLAGS="-std=c++14 -Xclang -fopenmp -O2 -MMD -MP -I$OMP/include"
	LDFLAGS="-L$OMP/lib -lomp"
	;;
*)
	CXX=g++
	CXXFLAGS="-std=c++14 -fopenmp -O2 -MMD -MP"
	LDFLAGS="-lgomp"
	;;
esac

mkdir -p bin obj
for prog in kws_stream kws_infer se_stream se_infer; do
	echo "==> building $prog"
	( cd "$SRC" && make "$prog" TARGET_DIR="$DEMO/bin" OBJ_DIR="$DEMO/obj" CXX="$CXX" CXXFLAGS="$CXXFLAGS" LDFLAGS="$LDFLAGS" > "$DEMO/obj/$prog.build.log" 2>&1 ) \
		|| { echo "build of $prog failed; see obj/$prog.build.log"; tail -20 "obj/$prog.build.log"; exit 1; }
done

# python environment for the microphone scripts (listen.sh, denoise.sh); play_demo.sh does not need it
if [ ! -x .venv/bin/python ]; then
	echo "==> creating .venv (python3 + sounddevice + numpy)"
	if python3 -m venv .venv && .venv/bin/pip install -q --upgrade pip && .venv/bin/pip install -q sounddevice numpy; then
		:
	else
		rm -rf .venv
		echo "    could not create the Python environment: the microphone scripts need python3 with venv and pip"
		echo "    (Debian/Ubuntu: sudo apt install python3-venv libportaudio2), then run ./build.sh again."
		echo "    play_demo.sh works without it."
	fi
fi

echo
echo "built: $(ls bin | tr '\n' ' ')"
echo "next:  ./play_demo.sh   (hear the bundled clip before and after denoising)"
echo "       ./listen.sh      (say stop, go, left, right, forward, backward, follow)"
echo "       ./denoise.sh     (record 5 s with background noise, hear it cleaned)"

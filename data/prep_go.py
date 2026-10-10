#!/usr/bin/env python3
"""Encode a Go game corpus into uint16 move-token bins for the CNet gollm example.

Input: a text file (or .zip containing one) where each LINE is one game: the
3-char token "222" (game start) followed by 3-char moves "<color><col><row>",
e.g. "2220qp1dq0cd..." -> start, B q16/p16-style coords, W ...  Each move is
  id = color*361 + 19*(col-'a') + (row-'a')      (0..360 black, 361..721 white)
and "222" -> 722, so the vocabulary is 723 (matches go_gen.py / go_gen2.py).

Output (in --out): train.bin, val.bin (little-endian uint16), meta.txt. The two
bins are one contiguous token stream each (games concatenated, every game begins
with its 722), which is exactly what gollm's TokenData reader windows over. Every
--val_every_n-th game goes to val, the rest to train (mirrors the scripts' 90/10).
"""
import argparse, os, sys, zipfile, io
import numpy as np

VOCAB = 723
START = 722


def encode(tok: str) -> int:
    if tok == "222":
        return START
    c = (ord(tok[0]) - ord('0')) * 361
    x = 19 * (ord(tok[1]) - ord('a'))
    y = (ord(tok[2]) - ord('a'))
    v = c + x + y
    if not (0 <= v < VOCAB):
        raise ValueError(f"bad token {tok!r} -> {v}")
    return v


def open_lines(path):
    """Yield text lines from a .txt or from the first .txt inside a .zip."""
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist()
                        if n.endswith(".txt") and not n.startswith("__MACOSX"))
            with z.open(name) as f:
                for line in io.TextIOWrapper(f, encoding="utf-8", errors="replace"):
                    yield line
    else:
        with open(path, "r", errors="replace") as f:
            yield from f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="go_data.txt or .zip")
    ap.add_argument("--out", default=os.path.expanduser("~/godata"))
    ap.add_argument("--val_every_n", type=int, default=10, help="every Nth game -> val")
    ap.add_argument("--min_moves", type=int, default=2, help="skip games shorter than this")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    train, val = [], []
    games = skipped = bad = 0
    for line in open_lines(args.inp):
        s = line.strip()
        if not s:
            continue
        if len(s) % 3 != 0:
            bad += 1
            continue
        try:
            ids = [encode(s[i:i + 3]) for i in range(0, len(s), 3)]
        except (ValueError, IndexError):
            bad += 1
            continue
        if len(ids) < args.min_moves:
            skipped += 1
            continue
        games += 1
        (val if games % args.val_every_n == 0 else train).extend(ids)

    if not train or not val:
        sys.exit(f"empty split (train={len(train)} val={len(val)}) -- check --in / --val_every_n")

    np.asarray(train, dtype=np.uint16).tofile(os.path.join(args.out, "train.bin"))
    np.asarray(val, dtype=np.uint16).tofile(os.path.join(args.out, "val.bin"))
    with open(os.path.join(args.out, "meta.txt"), "w") as f:
        f.write(f"vocab={VOCAB}\ntrain_tokens={len(train)}\nval_tokens={len(val)}\n"
                f"games={games}\ndtype=uint16\n")

    print(f"DONE  games={games:,} (bad={bad}, skipped_short={skipped})  "
          f"train={len(train):,}  val={len(val):,}  vocab={VOCAB}  -> {args.out}")


if __name__ == "__main__":
    main()

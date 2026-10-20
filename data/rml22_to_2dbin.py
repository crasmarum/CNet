#!/usr/bin/env python3
"""Convert the DeepSig RML22.01A pickle to the RML1 .bin format, laid out as the
VT-CNN2 2xL image flattened row-major into a length-2L REAL vector:

    sample = [ I_0 .. I_{L-1},  Q_0 .. Q_{L-1} ]   (imag part = 0)

so a complex Hadamard-conv net can run VT-CNN2's 2D convolutions via kernel_index
spanning the flattened layout (row stride = L): a (1,3) kernel is {0,1,2}, a (2,3)
kernel is {0,1,2, L,L+1,L+2} (the second row mixes the Q channel into the I row).

RML1 on disk: int32 magic,N,Lout,C then N x { int32 label, 2*Lout float32
interleaved re,im } with Lout = 2L and im == 0. Per-example power normalization
over the 2xL image. 50/50 stratified split + per-SNR test bins, matching the
earlier RML22 pipeline (labels in sorted-modulation order).

Usage: python3 rml22_to_2dbin.py RML22.01A out_dir
"""
import sys, struct, pickle
import numpy as np

MAGIC = 0x524D4C31


def main():
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "."
    d = pickle.load(open(src, "rb"), encoding="latin1")
    mods = sorted({k[0] for k in d}); snrs = sorted({k[1] for k in d})
    m2l = {m: i for i, m in enumerate(mods)}
    L = np.asarray(d[list(d)[0]]).shape[2]      # 128
    Lout = 2 * L
    C = len(mods)
    print("mods:", mods, "\nsnrs:", snrs, "L:", L, "-> Lout:", Lout, "C:", C)
    rng = np.random.default_rng(0)

    def recs(arr):                               # arr: (n, 2, L) -> list of (2L,) real, power-normed
        o = []
        for x in arr:
            I = x[0].astype(np.float32); Q = x[1].astype(np.float32)
            p = float(np.sqrt((I * I + Q * Q).mean())) + 1e-9
            o.append(np.concatenate([I / p, Q / p]).astype(np.float32))   # [I | Q]
        return o

    def write(path, recs_list):
        with open(path, "wb") as f:
            f.write(struct.pack("<iiii", MAGIC, len(recs_list), Lout, C))
            for lab, v in recs_list:
                f.write(struct.pack("<i", lab))
                iq = np.zeros(2 * Lout, np.float32); iq[0::2] = v     # re = v, im = 0
                f.write(iq.tobytes())
        print("wrote", len(recs_list), "->", path)

    train, test_by_snr = [], {s: [] for s in snrs}
    for (mod, snr), arr in d.items():
        arr = np.asarray(arr); idx = rng.permutation(len(arr))
        r = recs(arr[idx]); lab = m2l[mod]
        half = len(r) // 2
        for v in r[:half]:
            train.append((lab, v))
        for v in r[half:]:
            test_by_snr[snr].append((lab, v))

    rng.shuffle(train)
    test_all = [rec for s in snrs for rec in test_by_snr[s]]
    write(f"{out}/rml22_2d_train.bin", train)
    write(f"{out}/rml22_2d_test.bin", test_all)
    for s in snrs:
        write(f"{out}/rml22_2d_test_snr{s}.bin", test_by_snr[s])
    print("labels:", m2l)


if __name__ == "__main__":
    main()

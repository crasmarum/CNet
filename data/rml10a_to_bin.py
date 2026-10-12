#!/usr/bin/env python3
"""Convert the real DeepSig RML2016.10a dataset to the RML1 .bin format read by
the C++ RadioMLDataReader (see gen_radioml.py for the format spec).

Input:  RML2016.10a_dict.pkl  -- a Python2 pickle of
            {(mod_str, snr_int): float32 ndarray (1000, 2, 128)}
        11 modulations x 20 SNRs (-20..+18 dB step 2) x 1000 = 220000 examples.
Output (little-endian RML1: int32 magic,N,L,C then N x [int32 label, 2L f32 I/Q]):
    rml10a_train.bin           50% split, all SNRs mixed (110000 examples)
    rml10a_test.bin            the other 50% (110000)
    rml10a_test_snr<S>.bin     per-SNR held-out splits, one per SNR (for the curve)
Each example is power-normalised (I,Q divided by sqrt(mean(I^2+Q^2))). Labels are
0..10 in sorted-modulation order:
    8PSK AM-DSB AM-SSB BPSK CPFSK GFSK PAM4 QAM16 QAM64 QPSK WBFM

Usage: python3 rml10a_to_bin.py RML2016.10a_dict.pkl [out_dir]

Held-out benchmark of arch=11 (one-FFT learnable spectral conv bank -> pooled
|.|^2/|.|^4 power + z^2/z^4 phase moments -> MLP head) on this data, trained with
the batch-fixed GPU path (true Adam, 50/50 split):
    overall test accuracy ~0.47 ; per-SNR ~chance at -20 dB rising to a ~0.70
    plateau from ~4 dB up. Scaling capacity (n_channels 32->128, pool_out 4->8,
    hidden_dim 64->256) does NOT raise held-out accuracy -- it only overfits
    (train ->0.69, test stays ~0.47). So the ceiling is the pooled-power feature
    construction, not model size; it sits ~20-25 pts below SOTA real-valued CNNs
    (0.92-0.94 high-SNR, 0.60-0.65 overall). The earlier "76.2%" was a TRAIN
    accuracy on the synthetic gen_radioml set, not a held-out RML2016.10a number.
"""
import sys, struct, pickle
import numpy as np

MAGIC = 0x524D4C31   # 'RML1'
L = 128


def main():
    pkl = sys.argv[1] if len(sys.argv) > 1 else "RML2016.10a_dict.pkl"
    out = sys.argv[2] if len(sys.argv) > 2 else "."
    d = pickle.load(open(pkl, "rb"), encoding="latin1")
    mods = sorted({k[0] for k in d})
    snrs = sorted({k[1] for k in d})
    m2l = {m: i for i, m in enumerate(mods)}
    rng = np.random.default_rng(0)

    def recs_of(arr):
        out = []
        for x in arr:                       # x: (2,128) -> I,Q
            I = x[0].astype(np.float32); Q = x[1].astype(np.float32)
            p = float(np.sqrt((I * I + Q * Q).mean())) + 1e-9
            iq = np.empty(2 * L, np.float32); iq[0::2] = I / p; iq[1::2] = Q / p
            out.append(iq)
        return out

    train = []
    test_by_snr = {s: [] for s in snrs}
    for (mod, snr), arr in d.items():
        arr = np.asarray(arr)
        idx = rng.permutation(len(arr))
        r = recs_of(arr[idx]); lab = m2l[mod]
        for iq in r[:len(r) // 2]:
            train.append((lab, iq))
        for iq in r[len(r) // 2:]:
            test_by_snr[snr].append((lab, iq))

    def write_bin(path, recs, C):
        with open(path, "wb") as f:
            f.write(struct.pack("<iiii", MAGIC, len(recs), L, C))
            for lab, iq in recs:
                f.write(struct.pack("<i", int(lab)))
                f.write(iq.tobytes())
        print("wrote %d -> %s" % (len(recs), path))

    C = len(mods)
    order = rng.permutation(len(train))
    write_bin("%s/rml10a_train.bin" % out, [train[i] for i in order], C)
    alltest = []
    for s in snrs:
        write_bin("%s/rml10a_test_snr%d.bin" % (out, s), test_by_snr[s], C)
        alltest += test_by_snr[s]
    order = rng.permutation(len(alltest))
    write_bin("%s/rml10a_test.bin" % out, [alltest[i] for i in order], C)
    print("labels:", m2l)


if __name__ == "__main__":
    main()

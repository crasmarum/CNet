#!/bin/bash
# PG-19 from the DeepMind GCS bucket: test (100 books), validation (50), and the first $1 train books (file-list order).
set -u; NTRAIN=${1:-1200}; cd ~; mkdir -p pg19/test pg19/val pg19/train
B=https://storage.googleapis.com/deepmind-gutenberg
for s in test validation train; do curl -sL -o pg19/${s}_files.txt "https://huggingface.co/datasets/deepmind/pg19/resolve/main/data/${s}_files.txt"; done
wc -l pg19/*_files.txt
cat pg19/test_files.txt | xargs -P 16 -I{} sh -c 'curl -sL -o ~/pg19/{} "'$B'/{}"'
sed "s#validation/#val/#" pg19/validation_files.txt > /dev/null
cat pg19/validation_files.txt | xargs -P 16 -I{} sh -c 'f={}; curl -sL -o ~/pg19/val/$(basename $f) "'$B'/$f"'
head -n $NTRAIN pg19/train_files.txt | xargs -P 32 -I{} sh -c 'curl -sL -o ~/pg19/{} "'$B'/{}"'
ls pg19/test | wc -l; ls pg19/val | wc -l; ls pg19/train | wc -l
# concatenate in list order so splits are reproducible
(for f in $(cat pg19/test_files.txt); do cat pg19/$f; echo; done) > pg19/pg19_test.txt
(for f in $(cat pg19/validation_files.txt); do cat pg19/val/$(basename $f); echo; done) > pg19/pg19_val.txt
(for f in $(head -n $NTRAIN pg19/train_files.txt); do cat pg19/$f; echo; done) > pg19/pg19_train.txt
wc -c pg19/pg19_test.txt pg19/pg19_val.txt pg19/pg19_train.txt
echo PG19_FETCH_DONE

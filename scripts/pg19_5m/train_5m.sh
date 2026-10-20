#!/bin/bash
# Constant-state PG-19 byte models for the 5M-byte-window evaluation: no positional table (RoPE only), GLA first,
# then linear Born + fixed decay. Same recipe as the 2026-10-18 twins otherwise. Usage: train_5m.sh <batch> <steps>
B=${1:-8}; S=${2:-15000}; cd ~; mkdir -p ~/runs ~/models
C="-data $HOME/pg19/pg19_train.txt -emb 256 -heads 4 -tokens 4096 -blocks 6 -batch $B -steps $S -rope true -pos_table false -token_norm true -tw_cublas true -fft_fast true -born_chunk 512 -ce_eps 0.001 -grad_clip 0.5 -seed 1000"
./dp_shakespeare $C -gla true -save_path ~/models/pg19_5m_gla.mod > ~/runs/pg19_5m_gla.log 2>&1
echo "gla exit $? $(grep -a 'final val' ~/runs/pg19_5m_gla.log | tail -1)" >> ~/runs/status
./dp_shakespeare $C -born true -born_linear true -born_decay true -save_path ~/models/pg19_5m_born_decay.mod > ~/runs/pg19_5m_born_decay.log 2>&1
echo "born_decay exit $? $(grep -a 'final val' ~/runs/pg19_5m_born_decay.log | tail -1)" >> ~/runs/status
echo TRAIN5M_DONE >> ~/runs/status

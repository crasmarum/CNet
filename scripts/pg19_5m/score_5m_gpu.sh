#!/bin/bash
cd ~
/usr/bin/time -f "test pass real %e s" ./dp_shakespeare -tw_cublas true -fft_fast true -eval_stream ~/pg19/pg19_test.txt -model ~/models/pg19_5m_gla.mod -eval_window 5242880 -eval_batch 8 > ~/runs/es_test_5m_final2.log 2>&1
./dp_shakespeare -tw_cublas true -fft_fast true -eval_stream ~/pg19/pg19_val.txt -model ~/models/pg19_5m_gla.mod -eval_window 5242880 -eval_batch 4 > ~/runs/es_val_5m_final.log 2>&1
./dp_shakespeare -tw_cublas true -fft_fast true -eval_stream ~/pg19/pg19_val.txt -model ~/models/pg19_5m_gla.mod -eval_window 4096 -eval_batch 8 > ~/runs/es_val_4k_final.log 2>&1
echo ES_FINAL_DONE >> ~/runs/status
C="-data $HOME/pg19/pg19_train.txt -emb 256 -heads 4 -tokens 4096 -blocks 6 -batch 8 -steps 15000 -rope true -pos_table false -token_norm true -tw_cublas true -fft_fast true -born_chunk 512 -ce_eps 0.001 -grad_clip 0.5 -seed 1000"
./dp_shakespeare $C -born true -born_linear true -born_decay true -save_path ~/models/pg19_5m_born_decay.mod > ~/runs/pg19_5m_born_decay.log 2>&1
echo "born_decay exit $? $(grep -a 'final val' ~/runs/pg19_5m_born_decay.log | tail -1)" >> ~/runs/status
echo TRAIN5M_DONE >> ~/runs/status

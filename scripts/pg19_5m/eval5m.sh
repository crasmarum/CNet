#!/bin/bash
# Stream a text file through a constant-state CNet model in 5,242,880-byte windows (MBLM protocol), sharded by
# whole windows across P single-thread processes. Usage: eval5m.sh <model> <file> <tag> [P=56] [W=5242880]
M=$1; F=$2; TAG=$3; P=${4:-8}; W=${5:-5242880}
SZ=$(stat -c %s "$F"); NW=$(( (SZ + W - 1) / W )); OUT=~/eval5m_$TAG; mkdir -p $OUT; rm -f $OUT/*.txt
[ "$P" -gt "$NW" ] && P=$NW; T=$(( $(nproc) / P )); [ "$T" -lt 1 ] && T=1; [ "$T" -gt 16 ] && T=16   # threads per process
echo "file $F bytes $SZ windows $NW procs $P threads/proc $T window $W" > $OUT/info.txt
for ((k=0; k<NW; k++)); do
  while [ "$(jobs -rp | wc -l)" -ge "$P" ]; do sleep 5; done
  OMP_NUM_THREADS=$T ~/generate -model $M -eval $F -eval_offset $((k*W)) -eval_chars $W -eval_window $W -threads $T 2>&1 | grep -a "^EVAL" > $OUT/w$k.txt &
done
wait
cat $OUT/w*.txt | awk '{for(i=1;i<=NF;i++){split($i,a,"="); v[a[1]]=a[2]} b+=v["bytes"]; s+=v["scored"]; k+=v["skipped"]; n+=v["sum_nll"]} END{printf "WINDOWS %d bytes %d scored %d skipped %d  bits/char(scored)=%.4f  BPB_all=%.4f\n", NR, b, s, k, n/s/log(2), (n/log(2)+8*k)/b}' | tee $OUT/summary.txt
echo EVAL5M_DONE >> $OUT/summary.txt

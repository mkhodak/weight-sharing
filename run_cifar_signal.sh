#!/bin/bash

for i in {0..5}
do
    echo $((3125*2**i))
    python hyper.py cifar --features $((3125*2**i)) --full_analysis --search 32 | tee -a results/cifar_signal_$((3125*2**i))_WS.log
    
    python hyper.py cifar --features $((3125*2**i)) --full_analysis --search 32 --sweep | tee -a results/cifar_signal_$((3125*2**i))_ST.log
done

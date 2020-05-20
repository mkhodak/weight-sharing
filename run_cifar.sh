#!/bin/bash

mkdir results

for i in {0..9}
do

python hyper.py cifar --features 100_000 --search 256 --keep 0.5 --seed $i | tee -a results/cifar_$i.log

python hyper.py cifar --features 100_000 --sha_weights --search 256 --keep 0.5 --growth 1.41421356 --seed $i | tee -a results/cifar_fastWS_$i.log

python hyperband.py cifar --min_features 1_000 --features 100_000 --eta 4 --s_run 3 --search 256 --seed $i | tee -a results/cifar_SHA_$i.log

python hyperband.py cifar --min_features 1_000 --features 100_000 --eta 4 --s_run 0 --search 256 --seed $i | tee -a results/cifar_RND_$i.log

done
#!/bin/bash

for i in {0..9}
do

python hyper.py imdb --svm --features 16_777_216 --search 256 --keep 0.5 --seed $i | tee -a results/imdb_$i.log

python hyper.py imdb --svm --features 16_777_216 --sha_weights --search 256 --keep 0.5 --growth 1.41421356 --seed $i | tee -a results/imdb_fastWS_$i.log

python hyperband.py imdb --svm --min_features 65_536 --features 16_777_216 --eta 2 --s_run 3 --search 256 --seed $i | tee -a results/imdb_SHA_$i.log

python hyperband.py imdb --svm --min_features 65_536 --features 16_777_216 --eta 2 --s_run 0 --search 256 --seed $i | tee -a results/imdb_RND_$i.log

done

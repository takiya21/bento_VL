#/bin/bash
function terminate() {
  exit
}
trap 'terminate' {1,2,3,15}

num_epochs=("15")
in_w=("640")
lr=("0.0001" "0.00001")
weight_decay=("0.001" "0.0001")
optim=("Adam")
seed=("0")
bottle=("32" "64" "128" "256" "512")
for i in ${num_epochs[@]}
do
  for j in ${in_w[@]}
  do
    for k in ${lr[@]}
    do
      for l in ${weight_decay[@]}
      do
        for m in ${optim[@]}
        do
            for n in ${bottle[@]}
            do
                python closs_valid_train.py --num_epochs $i --in_w $j --lr $k --weight_decay $l --optim $m --seed 0 --bottle $n
            done
        done
      done
    done
  done
done

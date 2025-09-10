#!/bin/sh
env="Planning"
scenario="tracking"
model='F16'
algo="ppo"
exp="v1"
seed=5
device="cuda:0"
controller_type="pid"  # 可选: "ppo" 或 "pid"

echo "env is ${env}, scenario is ${scenario}, model is ${model}, algo is ${algo}, exp is ${exp}, seed is ${seed}, controller_type is ${controller_type}"
python train/train_F16sim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --model-name ${model} --experiment-name ${exp} \
    --seed ${seed} --device ${device} --n-training-threads 1 --n-rollout-threads 10000 --cuda \
    --log-interval 1 --save-interval 10 \
    --num-mini-batch 5 --buffer-size 100 --num-env-steps 3e8 \
    --lr 3e-4 --gamma 0.99 --ppo-epoch 16 --clip-params 0.2 --max-grad-norm 2 --entropy-coef 1e-3 \
    --hidden-size "128 128" --act-hidden-size "128 128" --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 8 \
    --controller-type ${controller_type}
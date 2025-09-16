#!/usr/bin/env bash
# ==============================================================================
# File: train_tracking_gps2_pid.sh
# Author: wdblink
# Description:
#   测试在 gps2 通道启用的前提下，训练 Planning/Tracking 任务，低层控制使用 PID。
#   本脚本将自动激活 conda 环境 RL-Ardupilot 并调用训练入口脚本。
#
# Usage:
#   bash scripts/train_tracking_gps2_pid.sh
#   或赋予可执行权限后直接运行：
#   chmod +x scripts/train_tracking_gps2_pid.sh && scripts/train_tracking_gps2_pid.sh
#
# Notes:
#   - tracking.yaml 已默认启用 gps2_enabled: true，无需额外开关。
#   - controller_type 设置为 pid，用于底层控制；高层策略仍使用 PPO 进行训练。
#   - 为快速测试，已将 rollout 线程数与步数设为相对较小的值，可按需调整。
# ==============================================================================
# ========== 任务与算法参数（可按需修改） ==========
env="Planning"
scenario="tracking"   # 将使用 envs/configs/tracking.yaml（其中 gps2_enabled: true）
model="F16"
algo="ppo"              # 高层策略算法
exp="gps2_pid_test"    # 实验名，用于 runs 目录区分
seed=1
device="cuda:0"        # 若无需 GPU，可改为 cpu
controller_type="pid"  # 低层控制器：pid

# 计算资源与训练规模（快速测试配置）
n_train_threads=1
n_rollout_threads=64
num_env_steps=1e6      # 测试可改为 1e5/5e5，正式训练可更大
buffer_size=64
num_mini_batch=2
ppo_epoch=4
clip_params=0.2
lr=3e-4
gamma=0.99
entropy_coef=1e-3
hidden_size="128 128"
act_hidden_size="128 128"
recurrent_hidden_size=128
recurrent_hidden_layers=1
data_chunk_length=8

# ========== 启动训练 ==========
echo "[INFO] env=${env}, scenario=${scenario}, model=${model}, algo=${algo}, exp=${exp}, controller_type=${controller_type}, seed=${seed}, device=${device}"
python train/train_F16sim.py \
  --env-name "${env}" --algorithm-name "${algo}" --scenario-name "${scenario}" --model-name "${model}" --experiment-name "${exp}" \
  --seed ${seed} --device ${device} --n-training-threads ${n_train_threads} --n-rollout-threads ${n_rollout_threads} --cuda \
  --log-interval 1 --save-interval 50 \
  --num-mini-batch ${num_mini_batch} --buffer-size ${buffer_size} --num-env-steps ${num_env_steps} \
  --lr ${lr} --gamma ${gamma} --ppo-epoch ${ppo_epoch} --clip-params ${clip_params} --max-grad-norm 2 --entropy-coef ${entropy_coef} \
  --hidden-size "${hidden_size}" --act-hidden-size "${act_hidden_size}" --recurrent-hidden-size ${recurrent_hidden_size} --recurrent-hidden-layers ${recurrent_hidden_layers} --data-chunk-length ${data_chunk_length} \
  --controller-type "${controller_type}" --use-nav-loss

echo "[INFO] 训练进程已启动，日志与权重将保存在 scripts/runs/ 目录下。"
#!/usr/bin/env python3
"""
文件: smoke_test_tracking.py
作者: wdblink

说明:
    本脚本用于对 Tracking 场景下的 gps2（光学定位）测量在同一步内的一致性进行冒烟测试（smoke test）。
    具体校验逻辑:
    - 在一次 env.step 调用内，TrackingTask.get_obs 中追加的三维 gps2 偏差(obs[22:25], 单位 km)
      应与 info['gps2_optical']['enu_m']（单位 m）减去目标(TrackingTask.target_*)并进行单位换算后一致。
    - 由于 tracking.yaml 中 noise_scale=0.0，观测不应引入额外高斯噪声，故应严格一致（数值容差 1e-6 km）。

用法:
    python scripts/smoke_test_tracking.py --steps 5 --device cpu
"""
from __future__ import annotations
import argparse
import os
import sys
from typing import Tuple

import torch

# 将项目根目录加入搜索路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(PROJECT_ROOT)

from envs.planning_env import PlanningEnv  # noqa: E402


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        argparse.Namespace: 命令行参数对象，包含 steps 和 device。
    """
    parser = argparse.ArgumentParser(description="Smoke test for Tracking gps2/obs consistency")
    parser.add_argument("--steps", type=int, default=5, help="执行的步数")
    parser.add_argument("--device", type=str, default="cpu", help="设备: cpu 或 cuda:0 等")
    return parser.parse_args()


def compute_gps2_delta_km(env: PlanningEnv, info: dict, idx: int = 0) -> Tuple[float, float, float]:
    """基于 info 和 task 目标，计算 gps2 测量相对目标的偏差（单位 km）。

    Args:
        env (PlanningEnv): 规划环境实例。
        info (dict): env.step 返回的 info 字典。
        idx (int): 指定的智能体索引（默认 0）。

    Returns:
        Tuple[float, float, float]: (dn_km, de_km, du_km)
    """
    gps2 = info["gps2_optical"]
    meas_enu_m = gps2["enu_m"][idx]  # Tensor[3]: [N,E,U] (m)

    # 目标（feet->m）
    tgt_n_m = env.task.target_npos[idx].item() * 0.3048
    tgt_e_m = env.task.target_epos[idx].item() * 0.3048
    tgt_u_m = env.task.target_altitude[idx].item() * 0.3048

    dn_km = (meas_enu_m[0].item() - tgt_n_m) / 1000.0
    de_km = (meas_enu_m[1].item() - tgt_e_m) / 1000.0
    du_km = (meas_enu_m[2].item() - tgt_u_m) / 1000.0
    return dn_km, de_km, du_km


def main() -> None:
    """程序入口：构建环境，执行若干步并对比 obs 与 info 中 gps2 偏差的一致性。"""
    args = parse_args()
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    env = PlanningEnv(
        num_envs=1,
        config="tracking",
        model="F16",
        random_seed=123,
        device=args.device,
        controller_type="pid",  # 使用 PID 底层，便于稳定复现
    )

    obs = env.reset()
    print(f"[INFO] 环境初始化完成: num_actions={env.num_actions}, num_obs={env.num_observation}, device={args.device}")

    for t in range(args.steps):
        # 高层动作: 前三维为 0（保持），第 4 维为融合门控，设为 1.0
        action = torch.zeros((env.n, env.num_actions), device=env.device)
        action[:, -1] = 1.0

        obs, reward, done, bad_done, exceed_time_limit, info = env.step(action)
        obs_cpu = obs.detach().cpu()

        # 取 obs 的 gps2 三维（索引 22,23,24，单位 km）
        obs_dn_km = obs_cpu[0, 22].item()
        obs_de_km = obs_cpu[0, 23].item()
        obs_du_km = obs_cpu[0, 24].item()

        # 由 info/gps2 与 task 目标重建期望值
        exp_dn_km, exp_de_km, exp_du_km = compute_gps2_delta_km(env, info, idx=0)

        # 打印对比
        print(f"[STEP {t}] obs_gps2_km = (N={obs_dn_km:+.6f}, E={obs_de_km:+.6f}, U={obs_du_km:+.6f}) | "
              f"exp_km = (N={exp_dn_km:+.6f}, E={exp_de_km:+.6f}, U={exp_du_km:+.6f})")

        # 数值一致性断言（单位 km），容差 1e-6 km
        tol = 1e-6
        assert abs(obs_dn_km - exp_dn_km) < tol, "gps2 N 分量不一致"
        assert abs(obs_de_km - exp_de_km) < tol, "gps2 E 分量不一致"
        assert abs(obs_du_km - exp_du_km) < tol, "gps2 U 分量不一致"

    print("[PASS] gps2 测量在同一步的 obs 与 info 保持一致。")


if __name__ == "__main__":
    main()
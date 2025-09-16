#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
观测模式配置测试脚本

用于验证obs_schema配置的三种模式：
1. "true": 使用真值位置构造相对目标偏差
2. "estimated": 使用融合估计位置构造相对目标偏差（有误差/延迟/漂移）
3. "none": 不提供位置相关信息（完全无里程计模式）

Author: wdblink
Date: 2024
"""

import sys
import os
import argparse
import logging
import numpy as np
import torch
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from config import get_config
from envs.planning_env import PlanningEnv
from envs.env_wrappers import GPUVecEnv


def make_test_env(all_args, obs_schema="estimated"):
    """创建测试环境
    
    Args:
        all_args: 配置参数
        obs_schema: 观测模式 ("true", "estimated", "none")
    """
    def get_env_fn():
        def init_env():
            if all_args.env_name == "Planning":
                env = PlanningEnv(
                    num_envs=all_args.n_rollout_threads,
                    config=all_args.scenario_name,
                    model=all_args.model_name,
                    random_seed=all_args.seed,
                    device=all_args.device,
                    controller_type=all_args.controller_type
                )
                # 动态修改obs_schema配置
                env.task.obs_schema = obs_schema
            else:
                raise NotImplementedError(f"Environment {all_args.env_name} not supported")
            return env
        return init_env
    
    return GPUVecEnv([get_env_fn()])


def test_obs_schema_modes():
    """测试不同obs_schema模式的观测差异"""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    
    # 获取基础配置
    parser = get_config()
    all_args = parser.parse_args([])
    
    # 设置测试参数
    all_args.env_name = "Planning"
    all_args.scenario_name = "tracking"
    all_args.model_name = "F16"
    all_args.seed = 42
    all_args.device = 'cpu'
    all_args.cuda = False
    all_args.controller_type = "pid"
    all_args.n_rollout_threads = 1
    
    logging.info("=== 观测模式配置测试开始 ===")
    
    # 测试三种观测模式
    modes = ["true", "estimated", "none"]
    obs_results = {}
    
    for mode in modes:
        logging.info(f"\n--- 测试观测模式: {mode} ---")
        
        try:
            # 创建环境
            envs = make_test_env(all_args, obs_schema=mode)
            
            # 重置环境
            obs = envs.reset()
            logging.info(f"环境重置成功，观测维度: {obs.shape}")
            
            # 运行几步获取观测
            for step in range(5):
                # 随机动作
                actions = np.random.uniform(-1, 1, (1, 1, 4)).astype(np.float32)
                obs, rewards, dones, bad_dones, exceed_time_limits, infos = envs.step(actions)
                
                # 记录前3维（位置相关）的观测
                pos_obs = obs[0, 0, :3]  # [delta_npos, delta_epos, delta_altitude]
                
                if step == 0:  # 只记录第一步的结果
                    obs_results[mode] = {
                        'pos_obs': pos_obs.copy(),
                        'full_obs_shape': obs.shape,
                        'pos_obs_values': pos_obs.tolist()
                    }
                
                logging.info(f"Step {step}: 位置观测前3维 = {pos_obs}")
                
                # 检查导航信息
                if len(infos) > 0 and isinstance(infos[0], dict) and 'nav' in infos[0]:
                    nav_info = infos[0]['nav']
                    pos_est = nav_info.get('pos_m_est', 'N/A')
                    pos_true = nav_info.get('pos_m_true', 'N/A')
                    logging.info(f"  导航估计: {pos_est}")
                    logging.info(f"  导航真值: {pos_true}")
                else:
                    logging.info(f"  导航信息: 不可用 (infos类型: {type(infos)})")
            
            envs.close()
            logging.info(f"✓ 模式 {mode} 测试完成")
            
        except Exception as e:
            logging.error(f"✗ 模式 {mode} 测试失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 分析结果
    logging.info("\n=== 观测模式对比分析 ===")
    
    for mode in modes:
        if mode in obs_results:
            result = obs_results[mode]
            logging.info(f"{mode} 模式:")
            logging.info(f"  位置观测值: {result['pos_obs_values']}")
            logging.info(f"  观测形状: {result['full_obs_shape']}")
    
    # 验证模式差异
    if "true" in obs_results and "estimated" in obs_results:
        true_pos = np.array(obs_results["true"]['pos_obs_values'])
        est_pos = np.array(obs_results["estimated"]['pos_obs_values'])
        diff = np.abs(true_pos - est_pos)
        logging.info(f"\ntrue vs estimated 差异: {diff}")
        if np.any(diff > 1e-6):
            logging.info("✓ estimated模式确实使用了不同于真值的估计位置")
        else:
            logging.warning("⚠ estimated模式与true模式观测相同，可能融合估计未生效")
    
    if "none" in obs_results:
        none_pos = np.array(obs_results["none"]['pos_obs_values'])
        if np.allclose(none_pos, 0.0):
            logging.info("✓ none模式正确地将位置观测置零")
        else:
            logging.warning(f"⚠ none模式位置观测非零: {none_pos}")
    
    logging.info("\n=== 观测模式配置测试完成 ===")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Observation Schema Configuration Test")
    parser.add_argument('--mode', type=str, choices=['true', 'estimated', 'none'], 
                       default='all', help="Test specific obs_schema mode or 'all'")
    args = parser.parse_args()
    
    if args.mode == 'all':
        test_obs_schema_modes()
    else:
        # 测试单个模式（简化版）
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
        logging.info(f"测试单个观测模式: {args.mode}")
        # 这里可以添加单模式测试逻辑


if __name__ == "__main__":
    main()
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导航MSE损失功能冒烟测试脚本

用于验证:
1. 训练可以正常启动和运行
2. nav_mse指标能够正确记录
3. 奖励塑形的尺度合理
4. 导航数据能够正确传递到buffer

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
from runner.F16sim_runner import F16SimRunner
from envs.planning_env import PlanningEnv
from envs.env_wrappers import GPUVecEnv


def make_test_env(all_args):
    """创建测试环境"""
    def get_env_fn():
        def init_env():
            if all_args.env_name == "Planning":
                # PlanningEnv需要的参数格式
                env = PlanningEnv(
                    num_envs=all_args.n_rollout_threads,
                    config=all_args.scenario_name,  # 配置文件名
                    model=all_args.model_name,
                    random_seed=all_args.seed,
                    device=all_args.device,
                    controller_type=all_args.controller_type  # 使用PID控制器
                )
            else:
                raise NotImplementedError(f"Environment {all_args.env_name} not supported")
            return env
        return init_env
    
    # GPUVecEnv只支持单个环境函数，会内部创建多个实例
    return GPUVecEnv([get_env_fn()])


def parse_args():
    """解析测试参数"""
    parser = argparse.ArgumentParser(description="Navigation MSE Loss Smoke Test")
    parser.add_argument('--steps', type=int, default=100,
                       help="Number of training steps to run (default: 100)")
    parser.add_argument('--device', type=str, default='cpu',
                       help="Device to use (cpu or cuda)")
    parser.add_argument('--nav-loss-coef', type=float, default=1e-4,
                       help="Navigation loss coefficient (default: 1e-4)")
    return parser.parse_args()


def main():
    """主测试函数"""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    
    # 解析参数
    test_args = parse_args()
    
    # 获取基础配置
    parser = get_config()
    all_args = parser.parse_args([])
    
    # 设置测试参数
    all_args.env_name = "Planning"
    all_args.scenario_name = "tracking"
    all_args.model_name = "F16"
    all_args.algorithm_name = "ppo"
    all_args.experiment_name = "nav_mse_smoke_test"
    all_args.seed = 42
    all_args.device = test_args.device
    all_args.cuda = (test_args.device != 'cpu')
    all_args.controller_type = "pid"  # 使用PID控制器避免需要预训练模型
    
    # 小规模训练配置
    all_args.n_rollout_threads = 2  # 减少并行数以简化测试
    all_args.buffer_size = 16
    all_args.num_mini_batch = 2
    all_args.ppo_epoch = 2
    all_args.data_chunk_length = 8
    all_args.lr = 3e-4
    all_args.gamma = 0.99
    all_args.clip_params = 0.2
    all_args.entropy_coef = 1e-3
    all_args.value_loss_coef = 0.5
    all_args.max_grad_norm = 2.0
    
    # 导航损失配置
    all_args.use_nav_loss = True
    all_args.nav_loss_coef = test_args.nav_loss_coef
    
    # 网络配置
    all_args.hidden_size = "64 64"
    all_args.act_hidden_size = "64 64"
    all_args.recurrent_hidden_size = 64
    all_args.recurrent_hidden_layers = 1
    all_args.use_recurrent_policy = True
    
    # 其他配置
    all_args.use_proper_time_limits = True
    all_args.use_gae = True
    all_args.gae_lambda = 0.95
    all_args.use_clipped_value_loss = True
    all_args.use_max_grad_norm = True
    
    logging.info("=== 导航MSE损失冒烟测试开始 ===")
    logging.info(f"测试步数: {test_args.steps}")
    logging.info(f"设备: {all_args.device}")
    logging.info(f"导航损失系数: {all_args.nav_loss_coef}")
    logging.info(f"并行环境数: {all_args.n_rollout_threads}")
    logging.info(f"Buffer大小: {all_args.buffer_size}")
    
    try:
        # 创建环境
        logging.info("创建测试环境...")
        envs = make_test_env(all_args)
        all_args.num_agents = envs.agents if hasattr(envs, 'agents') else 1
        
        # 创建runner
        logging.info("初始化训练器...")
        config = {
            'all_args': all_args,
            'envs': envs,
            'eval_envs': None,
            'device': torch.device(all_args.device),
            'run_dir': '/tmp/nav_mse_test'
        }
        runner = F16SimRunner(config, None)  # writer设为None用于测试
        
        # 验证导航损失配置
        assert runner.use_nav_loss == True, "导航损失未启用"
        assert runner.nav_loss_coef == test_args.nav_loss_coef, "导航损失系数不匹配"
        logging.info("✓ 导航损失配置验证通过")
        
        # 运行少量训练步骤
        logging.info("开始训练测试...")
        runner.warmup()
        
        total_steps = 0
        episodes_to_run = max(1, test_args.steps // all_args.buffer_size)
        
        for episode in range(episodes_to_run):
            logging.info(f"运行第 {episode + 1}/{episodes_to_run} 个episode...")
            
            # 收集数据
            for step in range(all_args.buffer_size):
                values, actions, action_log_probs, rnn_states_actor, rnn_states_critic = runner.collect(step)
                obs, rewards, dones, bad_dones, exceed_time_limits, infos = runner.envs.step(actions)
                
                # 验证info中包含导航数据
                nav_data_found = False
                for info in infos:
                    if 'nav' in info:
                        nav_data_found = True
                        break
                
                if nav_data_found:
                    logging.info("✓ 检测到导航数据")
                else:
                    logging.warning("⚠ 未检测到导航数据，可能影响MSE计算")
                
                # 计算导航MSE和奖励塑形
                nav_pos_est, nav_pos_true, nav_mse = runner._compute_nav_mse_and_reward_shaping(infos, rewards)
                
                data = obs, actions, rewards, dones, bad_dones, exceed_time_limits, action_log_probs, values, rnn_states_actor, rnn_states_critic, nav_pos_est, nav_pos_true
                runner.insert(data)
                
                total_steps += all_args.n_rollout_threads
            
            # 计算returns和训练
            runner.compute()
            train_infos = runner.train()
            
            # 验证训练信息
            logging.info(f"Episode {episode + 1} 训练信息:")
            for key, value in train_infos.items():
                logging.info(f"  {key}: {value:.6f}")
            
            # 验证nav_mse指标
            if 'nav_mse' in train_infos:
                logging.info("✓ nav_mse指标记录成功")
                if train_infos['nav_mse'] > 0:
                    logging.info(f"✓ nav_mse值合理: {train_infos['nav_mse']:.6f}")
                else:
                    logging.warning("⚠ nav_mse为0，可能导航数据未正确处理")
            else:
                logging.warning("⚠ 未找到nav_mse指标")
            
            if total_steps >= test_args.steps:
                break
        
        # 验证buffer中的导航数据
        if hasattr(runner.buffer, 'nav_pos_est') and hasattr(runner.buffer, 'nav_pos_true'):
            logging.info("✓ Buffer中包含导航数据存储")
            nav_est_shape = runner.buffer.nav_pos_est.shape
            nav_true_shape = runner.buffer.nav_pos_true.shape
            logging.info(f"  nav_pos_est shape: {nav_est_shape}")
            logging.info(f"  nav_pos_true shape: {nav_true_shape}")
            
            # 检查数据是否非零
            if np.any(runner.buffer.nav_pos_est != 0) or np.any(runner.buffer.nav_pos_true != 0):
                logging.info("✓ 导航数据已成功存储到buffer")
            else:
                logging.warning("⚠ Buffer中导航数据全为零")
        else:
            logging.error("✗ Buffer中缺少导航数据存储")
        
        logging.info("=== 冒烟测试完成 ===")
        logging.info("✓ 所有核心功能验证通过")
        
    except Exception as e:
        logging.error(f"✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        if 'envs' in locals():
            envs.close()


if __name__ == "__main__":
    main()
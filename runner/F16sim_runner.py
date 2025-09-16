import os
import sys
import time
import torch
import logging
import numpy as np
from typing import List
from pathlib import Path
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from base_runner import Runner, ReplayBuffer
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from algorithms.ppo.ppo_trainer import PPOTrainer as Trainer
from algorithms.ppo.ppo_policy import PPOPolicy as Policy
from tqdm import tqdm
# import cProfile
# import pdb
# import io
# import pstats

# profile = cProfile.Profile()

def _t2n(x):
    return x.detach().cpu().numpy()


class F16SimRunner(Runner):
    """F16仿真环境的PPO训练运行器
    
    支持导航MSE损失计算和奖励塑形功能，用于约束agent融合定位轨迹与真实轨迹的重合度。
    
    Author: wdblink
    """

    def load(self):
        self.obs_space = self.envs.observation_space
        self.act_space = self.envs.action_space
        self.num_agents = self.envs.agents

        # policy & algorithm
        self.policy = Policy(self.all_args, self.obs_space, self.act_space, device=self.device)
        self.trainer = Trainer(self.all_args, device=self.device)

        # buffer
        self.buffer = ReplayBuffer(self.all_args, self.num_agents, self.obs_space, self.act_space)
        
        # 导航损失相关配置
        self.use_nav_loss = getattr(self.all_args, 'use_nav_loss', False)
        self.nav_loss_coef = getattr(self.all_args, 'nav_loss_coef', 1e-4)
        
        # 调试：打印导航损失配置
        print(f"[初始化] use_nav_loss: {self.use_nav_loss}, nav_loss_coef: {self.nav_loss_coef}")
        
        # 导航MSE统计
        self.nav_mse_stats = []

        if self.model_dir is not None:
            self.restore()

    def run(self):
        """
        运行训练循环
        """
        print("[调试] F16SimRunner.run() 方法开始执行")
        self.warmup()

        start = time.time()
        self.total_num_steps = 0
        episodes = self.num_env_steps // self.buffer_size // self.n_rollout_threads

        for episode in tqdm(range(episodes), desc="Episode"):
            print(f"[调试] 开始第{episode}个episode，总共{episodes}个episodes")
            # global profile
            # profile.enable()
            for step in range(self.buffer_size):
                print(f"[调试] Episode {episode}, Step {step}")
                # Sample actions
                values, actions, action_log_probs, rnn_states_actor, rnn_states_critic = self.collect(step)

                # Obser reward and next obs
                obs, rewards, dones, bad_dones, exceed_time_limits, infos = self.envs.step(actions)
                # print('action:', actions)
                # print(episode, step, rewards)

                # 导航MSE计算和奖励塑形
                nav_pos_est, nav_pos_true, nav_mse = self._compute_nav_mse_and_reward_shaping(infos, rewards)

                # Extra recorded information
                # for info in infos:
                #     if 'heading_turn_counts' in info:
                #         heading_turns_list.append(info['heading_turn_counts'])

                data = obs, actions, rewards, dones, bad_dones, exceed_time_limits, action_log_probs, values, rnn_states_actor, rnn_states_critic, nav_pos_est, nav_pos_true

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            # profile.disable()
            # s = io.StringIO()
            # sortby = pstats.SortKey.CUMULATIVE
            # ps = pstats.Stats(profile, stream=s).sort_stats(sortby)
            # ps.print_stats()
            # print(s.getvalue())
            # pdb.set_trace()

            # post process
            self.total_num_steps = (episode + 1) * self.buffer_size * self.n_rollout_threads

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                logging.info("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                             .format(self.all_args.scenario_name,
                                     self.algorithm_name,
                                     self.experiment_name,
                                     episode,
                                     episodes,
                                     self.total_num_steps,
                                     self.num_env_steps,
                                     int(self.total_num_steps / (end - start))))

                train_infos["average_episode_rewards"] = self.buffer.rewards.sum() / ((self.buffer.masks[1:] == False).sum() 
                                                                                      + (self.buffer.bad_masks[1:] == False).sum())
                logging.info("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                
                # 记录导航MSE统计
                if self.nav_mse_stats:
                    nav_mse_mean = np.mean(self.nav_mse_stats)
                    nav_mse_std = np.std(self.nav_mse_stats)
                    train_infos["nav_mse_mean"] = nav_mse_mean
                    train_infos["nav_mse_std"] = nav_mse_std
                    logging.info("Navigation MSE - Mean: {:.6f}, Std: {:.6f}".format(nav_mse_mean, nav_mse_std))
                    self.nav_mse_stats.clear()  # 清空统计，准备下一个周期

                # if len(heading_turns_list):
                #     train_infos["average_heading_turns"] = np.mean(heading_turns_list)
                #     logging.info("average heading turns is {}".format(train_infos["average_heading_turns"]))
                self.log_info(train_infos, self.total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and episode != 0 and self.use_eval:
                self.eval(self.total_num_steps)

            # save model
            if (episode % self.save_interval == 0) or (episode == episodes - 1):
                self.save(episode)
                

    def warmup(self):
        # reset env
        obs = self.envs.reset()
        self.buffer.step = 0
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.policy.prep_rollout()
        values, actions, action_log_probs, rnn_states_actor, rnn_states_critic \
            = self.policy.get_actions(np.concatenate(self.buffer.obs[step]),
                                      np.concatenate(self.buffer.rnn_states_actor[step]),
                                      np.concatenate(self.buffer.rnn_states_critic[step]),
                                      np.concatenate(self.buffer.masks[step]))
        # split parallel data [N * M, shape] => [N, M, shape]
        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states_actor = np.array(np.split(_t2n(rnn_states_actor), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
    
    def _compute_nav_mse_and_reward_shaping(self, infos, rewards):
        """计算导航MSE并进行奖励塑形
        
        Args:
            infos: 环境返回的信息字典列表
            rewards: 原始奖励数组 (n_rollout_threads, num_agents, 1)
            
        Returns:
            tuple: (nav_pos_est, nav_pos_true, nav_mse_mean)
        """
        # 简化调试输出，只在第一次调用时打印
        if not hasattr(self, '_first_call_debug'):
            self._first_call_debug = True
            print(f"[调试] _compute_nav_mse_and_reward_shaping首次调用，infos类型: {type(infos)}")
            
            # 检查infos的结构
            if isinstance(infos, dict):
                print(f"[调试] infos是字典，键: {list(infos.keys())}")
            elif isinstance(infos, (list, tuple)) and len(infos) > 0:
                print(f"[调试] infos是列表/元组，长度: {len(infos)}")
                if isinstance(infos[0], dict):
                    print(f"[调试] infos[0]的键: {list(infos[0].keys())}")
                    if 'nav' in infos[0]:
                        print(f"[调试] ✅ 找到nav键")
                    else:
                        print(f"[调试] ❌ 未找到nav键")
        
        # 使用n_rollout_threads而不是len(infos)来确保形状匹配
        nav_pos_est = np.zeros((self.n_rollout_threads, self.num_agents, 3), dtype=np.float32)
        nav_pos_true = np.zeros((self.n_rollout_threads, self.num_agents, 3), dtype=np.float32)
        nav_mse_mean = 0.0
        
        if self.use_nav_loss:
            nav_mse_list = []
            
            # 处理所有n_rollout_threads个环境
            for i in range(self.n_rollout_threads):
                 # 检查是否有对应的info数据
                 # infos可能是字典或列表，需要兼容处理
                 info_data = None
                 if isinstance(infos, dict):
                     info_data = infos.get(i, {})
                 elif isinstance(infos, list) and i < len(infos):
                     info_data = infos[i]
                 
                 if info_data and isinstance(info_data, dict) and 'nav' in info_data:
                     nav_info = info_data['nav']
                     # 提取导航估计和真值 (单位：米)
                     pos_est = nav_info.get('pos_m_est', np.zeros(3))
                     pos_true = nav_info.get('pos_m_true', np.zeros(3))
                     
                     # 获取GPS2测量数据和采信度信息
                     gps2_info = info_data.get('gps2_optical', {})
                     if gps2_info:
                         gps2_pos = gps2_info.get('enu_m', np.zeros(3))
                         gps2_noise_std = gps2_info.get('noise_std_m', 0.0)
                         
                         # 计算GPS2与真值的偏差
                         if isinstance(gps2_pos, torch.Tensor):
                             gps2_pos_np = gps2_pos.cpu().numpy()
                         else:
                             gps2_pos_np = np.array(gps2_pos)
                         
                         if isinstance(pos_true, torch.Tensor):
                             pos_true_np = pos_true.cpu().numpy()
                         else:
                             pos_true_np = np.array(pos_true)
                         
                         # 计算GPS2偏差距离
                         gps2_error = np.linalg.norm(gps2_pos_np.flatten()[:3] - pos_true_np.flatten()[:3])
                        
                        # 获取agent对GPS2的真实采信度（融合门控值）
                         fuse_gate_info = info_data.get('fuse_gate', None)
                         if fuse_gate_info is not None:
                            if isinstance(fuse_gate_info, torch.Tensor):
                                agent_trust = fuse_gate_info.cpu().numpy().flatten()[0]
                            else:
                                agent_trust = float(fuse_gate_info)
                         else:
                            # 如果没有fuse_gate信息，使用噪声估算
                            agent_trust = 1.0 / (1.0 + gps2_noise_std)
                         
                         # 每10个环境打印一次，避免输出过多
                         if i % 10 == 0:
                             print(f"[GPS2监控] Env{i:2d}: GPS2偏差={gps2_error:.2f}m, 噪声std={gps2_noise_std:.2f}m, Agent采信度={agent_trust:.3f}")
                             
                             # 如果偏差较大或采信度异常，额外提醒
                             if gps2_error > 10.0:
                                 print(f"  ⚠️  GPS2偏差过大: {gps2_error:.2f}m")
                             if agent_trust < 0.3:
                                 print(f"  ⚠️  Agent对GPS2采信度较低: {agent_trust:.3f}")
                             elif agent_trust > 0.8:
                                 print(f"  ✅ Agent对GPS2采信度较高: {agent_trust:.3f}")
                 else:
                     # 如果没有对应的info或nav字段，使用默认值
                     pos_est = np.zeros(3)
                     pos_true = np.zeros(3)
                 
                 # 确保形状正确 (num_agents, 3)
                 if pos_est.ndim == 1:
                     pos_est = pos_est.reshape(1, -1)
                 if pos_true.ndim == 1:
                     pos_true = pos_true.reshape(1, -1)
                 
                 nav_pos_est[i] = pos_est[:self.num_agents]
                 nav_pos_true[i] = pos_true[:self.num_agents]
                 
                 # 计算MSE (逐分量平方误差的均值)
                 mse = np.mean((pos_est - pos_true) ** 2)
                 nav_mse_list.append(mse)
                 
                 # 奖励塑形：减去MSE损失 (只对有效的环境进行)
                 if i < len(rewards):
                     rewards[i] -= self.nav_loss_coef * mse
            
            if nav_mse_list:
                nav_mse_mean = np.mean(nav_mse_list)
                self.nav_mse_stats.extend(nav_mse_list)
        
        return nav_pos_est, nav_pos_true, nav_mse_mean

    def insert(self, data: List[np.ndarray]):
        """将数据插入buffer，支持导航信息存储
        
        Args:
            data: 包含obs, actions, rewards等的数据列表，可能包含导航数据
        """
        if len(data) == 10:  # 兼容旧格式
            obs, actions, rewards, dones, bad_dones, exceed_time_limits, action_log_probs, values, rnn_states_actor, rnn_states_critic = data
            nav_pos_est, nav_pos_true = None, None
        else:  # 新格式包含导航数据
            obs, actions, rewards, dones, bad_dones, exceed_time_limits, action_log_probs, values, rnn_states_actor, rnn_states_critic, nav_pos_est, nav_pos_true = data

        dones_env = np.any(dones.squeeze(axis=-1), axis=-1)
        bad_dones_env = np.any(bad_dones.squeeze(axis=-1), axis=-1)
        reset_env = np.any((dones + bad_dones + exceed_time_limits).squeeze(axis=-1), axis=-1)

        rnn_states_actor[reset_env == True] = np.zeros(((reset_env == True).sum(), *rnn_states_actor.shape[1:]), dtype=np.float32)
        rnn_states_critic[reset_env == True] = np.zeros(((reset_env == True).sum(), *rnn_states_critic.shape[1:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        bad_masks[bad_dones_env == True] = np.zeros(((bad_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        self.buffer.insert(obs, actions, rewards, masks, action_log_probs, values, rnn_states_actor, rnn_states_critic, bad_masks, nav_pos_est=nav_pos_est, nav_pos_true=nav_pos_true)

    @torch.no_grad()
    def eval(self, total_num_steps):
        logging.info("\nStart evaluation...")
        total_episodes, eval_episode_rewards = 0, []
        eval_cumulative_rewards = np.zeros((self.n_eval_rollout_threads, *self.buffer.rewards.shape[2:]), dtype=np.float32)

        eval_obs = self.eval_envs.reset()
        eval_masks = np.ones((self.n_eval_rollout_threads, *self.buffer.masks.shape[2:]), dtype=np.float32)
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)

        while total_episodes < self.eval_episodes:

            self.policy.prep_rollout()
            eval_actions, eval_rnn_states = self.policy.act(np.concatenate(eval_obs),
                                                            np.concatenate(eval_rnn_states),
                                                            np.concatenate(eval_masks), deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_bad_dones, eval_exceed_time_limits, eval_infos = self.eval_envs.step(eval_actions)

            eval_cumulative_rewards += eval_rewards
            eval_dones_env = np.all(eval_dones.squeeze(axis=-1), axis=-1)
            eval_reset_env = np.all((eval_dones + eval_bad_dones + eval_exceed_time_limits).squeeze(axis=-1), axis=-1)
            total_episodes += np.sum(eval_reset_env)
            eval_episode_rewards.append(eval_cumulative_rewards[eval_reset_env == True])
            eval_cumulative_rewards[eval_reset_env == True] = 0

            eval_masks = np.ones_like(eval_masks, dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), *eval_masks.shape[1:]), dtype=np.float32)
            eval_rnn_states[eval_reset_env == True] = np.zeros(((eval_reset_env == True).sum(), *eval_rnn_states.shape[1:]), dtype=np.float32)

        eval_infos = {}
        eval_infos['eval_average_episode_rewards'] = np.concatenate(eval_episode_rewards).mean(axis=1)  # shape: [num_agents, 1]
        logging.info(" eval average episode rewards: " + str(np.mean(eval_infos['eval_average_episode_rewards'])))
        self.log_info(eval_infos, total_num_steps)
        logging.info("...End evaluation")

    @torch.no_grad()
    def render(self):
        logging.info("\nStart render ...")
        self.render_opponent_index = self.all_args.render_opponent_index
        render_episode_rewards = 0
        render_obs = self.envs.reset()
        render_masks = np.ones((1, *self.buffer.masks.shape[2:]), dtype=np.float32)
        render_rnn_states = np.zeros((1, *self.buffer.rnn_states_actor.shape[2:]), dtype=np.float32)
        self.envs.render(mode='txt', filepath=f'{self.run_dir}/{self.experiment_name}.txt.acmi')
        while True:
            self.policy.prep_rollout()
            render_actions, render_rnn_states = self.policy.act(np.concatenate(render_obs),
                                                                np.concatenate(render_rnn_states),
                                                                np.concatenate(render_masks),
                                                                deterministic=True)
            render_actions = np.expand_dims(_t2n(render_actions), axis=0)
            render_rnn_states = np.expand_dims(_t2n(render_rnn_states), axis=0)
            
            # Obser reward and next obs
            render_obs, render_rewards, render_dones, render_bad_dones, render_exceed_time_limits, render_infos = self.envs.step(render_actions)
            render_episode_rewards += render_rewards
            self.envs.render(mode='txt', filepath=f'{self.run_dir}/{self.experiment_name}.txt.acmi')
            if render_dones.all():
                break
        render_infos = {}
        render_infos['render_episode_reward'] = render_episode_rewards
        logging.info("render episode reward of agent: " + str(render_infos['render_episode_reward']))

    def save(self, episode):
        save_dir = Path(str(self.save_dir) + '/episode_{}'.format(str(episode)))
        os.makedirs(str(save_dir))
        policy_actor_state_dict = self.policy.actor.state_dict()
        torch.save(policy_actor_state_dict, str(save_dir) + '/actor_latest.pt')
        policy_critic_state_dict = self.policy.critic.state_dict()
        torch.save(policy_critic_state_dict, str(save_dir) + '/critic_latest.pt')

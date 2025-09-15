import sys
import os
import gymnasium as gym
import numpy as np
import torch
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from env_base import BaseEnv
from models.F16_model import F16Model
from models.UAV_model import UAVModel
from tasks.tracking_task import TrackingTask
from utils.utils import wrap_PI
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from algorithms.ppo.ppo_actor import PPOActor
from algorithms.pid.controller import Controller

CURRENT_WORK_PATH = os.getcwd()
ego_run_dir = CURRENT_WORK_PATH + "/../scripts/runs/2024-05-26_02-14-24_Control_control_ppo_v1/episode_249"

class Args:
    def __init__(self) -> None:
        self.gain = 0.01
        self.hidden_size = '128 128'
        self.act_hidden_size = '128 128'
        self.activation_id = 1
        self.use_feature_normalization = True
        self.use_recurrent_policy = True
        self.recurrent_hidden_size = 128
        self.recurrent_hidden_layers = 1
        self.tpdv = dict(dtype=torch.float32, device=torch.device('cpu'))
        self.use_prior = False

class PlanningEnv(BaseEnv):
    """
    PlanningEnv is a fly-planning env for single agent to do tracking task.
    """
    def __init__(self, num_envs=1, config='tracking', model='F16', random_seed=None, device="cuda:0", controller_type='ppo'):
        super().__init__(num_envs, config, model, random_seed, device)
        self.low_level_action_space = gym.spaces.Box(low=-np.inf,
                                                     high=np.inf,
                                                     shape=(4, ))
        self.controller_type = controller_type
        
        if controller_type == 'ppo':
            args = Args()
            self.controller = PPOActor(args, self.observation_space, self.low_level_action_space, device=self.device)
            self.controller.eval()
            self.controller.load_state_dict(torch.load(ego_run_dir + f"/actor_latest.pt"))
            self.ego_rnn_states = torch.zeros((self.n, 1, 128), device=torch.device(device))
        elif controller_type == 'pid':
            # 初始化PID控制器
            self.controller = Controller(dt=0.02, n=self.n, device=device)
        else:
            raise ValueError(f"Unsupported controller type: {controller_type}")

    def load(self, random_seed, config, model):
        if random_seed is not None:
            self.seed(random_seed)
        if model == 'F16':
            self.model = F16Model(self.config, self.n, self.device, random_seed)
        elif model == 'UAV':
            self.model = UAVModel(self.config, self.n, self.device, random_seed)
        else:
            raise NotImplementedError
        if config == 'tracking':
            self.task = TrackingTask(self.config, self.n, self.device, random_seed)
        else:
            raise NotImplementedError
    
    def low_level_obs(self, target_pitch, target_heading, target_vt):
        """
        Convert actions into the format of observation_space of low level controller.

        observation(dim 22):
            0. ego_delta_pitch      (unit: rad)
            1. ego_delta_heading       (unit rad)
            2. ego_delta_vt            (unit: mh)
            3. ego_altitude            (unit: 5km)
            4. ego_roll_sin
            5. ego_roll_cos
            6. ego_pitch_sin
            7. ego_pitch_cos
            8. ego_vt                  (unit: mh)
            9. ego_alpha_sin
            10. ego_alpha_cos
            11. ego_beta_sin
            12. ego_beta_cos
            13. ego_P                  (unit: rad/s)
            14. ego_Q                  (unit: rad/s)
            15. ego_R                  (unit: rad/s)
            16. ego_T                  (unit: %)
            17. ego_el                 (unit: %)
            18. ego_ail                (unit: %)
            19. ego_rud                (unit: %)
            20. ego_lef                (unit: %)
            21. EAS2TAS
        """
        npos, epos, altitude = self.model.get_position()
        roll, pitch, heading = self.model.get_posture()
        vt = self.model.get_vt()
        EAS = self.model.get_EAS()
        alpha = self.model.get_AOA()
        beta = self.model.get_AOS()
        P, Q, R = self.model.get_angular_velocity()
        T = self.model.get_thrust()
        el, ail, rud, lef = self.model.get_control_surface()
        eas2tas = self.model.get_EAS2TAS()

        norm_delta_pitch = wrap_PI((pitch - target_pitch).reshape(-1, 1))
        norm_delta_heading = wrap_PI((heading - target_heading).reshape(-1, 1))
        norm_delta_vt = (vt - target_vt).reshape(-1, 1) * 0.3048 / 340
        norm_altitude = altitude.reshape(-1, 1) * 0.3048 / 5000
        roll_sin = torch.sin(roll.reshape(-1, 1))
        roll_cos = torch.cos(roll.reshape(-1, 1))
        pitch_sin = torch.sin(pitch.reshape(-1, 1))
        pitch_cos = torch.cos(pitch.reshape(-1, 1))
        # norm_vt = vt.reshape(-1, 1) * 0.3048 / 340
        norm_EAS = EAS.reshape(-1, 1) * 0.3048 / 340
        alpha_sin = torch.sin(alpha.reshape(-1, 1))
        alpha_cos = torch.cos(alpha.reshape(-1, 1))
        beta_sin = torch.sin(beta.reshape(-1, 1))
        beta_cos = torch.cos(beta.reshape(-1, 1))
        norm_P = P.reshape(-1, 1)
        norm_Q = Q.reshape(-1, 1)
        norm_R = R.reshape(-1, 1)
        norm_T = T.reshape(-1, 1) / 0.225 / 76300 * 0.3048
        norm_el = el.reshape(-1, 1) / 45
        norm_ail = ail.reshape(-1, 1) / 45
        norm_rud = rud.reshape(-1, 1) / 45
        norm_lef = lef.reshape(-1, 1) / 45
        obs = torch.hstack((norm_delta_pitch, norm_delta_heading))
        obs = torch.hstack((obs, norm_delta_vt))
        obs = torch.hstack((obs, norm_altitude))
        obs = torch.hstack((obs, roll_sin))
        obs = torch.hstack((obs, roll_cos))
        obs = torch.hstack((obs, pitch_sin))
        obs = torch.hstack((obs, pitch_cos))
        obs = torch.hstack((obs, norm_EAS))
        obs = torch.hstack((obs, alpha_sin))
        obs = torch.hstack((obs, alpha_cos))
        obs = torch.hstack((obs, beta_sin))
        obs = torch.hstack((obs, beta_cos))
        obs = torch.hstack((obs, norm_P))
        obs = torch.hstack((obs, norm_Q))
        obs = torch.hstack((obs, norm_R))
        obs = torch.hstack((obs, norm_T))
        obs = torch.hstack((obs, norm_el))
        obs = torch.hstack((obs, norm_ail))
        obs = torch.hstack((obs, norm_rud))
        obs = torch.hstack((obs, norm_lef))
        obs = torch.hstack((obs, eas2tas.reshape(-1, 1)))
        return obs
    
    def step(self, action, render=False, count=0):
        self.reset()
        action = torch.clamp(action, -1, 1)
        # set target
        roll, pitch, yaw = self.model.get_posture()
        vt = self.model.get_vt()
        target_pitch = pitch + action[:, 0] * 0.3
        target_heading = yaw + action[:, 1] * 0.3
        target_vt = vt + action[:, 2] * 30
        for i in range(50):
            # 在每个内部仿真步开始前清理 gps2 测量缓存，确保同一步内 obs 与 info 复用同一份采样
            if getattr(self, 'gps2_enabled', False):
                # 同一步缓存：首次访问时生成一次，后续访问复用
                self._gps2_clear_cache()
            # low-level control
            if self.controller_type == 'ppo':
                ego_obs = self.low_level_obs(target_pitch, target_heading, target_vt)
                masks = torch.ones((self.n, 1), device=self.device)
                with torch.no_grad():
                    ego_actions, _, self.ego_rnn_states = self.controller(ego_obs, self.ego_rnn_states, masks, deterministic=True)
            elif self.controller_type == 'pid':
                # 使用PID控制器
                # 计算速度缩放因子
                vt_current = self.model.get_vt()
                self.controller.calc_speed_scaler(vt_current)
                
                # 更新控制器目标
                _, _, altitude = self.model.get_position()
                target_altitude = altitude.reshape(-1, 1)  # 保持当前高度，确保正确的张量形状
                target_vt_reshaped = target_vt.reshape(-1, 1)  # 确保正确的张量形状
                self.controller.cal_pitch_throttle(target_altitude, target_vt_reshaped, self)
                self.controller.update_heading_hold(target_heading, self)
                self.controller.stabilize(self)
                
                # 获取控制动作
                ego_actions = self.controller.get_action()
            
            # step
            self.model.update(ego_actions)
            # 推送 gps2 光学定位缓冲样本（延迟观测使用）
            if getattr(self, 'gps2_enabled', False):
                self._gps2_push_sample()
            done = self.is_done.bool()
            bad_done = self.bad_done.bool()
            exceed_time_limit = self.exceed_time_limit.bool()
            reset = (done | bad_done) | exceed_time_limit
            self.model.s[reset] = self.model.recent_s[reset]
            # self.model.u[reset] = self.model.recent_u[reset]

            self.step_count += 1
            obs = self.obs()
            info = self.info()
            done, bad_done, exceed_time_limit, info = self.done(info)
            reward = self.reward()
            if render:
                self.render(count=count)
            count += 1
        return obs, reward, done, bad_done, exceed_time_limit, info

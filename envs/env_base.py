import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import numpy as np
import torch
import gymnasium as gym
import random
from models.model_base import BaseModel
from tasks.task_base import BaseTask
from utils.utils import parse_config, enu_to_geodetic, _t2n


class BaseEnv(gym.Env):

    def __init__(self,
                 num_envs=10,
                 config='heading',
                 model='F16',
                 random_seed=None,
                 device="cuda:0"):
        super().__init__()
        self.config = parse_config(config)
        self.num_envs = num_envs
        self.num_agents = getattr(self.config, 'num_agents', 100)
        self.n = self.num_agents * self.num_envs
        self.device = torch.device(device)

        # 新增：从配置读取地理参考原点（用于 ENU <-> 经纬高 转换）
        # 说明：
        # - origin_lat: 参考纬度（度）
        # - origin_lon: 参考经度（度）
        # - origin_alt: 参考海拔（米，ASL）
        # 注意：内部状态 s 的位置/高度单位为英尺，render 中已转换为米再传入地理转换函数。
        self.origin_lat = getattr(self.config, 'origin_lat', 0.0)
        self.origin_lon = getattr(self.config, 'origin_lon', 0.0)
        self.origin_alt = getattr(self.config, 'origin_alt', 0.0)

        self.load(random_seed, config, model)

        self.step_count = torch.zeros(self.n, dtype=torch.int64, device=self.device)
        self.is_done = torch.ones(self.n, dtype=torch.bool, device=self.device)
        self.bad_done = torch.ones(self.n, dtype=torch.bool, device=self.device)
        self.exceed_time_limit = torch.ones(self.n, dtype=torch.bool, device=self.device)
        self.create_records = False

    def seed(self, random_seed):
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

    def load(self, random_seed, config, model):
        if random_seed is not None:
            self.seed(random_seed)
        self.model = BaseModel(self.config, self.n, self.device, random_seed)
        self.task = BaseTask(self.config, self.n, self.device, random_seed)
    
    @property
    def observation_space(self):
        return self.task.observation_space

    @property
    def action_space(self):
        return self.task.action_space
    
    @property
    def num_observation(self):
        return self.task.num_observation
    
    @property
    def num_actions(self):
        return self.task.num_actions

    def obs(self):
        return self.task.get_obs(self)

    def reward(self):
        return self.task.get_reward(self)

    def done(self, info):
        done, bad_done, exceed_time_limit, info = self.task.get_termination(self, info)
        self.is_done = self.is_done + done
        self.bad_done = self.bad_done + bad_done
        self.exceed_time_limit = self.exceed_time_limit + exceed_time_limit
        return self.is_done, self.bad_done, self.exceed_time_limit, info
    
    def info(self):
        return {}

    def get_number_of_agents(self):
        return self.n

    def reset(self):
        done = self.is_done.bool()
        bad_done = self.bad_done.bool()
        exceed_time_limit = self.exceed_time_limit.bool()
        reset = (done | bad_done) | exceed_time_limit

        self.model.reset(self)
        self.task.reset(self)

        self.step_count[reset] = 0
        self.is_done[:] = 0
        self.bad_done[:] = 0
        self.exceed_time_limit[:] = 0
        obs = self.obs()
        return obs

    def step(self, action, render=False, count=0):
        self.reset()
        self.model.update(action)
        self.step_count += 1
        obs = self.obs()
        info = self.info()
        done, bad_done, exceed_time_limit, info = self.done(info)
        reward = self.reward()
        if render:
            self.render(count=count)
        return obs, reward, done, bad_done, exceed_time_limit, info
    
    def render(self, count, filename='./tracks/F16SimRecording-'):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        :param mode: str, the mode to render with
        """
        if count == 0:
            self.create_records = False
            self.filename = filename + str(count) + '.txt.acmi'
        if not self.create_records:
            with open(self.filename, mode='w', encoding='utf-8') as f:
                f.write("FileType=text/acmi/tacview\n")
                f.write("FileVersion=2.0\n")
                f.write("0,ReferenceTime=2023-04-01T00:00:00Z\n")
            self.create_records = True
        with open(self.filename, mode='a', encoding='utf-8') as f:
            timestamp = self.step_count[0] * self.model.dt
            f.write(f"#{timestamp:.2f}\n")
            for i in range(self.n):
                npos, epos, alt = self.model.get_position()
                roll, pitch, yaw = self.model.get_posture()
                # ENU 入参单位为米，这里将内部英尺制转换为米
                npos = _t2n(npos) * 0.3048
                epos = _t2n(epos) * 0.3048
                alt = _t2n(alt) * 0.3048
                roll = _t2n(roll)[0] * 180 / np.pi
                pitch = _t2n(pitch)[0] * 180 / np.pi
                yaw = _t2n(yaw)[0] * 180 / np.pi

                # 修改点：使用配置的地理原点将 ENU 转为经纬高
                # 注意：enu_to_geodetic 的入参顺序为 (xEast, yNorth, zUp, lat_ref, lon_ref, h_ref)
                # 其中 lat_ref/lon_ref 单位为度，h_ref 单位为米
                lat, lon, alt = enu_to_geodetic(
                    epos, npos, alt,
                    self.origin_lat, self.origin_lon, self.origin_alt
                )

                log_msg = f"{100 + i},T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
                log_msg += f"Name=F16,"
                log_msg += f"Color=Red"
                if log_msg is not None:
                    f.write(log_msg + "\n")
        reset = torch.any(self.bad_done + self.is_done + self.exceed_time_limit)
        if reset:
            self.create_records = False
            self.filename = filename + str(count) + '.txt.acmi'

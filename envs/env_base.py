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

        # 新增：gps2（光学定位）测量通道配置与初始化
        # 参数说明：
        # - gps2_enabled: 是否启用光学定位通道
        # - gps2_delay_s: 平均延迟（秒），默认 0.2
        # - gps2_delay_jitter_s: 延迟抖动（秒，标准差），默认 0.02
        # - gps2_noise_std_m: 正常姿态噪声标准差（米）
        # - gps2_noise_std_m_max: 超限姿态噪声标准差（米）
        # - gps2_gimbal_limit_deg: 云台姿态限位（度）
        self.gps2_enabled = getattr(self.config, 'gps2_enabled', False)
        self.gps2_delay_s = getattr(self.config, 'gps2_delay_s', 0.2)
        self.gps2_delay_jitter_s = getattr(self.config, 'gps2_delay_jitter_s', 0.02)
        self.gps2_noise_std_m = getattr(self.config, 'gps2_noise_std_m', 1.5)
        self.gps2_noise_std_m_max = getattr(self.config, 'gps2_noise_std_m_max', 8.0)
        self.gps2_gimbal_limit_deg = getattr(self.config, 'gps2_gimbal_limit_deg', 25.0)
        if self.gps2_enabled:
            self._gps2_setup()
        # 新增：gps2同一步测量缓存（确保obs与info一致）
        self._gps2_meas_cache = None  # dict或None
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
        """返回附加信息。

        新增：当启用 gps2 通道时，返回光学定位测量（含延迟与姿态相关噪声）。

        Returns:
            dict: 可能包含键 "gps2_optical"，其值为字典，含字段：
                - enu_m (Tensor[n,3]): ENU 下的位置测量（米）
                - delay_s (Tensor[n]): 本步使用的延迟（秒）
                - noise_std_m (Tensor[n]): 本步使用的噪声标准差（米）
        """
        if getattr(self, 'gps2_enabled', False):
            return {"gps2_optical": self.get_gps2_optical()}
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

        # 当启用 gps2 通道时，复位其缓冲
        if getattr(self, 'gps2_enabled', False):
            self._gps2_reset_buffers()
            # 清理gps2测量缓存
            self._gps2_clear_cache()

        self.step_count[reset] = 0
        self.is_done[:] = 0
        self.bad_done[:] = 0
        self.exceed_time_limit[:] = 0
        obs = self.obs()
        return obs

    def step(self, action, render=False, count=0):
        # 每步开始清理gps2测量缓存，确保同一步复用一次采样
        if getattr(self, 'gps2_enabled', False):
            self._gps2_clear_cache()
        self.reset()
        self.model.update(action)
        # 推送真值样本至 gps2 循环缓冲（用于延迟回放）
        if getattr(self, 'gps2_enabled', False):
            self._gps2_push_sample()
        self.step_count += 1
        obs = self.obs()
        info = self.info()
        done, bad_done, exceed_time_limit, info = self.done(info)
        reward = self.reward()
        if render:
            self.render(count=count)
        return obs, reward, done, bad_done, exceed_time_limit, info
    
    def render(self, count, filename='./tracks/F16SimRecording-'):
        if self.create_records:
            if count == 0:
                npos, epos, altitude = self.model.get_position()
                roll, pitch, heading = self.model.get_posture()
                vt = self.model.get_vt()
                el, ail, rud, lef = self.model.get_control_surface()
                T = self.model.get_thrust()
                EAS = self.model.get_EAS()
                easy2tas = self.model.get_EAS2TAS()
                npos_m = npos * 0.3048
                epos_m = epos * 0.3048
                alt_m = altitude * 0.3048
                if abs(self.origin_lat) > 1e-10 and abs(self.origin_lon) > 1e-10:
                    lat, lon, alt = enu_to_geodetic(epos_m, npos_m, alt_m, self.origin_lat, self.origin_lon, self.origin_alt)
                else:
                    lat = epos_m * 1e-5
                    lon = npos_m * 1e-5
                    alt = alt_m
                heading_deg = heading * 180 / torch.pi
                with open(self.filename, 'w+') as f:
                    f.write(f"FileType=text/acmi/tacview\nFileVersion=2.2\n0,ReferenceTime=2011-11-04T13:00:00Z\n100,0,DateTime=2011-11-04T13:00:00Z\n100,0,ReferenceLongitude={lat[0].item()},{lon[0].item()},{alt[0].item()}\n50,0|-2,Flight Recorder|t=35663\n".format(self.filename))
                    f.write("100,NewEvent=Top Gun start|0:00:00:00\n")
                    for i in range(self.n):
                        f.write("0,{}|A2A-Missile|Lock On|Coalition=Enemies|Color=Red\n".format(i+1))
            elif self.step_count[-1] == 1:
                self.filename = filename
            else:
                npos, epos, altitude = self.model.get_position()
                roll, pitch, heading = self.model.get_posture()
                vt = self.model.get_vt()
                n = len(npos)
                for i in range(n):
                    if abs(self.origin_lat) > 1e-10 and abs(self.origin_lon) > 1e-10:
                        lat, lon, alt = enu_to_geodetic(epos[i].item()*0.3048, npos[i].item()*0.3048, altitude[i].item()*0.3048, self.origin_lat, self.origin_lon, self.origin_alt)
                    else:
                        lat = epos[i].item() * 0.3048 * 1e-5
                        lon = npos[i].item() * 0.3048 * 1e-5
                        alt = altitude[i].item() * 0.3048
                    heading_deg = heading[i].item() * 180 / torch.pi
                    with open(self.filename, 'a+') as f:
                        f.write("100,{}\n100,{}|T={}km/h|LatLng={}|{}|Alt={}m|TAS={}m/s|Roll={}deg|Pitch={}deg|Hdg={}deg\n".format(self.step_count[-1].item(), i+1, vt[i].item()*1.8, lat, lon, alt, vt[i].item()*0.3048, roll[i].item()*180/torch.pi, pitch[i].item()*180/torch.pi, heading_deg))
            if (self.step_count[-1] == 2500):
                self.create_records = False
                self.filename = filename + str(count) + '.txt.acmi'

    def _gps2_setup(self):
        """初始化 gps2 循环缓冲。

        说明：
            - 使用模型 dt（若存在）估算最大延迟对应的缓冲长度；否则采用默认 dt=0.02。
            - 缓冲字段：位置 [N, E, U]（英尺）与姿态 [roll, pitch]（弧度）。
        """
        model = getattr(self, 'model', None)
        dt = float(getattr(model, 'dt', 0.02))
        max_delay = float(self.gps2_delay_s) + 3.0 * float(self.gps2_delay_jitter_s)
        buf_len = int(np.ceil(max(2.0, max_delay / dt + 10.0)))
        self._gps2_buf_len = buf_len
        self._gps2_ptr = 0
        self._gps2_pos_buf = torch.zeros((buf_len, self.n, 3), dtype=torch.float32, device=self.device)
        self._gps2_att_buf = torch.zeros((buf_len, self.n, 2), dtype=torch.float32, device=self.device)

    def _gps2_reset_buffers(self):
        """以当前样本填充 gps2 缓冲，避免冷启动缺样本。"""
        npos, epos, alt = self.model.get_position()
        roll, pitch, _ = self.model.get_posture()
        pos = torch.stack([npos, epos, alt], dim=-1)  # (n,3) 英尺
        att = torch.stack([roll, pitch], dim=-1)      # (n,2) 弧度
        self._gps2_pos_buf[:] = pos.unsqueeze(0).repeat(self._gps2_buf_len, 1, 1)
        self._gps2_att_buf[:] = att.unsqueeze(0).repeat(self._gps2_buf_len, 1, 1)
        self._gps2_ptr = 0

    def _gps2_push_sample(self):
        """将当前真值推入 gps2 循环缓冲。"""
        npos, epos, alt = self.model.get_position()
        roll, pitch, _ = self.model.get_posture()
        self._gps2_pos_buf[self._gps2_ptr, :, :] = torch.stack([npos, epos, alt], dim=-1)
        self._gps2_att_buf[self._gps2_ptr, :, :] = torch.stack([roll, pitch], dim=-1)
        self._gps2_ptr = (self._gps2_ptr + 1) % self._gps2_buf_len

    def _gps2_clear_cache(self):
        """清空本步的 gps2 测量缓存。
        
        设计意图：
            - 在每一步开始时调用，保证当步内首次请求gps2测量时进行一次采样；
            - 当步内后续请求（如get_obs与info）复用同一份采样，确保噪声与延迟一致。
        """
        self._gps2_meas_cache = None

    def get_gps2_optical(self):
        """生成光学定位（gps2）测量：延迟回放 + 姿态相关噪声。

        延迟：均值 gps2_delay_s，抖动 gps2_delay_jitter_s（高斯），按 dt 转为步数索引历史样本。
        噪声：当 max(|roll|,|pitch|) 超过 gps2_gimbal_limit_deg 时，使用更大噪声标准差；单位为米。

        同一步缓存：
            - 若本步已生成过测量，则直接返回缓存，确保obs与info一致；
            - 若尚未生成，则采样一次并写入缓存。

        Returns:
            dict: 包含以下键
                - enu_m (Tensor[n,3]): ENU 下位置测量（米）
                - delay_s (Tensor[n]): 实际延迟（秒）
                - noise_std_m (Tensor[n]): 使用的噪声标准差（米）
        """
        # 优先返回缓存，确保同一步一致性
        if getattr(self, "_gps2_meas_cache", None) is not None:
            return self._gps2_meas_cache

        dt = float(getattr(getattr(self, 'model', None), 'dt', 0.02))
        base = max(0.0, float(self.gps2_delay_s))
        jitter = max(0.0, float(self.gps2_delay_jitter_s))

        if jitter > 0.0:
            delays = torch.clamp(base + torch.randn(self.n, device=self.device) * jitter, min=0.0)
        else:
            delays = torch.full((self.n,), base, device=self.device)

        steps = torch.clamp((delays / dt).round().to(torch.int64), min=0, max=self._gps2_buf_len - 1)
        idx = (self._gps2_ptr - 1 - steps) % self._gps2_buf_len
        agents = torch.arange(self.n, device=self.device)

        pos_ft = self._gps2_pos_buf[idx, agents, :]  # (n,3) 英尺 [N,E,U]
        att = self._gps2_att_buf[idx, agents, :]     # (n,2) 弧度 [roll,pitch]

        max_angle_deg = att.abs().max(dim=1).values * (180.0 / np.pi)
        std_m = torch.where(
            max_angle_deg > float(self.gps2_gimbal_limit_deg),
            torch.tensor(float(self.gps2_noise_std_m_max), device=self.device),
            torch.tensor(float(self.gps2_noise_std_m), device=self.device),
        )

        pos_m = pos_ft * 0.3048
        noise = torch.randn_like(pos_m) * std_m.view(-1, 1)
        meas_m = pos_m + noise

        # 写入缓存并返回
        self._gps2_meas_cache = {
            'enu_m': meas_m,
            'delay_s': delays,
            'noise_std_m': std_m,
        }
        return self._gps2_meas_cache

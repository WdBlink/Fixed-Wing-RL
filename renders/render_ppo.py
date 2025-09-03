import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import torch

# add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from envs.control_env import ControlEnv
from algorithms.ppo.ppo_actor import PPOActor
import logging
logging.basicConfig(level=logging.DEBUG)


class Args:
    def __init__(self, device: str) -> None:
        self.gain = 0.01
        self.hidden_size = '128 128'
        self.act_hidden_size = '128 128'
        self.activation_id = 1
        self.use_feature_normalization = True
        self.use_recurrent_policy = True
        self.recurrent_hidden_size = 128
        self.recurrent_hidden_layers = 1
        self.tpdv = dict(dtype=torch.float32, device=torch.device(device))
        self.use_prior = False


def _parse_args():
    parser = argparse.ArgumentParser(description="Render PPO policy and save flight data")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Path to a run directory (e.g., scripts/runs/<timestamp>_Control_control_F16_ppo_v1) or directly to an episode directory (scripts/runs/.../episode_<N>)")
    parser.add_argument("--config", type=str, choices=["heading", "control", "tracking"], default="control",
                        help="Environment scenario to render; must match training scenario")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for inference, e.g., 'cpu' or 'cuda:0'")
    parser.add_argument("--max-steps", type=int, default=10000,
                        help="Maximum simulation steps to run during rendering")
    parser.add_argument("--seed", type=int, default=5, help="Random seed used to init env")
    return parser.parse_args()


def _resolve_episode_dir(run_dir: str) -> str:
    """Return an episode directory that contains actor_latest.pt.
    If run_dir already contains the file, use it; otherwise pick the episode_ with the largest index.
    """
    run_path = Path(run_dir)
    # If user passed an episode folder directly
    if (run_path / "actor_latest.pt").exists():
        return str(run_path)
    # Otherwise, search for episode_* subfolders
    episode_dirs = []
    if run_path.exists() and run_path.is_dir():
        for p in run_path.iterdir():
            if p.is_dir() and p.name.startswith("episode_") and (p / "actor_latest.pt").exists():
                try:
                    idx = int(p.name.split("_")[-1])
                except ValueError:
                    idx = -1
                episode_dirs.append((idx, p))
    if not episode_dirs:
        raise FileNotFoundError(f"No episode directory with actor_latest.pt found under: {run_dir}")
    # Choose episode with largest index
    episode_dirs.sort(key=lambda x: x[0])
    return str(episode_dirs[-1][1])


def _t2n(x):
    return x.detach().cpu().numpy()


def main():
    cli = _parse_args()
    device = cli.device
    config = cli.config
    episode_dir = _resolve_episode_dir(cli.run_dir)
    os.makedirs("./result", exist_ok=True)

    # init env
    env = ControlEnv(num_envs=1, config=config, model='F16', random_seed=cli.seed, device=device)

    # build policy and load weights
    net_args = Args(device)
    policy = PPOActor(net_args, env.observation_space, env.action_space, device=torch.device(device))
    policy.eval()
    state_dict = torch.load(os.path.join(episode_dir, "actor_latest.pt"), map_location=torch.device(device))
    policy.load_state_dict(state_dict)

    logging.info("Start render")
    obs = env.reset()

    # initial measurements
    npos, epos, altitude = env.model.get_position()
    npos_buf = np.mean(_t2n(npos))
    epos_buf = np.mean(_t2n(epos))
    altitude_buf = np.mean(_t2n(altitude))

    roll, pitch, yaw = env.model.get_posture()
    roll_buf = np.mean(_t2n(roll))
    pitch_buf = np.mean(_t2n(pitch))
    yaw_buf = np.mean(_t2n(yaw))

    vt = env.model.get_vt()
    vt_buf = np.mean(_t2n(vt))

    alpha = env.model.get_AOA()
    alpha_buf = np.mean(_t2n(alpha))

    beta = env.model.get_AOS()
    beta_buf = np.mean(_t2n(beta))

    G = env.model.get_G()
    G_buf = np.mean(_t2n(G))

    T = env.model.get_thrust()
    T_buf = np.mean(_t2n(T))
    throttle_buf = np.mean(_t2n(T * 0.3048 / 82339 / 0.225))

    el, ail, rud, lef = env.model.get_control_surface()
    el_buf = np.mean(_t2n(el))
    ail_buf = np.mean(_t2n(ail))
    rud_buf = np.mean(_t2n(rud))

    if config == 'heading':
        target_altitude_buf = np.mean(_t2n(env.task.target_altitude))
        target_heading_buf = np.mean(_t2n(env.task.target_heading))
        target_vt_buf = np.mean(_t2n(env.task.target_vt))
    elif config == 'control':
        target_pitch_buf = np.mean(_t2n(env.task.target_pitch))
        target_heading_buf = np.mean(_t2n(env.task.target_heading))
        target_vt_buf = np.mean(_t2n(env.task.target_vt))
    elif config == 'tracking':
        target_npos_buf = np.mean(_t2n(env.task.target_npos))
        target_epos_buf = np.mean(_t2n(env.task.target_epos))
        target_altitude_buf = np.mean(_t2n(env.task.target_altitude))

    counts = 0
    env.render(count=counts)
    rnn_states = torch.zeros((1, 1, 128), device=torch.device(device))
    masks = torch.ones((1, 1), device=torch.device(device))
    start = time.time()
    episode_rewards = 0
    unreach_target = 0
    reset_target = 0

    while True:
        with torch.no_grad():
            actions, _, rnn_states = policy(obs, rnn_states, masks, deterministic=True)
        obs, rewards, dones, bad_dones, exceed_time_limits, infos = env.step(actions, render=True, count=counts)

        unreach_target += int(_t2n(bad_dones))
        reset_target += int(_t2n(dones))

        # record
        npos, epos, altitude = env.model.get_position()
        npos_buf = np.hstack((npos_buf, np.mean(_t2n(npos))))
        epos_buf = np.hstack((epos_buf, np.mean(_t2n(epos))))
        altitude_buf = np.hstack((altitude_buf, np.mean(_t2n(altitude))))

        roll, pitch, yaw = env.model.get_posture()
        roll_buf = np.hstack((roll_buf, np.mean(_t2n(roll))))
        pitch_buf = np.hstack((pitch_buf, np.mean(_t2n(pitch))))
        yaw_buf = np.hstack((yaw_buf, np.mean(_t2n(yaw))))

        vt = env.model.get_vt()
        vt_buf = np.hstack((vt_buf, np.mean(_t2n(vt))))

        alpha = env.model.get_AOA()
        alpha_buf = np.hstack((alpha_buf, np.mean(_t2n(alpha))))

        beta = env.model.get_AOS()
        beta_buf = np.hstack((beta_buf, np.mean(_t2n(beta))))

        G = env.model.get_G()
        G_buf = np.hstack((G_buf, np.mean(_t2n(G))))

        T = env.model.get_thrust()
        T_buf = np.hstack((T_buf, np.mean(_t2n(T))))
        throttle_buf = np.hstack((throttle_buf, np.mean(_t2n(T * 0.3048 / 82339 / 0.225))))

        el, ail, rud, lef = env.model.get_control_surface()
        el_buf = np.hstack((el_buf, np.mean(_t2n(el))))
        ail_buf = np.hstack((ail_buf, np.mean(_t2n(ail))))
        rud_buf = np.hstack((rud_buf, np.mean(_t2n(rud))))

        if config == 'heading':
            target_altitude_buf = np.hstack((target_altitude_buf, np.mean(_t2n(env.task.target_altitude))))
            target_heading_buf = np.hstack((target_heading_buf, np.mean(_t2n(env.task.target_heading))))
            target_vt_buf = np.hstack((target_vt_buf, np.mean(_t2n(env.task.target_vt))))
        elif config == 'control':
            target_pitch_buf = np.hstack((target_pitch_buf, np.mean(_t2n(env.task.target_pitch))))
            target_heading_buf = np.hstack((target_heading_buf, np.mean(_t2n(env.task.target_heading))))
            target_vt_buf = np.hstack((target_vt_buf, np.mean(_t2n(env.task.target_vt))))
        elif config == 'tracking':
            target_npos_buf = np.hstack((target_npos_buf, np.mean(_t2n(env.task.target_npos))))
            target_epos_buf = np.hstack((target_epos_buf, np.mean(_t2n(env.task.target_epos))))
            target_altitude_buf = np.hstack((target_altitude_buf, np.mean(_t2n(env.task.target_altitude))))

        counts += 1
        print(counts, _t2n(rewards))
        episode_rewards += _t2n(rewards)
        if counts >= cli.max_steps:
            break

    # save result
    np.save('./result/npos.npy', npos_buf)
    np.save('./result/epos.npy', epos_buf)
    np.save('./result/altitude.npy', altitude_buf)
    np.save('./result/roll.npy', roll_buf)
    np.save('./result/pitch.npy', pitch_buf)
    np.save('./result/yaw.npy', yaw_buf)
    np.save('./result/vt.npy', vt_buf)
    np.save('./result/alpha.npy', alpha_buf)
    np.save('./result/beta.npy', beta_buf)
    np.save('./result/G.npy', G_buf)

    np.save('./result/T.npy', T_buf)
    np.save('./result/throttle.npy', throttle_buf)
    np.save('./result/ail.npy', ail_buf)
    np.save('./result/el.npy', el_buf)
    np.save('./result/rud.npy', rud_buf)

    if config == 'heading':
        np.save('./result/target_altitude.npy', target_altitude_buf)
        np.save('./result/target_heading.npy', target_heading_buf)
        np.save('./result/target_vt.npy', target_vt_buf)
    elif config == 'control':
        np.save('./result/target_pitch.npy', target_pitch_buf)
        np.save('./result/target_heading.npy', target_heading_buf)
        np.save('./result/target_vt.npy', target_vt_buf)
    elif config == 'tracking':
        np.save('./result/target_npos.npy', target_npos_buf)
        np.save('./result/target_epos.npy', target_epos_buf)
        np.save('./result/target_altitude.npy', target_altitude_buf)

    end = time.time()
    print('total time:', end - start)
    print('episode reward:', episode_rewards)
    denom = (unreach_target + reset_target) if (unreach_target + reset_target) != 0 else 1
    print('average episode reward:', episode_rewards / denom)
    print('unreach target:', unreach_target)
    print('reset target:', reset_target)
    print('success rate:', reset_target / denom)


if __name__ == "__main__":
    main()

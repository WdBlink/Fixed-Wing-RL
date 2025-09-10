import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_if_exists(base: Path, name: str):
    p = base / f"{name}.npy"
    if p.exists():
        try:
            return np.load(str(p))
        except Exception as e:
            print(f"Warning: failed to load {p}: {e}")
    else:
        print(f"Info: {p} not found, skip plotting {name}")
    return None


def make_time_axis(ref_len: int):
    # Keep consistent with original script: start from index 1
    n = max(ref_len - 1, 0)
    return np.arange(n)


def main():
    parser = argparse.ArgumentParser(description="Plot flight results saved as .npy files")
    parser.add_argument("--result-dir", type=str, default="./result",
                        help="Directory that contains result .npy files (default: ./result)")
    parser.add_argument("--scenario", type=str, choices=["heading", "control", "tracking"], default="control",
                        help="Scenario type (only affects which target curves are attempted to plot)")
    args = parser.parse_args()

    result_dir = Path(args.result_dir).resolve()
    if not result_dir.exists():
        raise FileNotFoundError(f"Result directory not found: {result_dir}")

    # Load available arrays
    npos = load_if_exists(result_dir, "npos")
    epos = load_if_exists(result_dir, "epos")
    altitude = load_if_exists(result_dir, "altitude")

    roll = load_if_exists(result_dir, "roll")
    pitch = load_if_exists(result_dir, "pitch")
    yaw = load_if_exists(result_dir, "yaw")
    yaw_rate = load_if_exists(result_dir, "yaw_rate")

    roll_dem = load_if_exists(result_dir, "roll_dem")
    pitch_dem = load_if_exists(result_dir, "pitch_dem")
    yaw_rate_dem = load_if_exists(result_dir, "yaw_rate_dem")

    vt = load_if_exists(result_dir, "vt")
    alpha = load_if_exists(result_dir, "alpha")
    beta = load_if_exists(result_dir, "beta")
    G = load_if_exists(result_dir, "G")

    T = load_if_exists(result_dir, "T")
    throttle = load_if_exists(result_dir, "throttle")
    ail = load_if_exists(result_dir, "ail")
    el = load_if_exists(result_dir, "el")
    rud = load_if_exists(result_dir, "rud")

    # Targets (try multiple names depending on scenario/data availability)
    target_altitude = load_if_exists(result_dir, "target_altitude")
    target_heading = load_if_exists(result_dir, "target_heading")
    target_vt = load_if_exists(result_dir, "target_vt")
    target_pitch = load_if_exists(result_dir, "target_pitch")
    target_npos = load_if_exists(result_dir, "target_npos")
    target_epos = load_if_exists(result_dir, "target_epos")

    # Choose a reference length for time axis: first available among common series
    candidates = [altitude, vt, roll, pitch, yaw, T, throttle, ail, el, rud, alpha, beta, G]
    ref = next((x for x in candidates if x is not None), None)
    if ref is None:
        print("No time-series data found in result directory. Nothing to plot.")
        return
    t = make_time_axis(ref.shape[0])

    # 1) Position track (epos vs npos)
    if epos is not None and npos is not None:
        plt.figure()
        plt.plot(epos, npos, color='r')
        plt.xlabel('epos/feet')
        plt.ylabel('npos/feet')
        plt.title('position')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    # 2) Altitude
    if altitude is not None:
        plt.figure()
        plt.plot(t, altitude[1:], color='r', label='real')
        if target_altitude is not None:
            # Handle length mismatch by truncating target data to match actual data length
            target_len = min(len(target_altitude[1:]), len(altitude[1:]))
            plt.plot(t[:target_len], target_altitude[1:target_len+1], color='b', label='target')
        plt.legend()
        plt.xlabel('time/0.02s')
        plt.ylabel('altitude/feet')
        plt.title('altitude')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    # 3) Airspeed (vt)
    if vt is not None:
        plt.figure()
        plt.plot(t, vt[1:], color='r', label='real')
        if target_vt is not None:
            # Handle length mismatch by truncating target data to match actual data length
            target_len = min(len(target_vt[1:]), len(vt[1:]))
            plt.plot(t[:target_len], target_vt[1:target_len+1], color='b', label='target')
        plt.legend()
        plt.xlabel('time/0.02s')
        plt.ylabel('vt/feet')
        plt.title('vt')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    # 4) Roll
    if roll is not None:
        plt.figure()
        plt.plot(t, roll[1:] * 180 / np.pi, color='b', label='real')
        if roll_dem is not None:
            plt.plot(t, roll_dem[1:] * 180 / np.pi, color='r', label='demand')
        plt.legend()
        plt.xlabel('time/0.02s')
        plt.ylabel('roll/deg')
        plt.title('roll')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    # 5) Pitch
    if pitch is not None:
        plt.figure()
        plt.plot(t, pitch[1:] * 180 / np.pi, color='b', label='real')
        if pitch_dem is not None:
            plt.plot(t, pitch_dem[1:] * 180 / np.pi, color='r', label='demand')
        if target_pitch is not None:
            plt.plot(t, target_pitch[1:] * 180 / np.pi, color='g', linestyle='--', label='target')
        plt.legend()
        plt.xlabel('time/0.02s')
        plt.ylabel('pitch/deg')
        plt.title('pitch')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    # 6) Yaw
    if yaw is not None:
        plt.figure()
        plt.plot(t, yaw[1:] * 180 / np.pi, color='b', label='real')
        if target_heading is not None:
            # Handle length mismatch by truncating target data to match actual data length
            target_len = min(len(target_heading[1:]), len(yaw[1:]))
            plt.plot(t[:target_len], target_heading[1:target_len+1] * 180 / np.pi, color='r', label='target')
        plt.legend()
        plt.xlabel('time/0.02s')
        plt.ylabel('yaw/deg')
        plt.title('yaw')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    # 7) Yaw rate
    if yaw_rate is not None:
        plt.figure()
        plt.plot(t, yaw_rate[1:] * 180 / np.pi, color='b', label='real')
        if yaw_rate_dem is not None:
            plt.plot(t, yaw_rate_dem[1:] * 180 / np.pi, color='r', label='demand')
        plt.legend()
        plt.xlabel('time/0.02s')
        plt.ylabel('yaw_rate/deg/s')
        plt.title('yaw_rate')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    # 8) Thrust
    if T is not None:
        plt.figure()
        plt.plot(t, T[1:], color='k')
        plt.xlabel('time/0.02s')
        plt.ylabel('T/lbf')
        plt.title('T')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    # 9) Throttle
    if throttle is not None:
        plt.figure()
        plt.plot(t, 100 * throttle[1:], color='k')
        plt.xlabel('time/0.02s')
        plt.ylabel('throttle/%')
        plt.title('throttle')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    # 10) Control surfaces
    if el is not None:
        plt.figure()
        plt.plot(t, el[1:], color='k')
        plt.xlabel('time/0.02s')
        plt.ylabel('el/deg')
        plt.title('el')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    if ail is not None:
        plt.figure()
        plt.plot(t, ail[1:], color='k')
        plt.xlabel('time/0.02s')
        plt.ylabel('ail/deg')
        plt.title('ail')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    if rud is not None:
        plt.figure()
        plt.plot(t, rud[1:], color='k')
        plt.xlabel('time/0.02s')
        plt.ylabel('rud/deg')
        plt.title('rud')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    # 11) Angle of attack/sideslip and G
    if alpha is not None:
        plt.figure()
        plt.plot(t, alpha[1:], color='m')
        plt.xlabel('time/0.02s')
        plt.ylabel('alpha/rad')
        plt.title('alpha (AOA)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    if beta is not None:
        plt.figure()
        plt.plot(t, beta[1:], color='c')
        plt.xlabel('time/0.02s')
        plt.ylabel('beta/rad')
        plt.title('beta (AOS)')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    if G is not None:
        plt.figure()
        plt.plot(t, G[1:], color='tab:orange')
        plt.xlabel('time/0.02s')
        plt.ylabel('G')
        plt.title('G-force')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()


if __name__ == "__main__":
    main()

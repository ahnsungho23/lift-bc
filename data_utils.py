import h5py
import numpy as np

DATA_PATH = "/home/sungho/bc_ws/robomimic/datasets/lift/ph/low_dim_v15.hdf5"
OBS_KEYS = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]


def load_demos(path, num_demos=None):
    obs_list = []
    act_list = []

    with h5py.File(path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda name: int(name.split("_")[1]))
        if num_demos is not None:
            demos = demos[:num_demos]

        for name in demos:
            demo = f["data"][name]
            parts = [demo["obs"][key][:] for key in OBS_KEYS]
            obs_list.append(np.concatenate(parts, axis=1))   # (T, 19)
            act_list.append(demo["actions"][:])              # (T, 7)

    all_obs = np.concatenate(obs_list, axis=0)
    all_actions = np.concatenate(act_list, axis=0)
    return all_obs, all_actions


if __name__ == "__main__":
    obs, actions = load_demos(DATA_PATH)
    print("obs:", obs.shape, obs.dtype)
    print("actions:", actions.shape, actions.dtype)

    obs10, actions10 = load_demos(DATA_PATH, num_demos=10)
    print("데모 10개만:", obs10.shape, actions10.shape)
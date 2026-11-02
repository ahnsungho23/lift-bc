import numpy as np
import imageio
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

path = "/home/sungho/bc_ws/robomimic/datasets/lift/ph/low_dim_v15.hdf5"

ObsUtils.initialize_obs_utils_with_obs_specs({
    "obs": {"low_dim": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]}
})

env_meta = FileUtils.get_env_metadata_from_dataset(path)
env = EnvUtils.create_env_from_metadata(
    env_meta=env_meta, render=False, render_offscreen=True, use_image_obs=False,
)

obs = env.reset()

frames = []
for t in range(100):
    action = np.random.uniform(-1, 1, size=7)
    obs, reward, done, info = env.step(action)
    img = env.render(mode="rgb_array", height=256, width=256, camera_name="frontview")
    frames.append(np.array(img, dtype=np.uint8, copy=True))   # ← 매 프레임 복사해서 저장

print("프레임:", len(frames), frames[0].shape, frames[0].dtype)
print("성공?", env.is_success())
imageio.mimsave("random_rollout.gif", frames, fps=20)
print("random_rollout.gif 저장 완료")

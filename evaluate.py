import numpy as np
import torch
import imageio
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

from data_utils import DATA_PATH, OBS_KEYS
from model import BCPolicy


def make_env():
    """데이터셋에 저장된 설정 그대로 Lift 환경 만들기 (check_env.py와 같음)"""
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": OBS_KEYS}})
    env_meta = FileUtils.get_env_metadata_from_dataset(DATA_PATH)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=True, use_image_obs=False,
    )
    return env


def obs_to_vector(obs):
    parts = [obs[key] for key in OBS_KEYS]
    return np.concatenate(parts, axis = 0)


def get_action(policy, obs_vec):
    x = torch.from_numpy(obs_vec).float().unsqueeze(0)
    with torch.no_grad():
        out = policy(x)
    return  out.squeeze(0).numpy()


def evaluate(policy, env, num_episodes=50, horizon=400, gif_path=None):
    policy.eval()
    successes = 0

    for ep in range(num_episodes):
        obs = env.reset()                      # 큐브가 새로운 위치에 놓임
        frames = []
        success = False

        for t in range(horizon):
            obs_vec = obs_to_vector(obs)
            action = get_action(policy, obs_vec)
            action = np.clip(action, -1, 1)    # 행동을 -1~1 범위로 자르기
            obs, reward, done, info = env.step(action)

            if gif_path is not None and ep == 0:   # 첫 시도만 GIF로 저장
                img = env.render(mode="rgb_array", height=256, width=256, camera_name="frontview")
                frames.append(np.array(img, dtype=np.uint8, copy=True))

            if env.is_success()["task"]:       # 큐브를 들었으면 성공
                success = True
                break

        successes += int(success)
        print(f"episode {ep:2d} | {'성공' if success else '실패'} | {t + 1} steps")

        if gif_path is not None and ep == 0:
            imageio.mimsave(gif_path, frames, fps=20)

    return successes / num_episodes


if __name__ == "__main__":
    np.random.seed(0)        # 큐브 위치를 매번 같게 (공정한 비교를 위해)
    torch.manual_seed(0)

    policy = BCPolicy()
    policy.load_state_dict(torch.load("checkpoints/bc_all.pt", weights_only=True))

    env = make_env()
    rate = evaluate(policy, env, num_episodes=50, gif_path="results/rollout_all.gif")
    print(f"\n성공률: {rate:.2f}")
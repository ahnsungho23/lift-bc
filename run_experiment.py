import math
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_utils import load_demos, DATA_PATH
from model import DemoDataset, BCPolicy
from evaluate import make_env, evaluate

TOTAL_UPDATES = 7600     # 모든 실험에서 가중치 갱신 횟수를 같게
BATCH_SIZE = 256
LR = 1e-3


def train_policy(num_demos, shuffle=True, seed=0):
    """train.py와 같은 학습. 갱신 횟수가 TOTAL_UPDATES가 되도록 에포크 수를 정함"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    obs, actions = load_demos(DATA_PATH, num_demos=num_demos)
    ds = DemoDataset(obs, actions)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)
    epochs = math.ceil(TOTAL_UPDATES / len(loader))

    policy = BCPolicy()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    for epoch in range(epochs):
        total_loss = 0
        for s_batch, a_batch in loader:
            pred = policy(s_batch)
            loss = loss_fn(pred, a_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

    final_loss = total_loss / len(loader)
    print(f"\n[데모 {num_demos}개, shuffle={shuffle}] 스텝 {len(ds)}개, 에포크 {epochs}, 최종 손실 {final_loss:.4f}")
    return policy, final_loss, len(ds), epochs


if __name__ == "__main__":
    configs = [
        (25, True),
        (50, True),
        (100, True),
        (200, True),
        (200, False),    # 섞지 않으면 어떻게 되나
    ]

    env = make_env()
    results = []

    for num_demos, shuffle in configs:
        policy, final_loss, num_steps, epochs = train_policy(num_demos, shuffle=shuffle)
        torch.save(policy.state_dict(), f"checkpoints/bc_{num_demos}_shuffle{int(shuffle)}.pt")

        np.random.seed(0)    # 모든 정책을 같은 50개 큐브 위치에서 평가
        rate = evaluate(policy, env, num_episodes=50)
        print(f"==> 데모 {num_demos}개, shuffle={shuffle}: 성공률 {rate:.2f}")

        results.append({
            "num_demos": num_demos, "shuffle": shuffle, "num_steps": num_steps,
            "epochs": epochs, "final_loss": final_loss, "success_rate": rate,
        })
        with open("results/results.json", "w") as f:    # 실험 하나 끝날 때마다 저장
            json.dump(results, f, indent=2)

    # 데모 수별 성공률 그래프 (shuffle=True만)
    main = [r for r in results if r["shuffle"]]
    xs = [r["num_demos"] for r in main]
    ys = [r["success_rate"] * 100 for r in main]

    plt.figure()
    plt.plot(xs, ys, marker="o", label="shuffle=True")
    no_shuffle = [r for r in results if not r["shuffle"]]
    if no_shuffle:
        plt.scatter([no_shuffle[0]["num_demos"]], [no_shuffle[0]["success_rate"] * 100],
                    marker="x", s=80, color="red", label="shuffle=False")
    for x, y in zip(xs, ys):
        plt.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, 8), ha="center")
    plt.xlabel("number of demos")
    plt.ylabel("success rate (%)")
    plt.xticks(xs)
    plt.ylim(0, 105)
    plt.legend()
    plt.title("BC on Lift: success rate vs. number of demos")
    plt.savefig("results/success_vs_demos.png", dpi=150, bbox_inches="tight")
    print("\n그래프 저장: results/success_vs_demos.png")
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_utils import load_demos, DATA_PATH
from model import DemoDataset, BCPolicy

torch.manual_seed(0)
os.makedirs("checkpoints", exist_ok=True)   # 학습된 모델 저장 폴더
os.makedirs("results", exist_ok=True)       # 그래프 저장 폴더

num_demos = None     # None이면 200개 전부
batch_size = 256
learning_rate = 1e-3
epochs = 200

# ===== 재료 1: 데이터 =====
obs, actions = load_demos(DATA_PATH, num_demos=num_demos)
print("데이터:", obs.shape, actions.shape)

ds = DemoDataset(obs, actions)
loader = DataLoader(ds, batch_size, shuffle = True) # (batch_size 변수 쓰기, 섞기)

# ===== 재료 2: 모델 =====
policy = BCPolicy()

# ===== 재료 3: 손실 함수 =====
loss_fn = nn.MSELoss()      #MNIST처럼 값하나가 필요한 것이 아니라, 여러개의 정확한 값을 얻기위해 MSE

# ===== 재료 4: 옵티마이저 =====
optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

# ===== 학습 =====
loss_history = []
for epoch in range(epochs):
    total_loss = 0
    for s_batch, a_batch in loader:
        pred = policy(s_batch)
        loss = loss_fn(pred, a_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    loss_history.append(avg_loss)
    if epoch % 20 == 0 or epoch == epochs - 1:
        print(f"epoch {epoch:3d} | loss {avg_loss:.5f}")

# ===== 저장 =====
torch.save(policy.state_dict(), "checkpoints/bc_all.pt")
print("모델 저장: checkpoints/bc_all.pt")

plt.figure()
plt.plot(loss_history)
plt.xlabel("epoch")
plt.ylabel("MSE loss")
plt.yscale("log")
plt.savefig("results/train_loss_all.png", dpi=150, bbox_inches="tight")
print("그래프 저장: results/train_loss_all.png")
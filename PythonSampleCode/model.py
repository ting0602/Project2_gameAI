import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class DQNAgent:
    def __init__(self, state_size, action_size, batch_size=32, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.batch_size = batch_size
        self.gamma = gamma    # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters())

    def _build_model(self):
        # 定義一個神經網絡模型
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        # 將狀態、行動、獎勵、下一個狀態、結束標誌儲存在記憶庫中
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # 使用epsilon-greedy策略選擇行動
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model(torch.Tensor(state)).detach().numpy())

    def replay(self):
        # 從記憶庫中取出一批樣本進行訓練
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # 計算目標Q值
                target = (reward + self.gamma * np.amax(self.model(torch.Tensor(next_state)).detach().numpy()))

            # 計算當前Q值
            current = self.model(torch.Tensor(state))[action].detach().numpy()

            # 計算損失函數
            loss = nn.MSELoss()(torch.Tensor([target]), torch.Tensor([current]))

            # 反向傳播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # 降低探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
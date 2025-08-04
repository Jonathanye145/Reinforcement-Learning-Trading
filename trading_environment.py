
# trading_bot/trading_environment.py
# Defines the custom Gym environment for reinforcement learning.

import gym
import numpy as np
import pandas as pd
from gym import spaces
from config import INITIAL_BALANCE, COMMISSION_RATE, STOP_LOSS_THRESHOLD, RL_FEATURES, MAX_ALLOCATION

class TradingEnv(gym.Env):
    def __init__(self, data: pd.DataFrame, window_size: int):
        super().__init__()
        self.data = data
        self.window_size = window_size
        self.initial_balance = INITIAL_BALANCE
        self.commission_rate = COMMISSION_RATE
        self.stop_loss_threshold = STOP_LOSS_THRESHOLD
        self.features = RL_FEATURES
        if not all(feature in self.data.columns for feature in self.features):
            raise ValueError("A feature from RL_FEATURES is missing from the data.")
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size, len(self.features)), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=MAX_ALLOCATION, shape=(1,), dtype=np.float32)

    def reset(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        self.portfolio_values = [self.initial_balance]
        return self._get_observation()

    def _get_observation(self):
        return self.data[self.features].iloc[self.current_step - self.window_size : self.current_step].values.astype(np.float32)

    def step(self, action):
        action = np.clip(action[0], self.action_space.low[0], self.action_space.high[0])
        current_price = self.data["Adj Close"].iloc[self.current_step]
        current_portfolio_value = self.balance + self.shares_held * current_price
        target_stock_value = current_portfolio_value * action
        shares_to_hold = target_stock_value / current_price if current_price > 0 else 0
        shares_to_trade = shares_to_hold - self.shares_held
        cost = abs(shares_to_trade) * current_price * self.commission_rate
        self.balance -= (shares_to_trade * current_price) + cost
        self.shares_held += shares_to_trade
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        new_price = self.data["Adj Close"].iloc[self.current_step] if not done else current_price
        portfolio_value = self.balance + self.shares_held * new_price
        reward = np.log(portfolio_value / self.portfolio_values[-1]) if self.portfolio_values[-1] != 0 else 0
        self.portfolio_values.append(portfolio_value)
        if portfolio_value < self.initial_balance * self.stop_loss_threshold:
            done = True
            reward = -1.0
        return self._get_observation(), reward, done, {"portfolio_value": portfolio_value}

if __name__ == '__main__':
    from config import WINDOW_SIZE
    print("\n--- Testing Trading Environment Module ---")
    num_rows = 50
    dummy_data = {"Adj Close": np.random.uniform(150, 250, num_rows)}
    for feature in RL_FEATURES: dummy_data[feature] = np.random.rand(num_rows)
    df = pd.DataFrame(dummy_data)
    env = TradingEnv(df, window_size=WINDOW_SIZE)
    obs = env.reset()
    print(f"Env created. Initial observation shape: {obs.shape}")
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step {i+1} -> Reward: {reward:.4f}, Portfolio Value: ${info['portfolio_value']:,.2f}")
    print("\nEnvironment test complete.")

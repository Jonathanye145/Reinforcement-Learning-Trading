
# trading_bot/model_handler.py
# Contains the logic for training and testing the PPO model.

import logging
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_environment import TradingEnv
from config import WINDOW_SIZE, TRAINING_TIMESTEPS

logger = logging.getLogger(__name__)

def train_model(train_data: pd.DataFrame):
    if train_data.empty:
        logger.error("Cannot train model with empty data.")
        return None
    logger.info(f"Starting model training for {TRAINING_TIMESTEPS} timesteps...")
    env = DummyVecEnv([lambda: TradingEnv(train_data, window_size=WINDOW_SIZE)])
    model = PPO("MlpLstmPolicy", env, verbose=0, n_steps=2048, batch_size=64,
                gamma=0.99, gae_lambda=0.95, n_epochs=10, ent_coef=0.01)
    model.learn(total_timesteps=TRAINING_TIMESTEPS)
    logger.info("Model training complete.")
    return model

def test_model(model, test_data: pd.DataFrame):
    if test_data.empty or model is None:
        logger.error("Cannot test model due to missing data or model.")
        return pd.DataFrame(), [], []
    env = TradingEnv(test_data, window_size=WINDOW_SIZE)
    obs = env.reset()
    actions, portfolio_values = [], [env.initial_balance]
    lstm_states = None
    done = False
    while not done:
        action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
        obs, reward, done, info = env.step(action)
        actions.append(action[0])
        portfolio_values.append(info["portfolio_value"])
    start_index = env.window_size
    end_index = start_index + len(actions)
    viz_data = test_data.iloc[start_index:end_index]
    return viz_data, actions, portfolio_values

if __name__ == '__main__':
    from config import RL_FEATURES
    import numpy as np
    logging.basicConfig(level=logging.INFO)
    print("\n--- Testing Model Handler Module ---")
    num_rows = 300
    dummy_data = {"Adj Close": np.random.uniform(150, 250, num_rows)}
    for feature in RL_FEATURES: dummy_data[feature] = np.random.rand(num_rows)
    df = pd.DataFrame(dummy_data)
    print("Training a model for a few steps (integration test)...")
    temp_model = train_model(df)
    if temp_model:
        print("\nTesting the model...")
        viz_data, actions, values = test_model(temp_model, df)
        if not viz_data.empty: print("Model testing function executed successfully.")
        else: print("Model testing function FAILED.")
    else:
        print("Model training function FAILED.")

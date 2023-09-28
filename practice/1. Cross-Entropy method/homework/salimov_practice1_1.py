import time
from typing import List

import gym
import joblib
import numpy as np
import pandas as pd
from gym import Env
from tqdm import tqdm


class CrossEntropyAgent:
    def __init__(self, state_n: int, action_n: int):
        self.state_n = state_n
        self.action_n = action_n
        self.model = np.ones((self.state_n, self.action_n)) / self.action_n

    def get_action(self, state: int) -> int:
        action = int(np.random.choice(np.arange(self.action_n), p=self.model[state]))
        return action

    def fit(self, elite_trajectories: List[dict]) -> None:
        new_model = np.zeros((self.state_n, self.action_n))
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory["states"], trajectory["actions"]):
                new_model[state][action] += 1

        for state in range(self.state_n):
            row_sum = np.sum(new_model[state])  # сумма всех действий по фикс состоянию
            if row_sum > 0:
                new_model[state] /= row_sum
            else:
                new_model[state] = self.model[state].copy()
        self.model = new_model
        return None


def get_trajectory(env: Env, agent: CrossEntropyAgent, max_iter: int, visualize: bool = False):
    trajectory = {"states": [], "actions": [], "rewards": []}
    state = env.reset()  # [0; 500]
    for _ in range(max_iter):
        trajectory["states"].append(state)
        action = agent.get_action(state)
        trajectory["actions"].append(action)
        state, reward, done, _ = env.step(action)
        trajectory["rewards"].append(reward)
        if visualize:
            env.render()
            time.sleep(0.5)
        if done:
            break
    return trajectory


def cross_entropy_agent_model(
        state_n: int,
        action_n: int,
        quantile_param: float = 0.9,
        iterations_N: int = 20,
        trajectories_K: int = 50,
        max_iter: int = 200):
    env = gym.make("Taxi-v3")
    agent = CrossEntropyAgent(action_n=action_n, state_n=state_n)
    best_mean_total_reward = -np.inf
    for n in range(iterations_N):
        # policy evaluation
        trajectories = [get_trajectory(env, agent, max_iter, visualize=False) for _ in range(trajectories_K)]
        total_rewards = [np.sum(trajectory["rewards"]) for trajectory in trajectories]
        # print(f"Iter: {n} | Total reward: {np.mean(total_rewards)}")  # в график в зависимости от итерации

        if np.mean(total_rewards) > best_mean_total_reward:
            best_mean_total_reward = np.mean(total_rewards)

        # policy improvement
        gamma_quantile = np.quantile(total_rewards, quantile_param)
        elite_trajectories = []
        for trajectory in trajectories:
            if np.sum(trajectory["rewards"]) > gamma_quantile:
                elite_trajectories.append(trajectory)

        agent.fit(elite_trajectories)

    return best_mean_total_reward


if __name__ == "__main__":
    # Rewards:
    # -1 per step unless other reward is triggered.
    # +20 delivering passenger.
    # -10 executing “pickup” and “drop-off” actions illegally.

    state_n = 500
    # 25 taxi positions,
    # 5 possible locations of the passenger (including the case when the passenger is in the taxi) [R, G, Y, B, IN_TAXI]
    # 4 destination locations. [R, G, Y, B]

    action_n = 6
    # 0: move south
    # 1: move north
    # 2: move east
    # 3: move west
    # 4: pickup passenger
    # 5: drop off passenger

    hyperparams = {
        "iterations_N": [20],
        "quantile_params": [0.1, 0.3, 0.5, 0.7, 0.9],
        "trajectories_K": [100, 1000, 5000, 10000, 15000],
        "max_iter": [100, 200, 500, 1000, 2000],
    }

    result = {
        "iterations_N": [],
        "quantile_params": [],
        "trajectories_K": [],
        "max_iter": [],
        "best_mean_total_reward": [],
    }

    for iterations_N in hyperparams["iterations_N"]:
        for quantile_param in hyperparams["quantile_params"]:
            for trajectories_K in hyperparams["trajectories_K"]:
                for max_iter in hyperparams["max_iter"]:
                    result["iterations_N"].append(iterations_N)
                    result["quantile_params"].append(quantile_param)
                    result["trajectories_K"].append(trajectories_K)
                    result["max_iter"].append(max_iter)

    best_mean_total_rewards = joblib.Parallel(n_jobs=10)(
        joblib.delayed(cross_entropy_agent_model)(state_n, action_n, quantile_param, iterations_N, trajectories_K,
                                                  max_iter) for
        iterations_N,
        quantile_param,
        trajectories_K,
        max_iter in tqdm(zip(
            result["iterations_N"],
            result["quantile_params"],
            result["trajectories_K"],
            result["max_iter"]
        )))

    result["best_mean_total_reward"] = best_mean_total_rewards

    df = pd.DataFrame.from_dict(result)
    df.to_csv("result_1.csv", index=False)

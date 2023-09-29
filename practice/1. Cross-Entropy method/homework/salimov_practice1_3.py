import time
from typing import List

import gym
import joblib
import matplotlib.pyplot as plt
import numpy as np
from joblib import parallel_config, delayed


class CrossEntropyAgent:
    def __init__(self, state_n: int, action_n: int):
        self.state_n = state_n
        self.action_n = action_n
        self.model = np.ones((self.state_n, self.action_n)) / self.action_n
        self.deterministic_model = np.zeros_like(self.model)

        self.is_train = True
        self.seed = 0

    def train(self):
        self.is_train = True

    def eval(self):
        self.is_train = False

    def sample_deterministic_model(self):
        np.random.seed(self.seed)
        self.deterministic_model = np.zeros_like(self.model)
        for state in range(state_n):
            action = int(np.random.choice(np.arange(self.action_n), p=self.model[state]))
            self.deterministic_model[state][action] = 1.0

    def get_action(self, state: int) -> int:
        assert np.sum(self.deterministic_model[state]) == 1
        assert max(self.deterministic_model[state]) == 1
        assert int(np.random.choice(np.arange(self.action_n), p=self.deterministic_model[state])) == int(
            np.argmax(self.deterministic_model[state]))
        if self.is_train:
            action = int(np.argmax(self.deterministic_model[state]))
        else:
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


def get_trajectory(agent: CrossEntropyAgent, max_iter: int, visualize: bool = False):
    trajectory = {"states": [], "actions": [], "rewards": []}
    env = gym.make("Taxi-v3")
    state = env.reset(seed=np.random.randint(1, 1000000))
    # print(state)
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
        deterministic_model_M: int = 10,
        max_iter: int = 200):
    agent = CrossEntropyAgent(action_n=action_n, state_n=state_n)
    list_of_total_rewards = []

    for n in range(iterations_N):
        mean_rewards_over_deterministic_models = []
        pool_of_trajectories_over_deterministic_models = []
        # policy evaluation
        agent.train()
        for m in range(deterministic_model_M):
            agent.sample_deterministic_model()
            # start = time.time()
            # trajectories = [get_trajectory(env, agent, max_iter, visualize=False) for _ in range(trajectories_K)]
            with parallel_config(n_jobs=10):
                trajectories = joblib.Parallel()(delayed(get_trajectory)(agent, max_iter) for _ in range(trajectories_K))

            # end = time.time()
            # print(end - start)
            mean_reward_over_deterministic_model = np.mean(
                [np.sum(trajectory["rewards"]) for trajectory in trajectories])
            pool_of_trajectories_over_deterministic_models.append(trajectories)
            mean_rewards_over_deterministic_models.append(mean_reward_over_deterministic_model)

        # policy improvement
        gamma_quantile = np.quantile(mean_rewards_over_deterministic_models, quantile_param)
        elite_trajectories = []
        for trajectories, total_reward in zip(pool_of_trajectories_over_deterministic_models,
                                              mean_rewards_over_deterministic_models):
            if total_reward > gamma_quantile:
                elite_trajectories.extend(trajectories)

        agent.fit(elite_trajectories)

        # policy get metric
        agent.eval()
        random_trajectories = [get_trajectory(agent, max_iter, visualize=False) for _ in range(trajectories_K)]
        random_total_rewards = [np.sum(trajectory["rewards"]) for trajectory in random_trajectories]
        list_of_total_rewards.append(np.mean(random_total_rewards))
        print(f"Iter: {n} | Total reward: {np.mean(random_total_rewards)}")

        agent.seed += 1

    return list_of_total_rewards


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

    best_quantile_param = 0.8
    best_iterations_N = 60
    best_trajectories_K = 5000
    best_max_iter = 500
    deterministic_model_M = 50

    fig = plt.figure(figsize=(20, 10))
    ax = plt.axes()

    list_of_total_rewards = cross_entropy_agent_model(state_n=state_n,
                                                      action_n=action_n,
                                                      quantile_param=best_quantile_param,
                                                      iterations_N=best_iterations_N,
                                                      trajectories_K=best_trajectories_K,
                                                      deterministic_model_M=deterministic_model_M,
                                                      max_iter=best_max_iter, )

    ax.plot(np.arange(best_iterations_N), list_of_total_rewards, label=f"deterministic_model_M={deterministic_model_M}")

    ax.legend()
    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylabel("mean total reward")
    fig.savefig("../result_3.png")

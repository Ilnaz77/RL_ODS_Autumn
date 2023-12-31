import time
from typing import List

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import Env


class CrossEntropyAgent:
    def __init__(self,
                 state_n: int,
                 action_n: int,
                 laplace_lambda: int = 10,
                 policy_lambda: float = 1.0,
                 is_laplace_smooth: bool = False,
                 is_policy_smooth: bool = False, ):

        # assert is_policy_smooth != is_laplace_smooth
        assert 0 < policy_lambda <= 1
        assert laplace_lambda > 0

        self.is_laplace_smooth = is_laplace_smooth
        self.is_policy_smooth = is_policy_smooth

        self.laplace_lambda = laplace_lambda
        self.policy_lambda = policy_lambda

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

        if self.is_laplace_smooth:
            new_model = new_model + self.laplace_lambda

        for state in range(self.state_n):
            row_sum = np.sum(new_model[state])  # сумма всех действий по фикс состоянию
            if row_sum > 0:
                new_model[state] /= row_sum
            else:
                new_model[state] = self.model[state].copy()

        if self.is_policy_smooth:
            self.model = self.policy_lambda * new_model + (1 - self.policy_lambda) * self.model
        else:
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
        env: Env,
        state_n: int,
        action_n: int,
        laplace_lambda: int = 10,
        policy_lambda: float = 1.0,
        is_laplace_smooth: bool = False,
        is_policy_smooth: bool = False,
        quantile_param: float = 0.9,
        iterations_N: int = 20,
        trajectories_K: int = 50,
        max_iter: int = 200):
    agent = CrossEntropyAgent(action_n=action_n,
                              state_n=state_n,
                              laplace_lambda=laplace_lambda,
                              policy_lambda=policy_lambda,
                              is_policy_smooth=is_policy_smooth,
                              is_laplace_smooth=is_laplace_smooth, )
    list_of_total_rewards = []
    for n in range(iterations_N):
        # policy evaluation
        trajectories = [get_trajectory(env, agent, max_iter, visualize=False) for _ in range(trajectories_K)]
        total_rewards = [np.sum(trajectory["rewards"]) for trajectory in trajectories]
        list_of_total_rewards.append(np.mean(total_rewards))
        # print(f'n: {n}, reward: {np.mean(total_rewards)}')
        # policy improvement
        gamma_quantile = np.quantile(total_rewards, quantile_param)
        elite_trajectories = []
        for trajectory in trajectories:
            if np.sum(trajectory["rewards"]) > gamma_quantile:
                elite_trajectories.append(trajectory)

        agent.fit(elite_trajectories)

    # trajectory = get_trajectory(env, agent, max_iter, visualize=True)
    # print('total reward:', sum(trajectory['rewards']))
    # print('model:\n', agent.model)
    return list_of_total_rewards


if __name__ == "__main__":
    env = gym.make("Taxi-v3")
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

    # laplace_lambda: int = 10
    # policy_lambda: float = 1.0
    # is_laplace_smooth: bool = False
    # is_policy_smooth: bool = False

    best_quantile_param = 0.7
    best_iterations_N = 20
    best_trajectories_K = 15000
    best_max_iter = 500

    fig = plt.figure(figsize=(20, 10))
    ax = plt.axes()

    list_of_total_rewards = cross_entropy_agent_model(env=env,
                                                      state_n=state_n,
                                                      action_n=action_n,
                                                      is_laplace_smooth=False,
                                                      is_policy_smooth=False,
                                                      quantile_param=best_quantile_param,
                                                      iterations_N=best_iterations_N,
                                                      trajectories_K=best_trajectories_K,
                                                      max_iter=best_max_iter, )
    ax.plot(np.arange(best_iterations_N), list_of_total_rewards, label=f"Standard best model")

    print("Laplace step ...")
    for laplace_lambda in [0.1, 0.5, 1, 5, 20]:
        list_of_total_rewards = cross_entropy_agent_model(env=env,
                                                          state_n=state_n,
                                                          action_n=action_n,
                                                          laplace_lambda=laplace_lambda,
                                                          is_laplace_smooth=True,
                                                          is_policy_smooth=False,
                                                          quantile_param=best_quantile_param,
                                                          iterations_N=best_iterations_N,
                                                          trajectories_K=best_trajectories_K,
                                                          max_iter=best_max_iter, )
        ax.plot(np.arange(best_iterations_N), list_of_total_rewards, label=f"Laplace smooth | lambda={laplace_lambda}")

    print("Policy step ...")
    for policy_lambda in [0.1, 0.3, 0.5, 0.7]:
        list_of_total_rewards = cross_entropy_agent_model(env=env,
                                                          state_n=state_n,
                                                          action_n=action_n,
                                                          policy_lambda=policy_lambda,
                                                          is_laplace_smooth=False,
                                                          is_policy_smooth=True,
                                                          quantile_param=best_quantile_param,
                                                          iterations_N=best_iterations_N,
                                                          trajectories_K=best_trajectories_K,
                                                          max_iter=best_max_iter, )
        ax.plot(np.arange(best_iterations_N), list_of_total_rewards, label=f"Policy smooth | lambda={policy_lambda}")

    ax.legend()
    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylabel("mean total reward")
    fig.savefig("../result_2.png")

import time
from typing import List

import gym
import numpy as np
from gym import Env


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
        env: Env,
        state_n: int,
        action_n: int,
        quantile_param: float = 0.9,
        iterations_N: int = 20,
        trajectories_K: int = 50,
        max_iter: int = 200):

    agent = CrossEntropyAgent(action_n=action_n, state_n=state_n)

    for n in range(iterations_N):
        # policy evaluation
        trajectories = [get_trajectory(env, agent, max_iter, visualize=False) for _ in range(trajectories_K)]
        total_rewards = [np.sum(trajectory["rewards"]) for trajectory in trajectories]
        print(f"Iter: {n} | Total reward: {np.mean(total_rewards)}")  # в график в зависимости от итерации

        # policy improvement
        gamma_quantile = np.quantile(total_rewards, quantile_param)
        elite_trajectories = []
        for trajectory in trajectories:
            if np.sum(trajectory["rewards"]) > gamma_quantile:
                elite_trajectories.append(trajectory)

        agent.fit(elite_trajectories)

    trajectory = get_trajectory(env, agent, max_iter, visualize=True)
    print('total reward:', sum(trajectory['rewards']))
    print('model:\n', agent.model)


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

    quantile_param = 0.9
    iterations_N = 20
    trajectories_K = 10000
    max_iter = 1000

    cross_entropy_agent_model(env=env,
                              state_n=state_n,
                              action_n=action_n,
                              quantile_param=quantile_param,
                              iterations_N=iterations_N,
                              trajectories_K=trajectories_K,
                              max_iter=max_iter,)

    # найти хорошее лямбда и сравнить графики с сглаживанием и без (стало лучше или нет)
    # сглаживание лапласа дало рез-т или нет ?
    # сглаживание политики дало рез-т или нет ?

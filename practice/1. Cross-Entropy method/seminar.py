import time
from typing import Tuple

import gym
import gym_maze  # pylint:disable
import numpy as np

if gym_maze:
    print("gym maze is exist")
else:
    raise Exception


def get_state(obs: Tuple[int, int]) -> int:
    # Вместо 2-х чисел координат возвращаем 1 число - номер ячейки
    return int(np.sqrt(state_n) * obs[0] + obs[1])


def get_trajectory(env, agent, max_iter=1000, visualize=False):
    trajectory = {"states": [], "actions": [], "rewards": []}

    observation = env.reset()
    state = get_state(observation)

    for _ in range(max_iter):
        trajectory["states"].append(state)

        action = agent.get_action(state)
        trajectory["actions"].append(action)

        observation, reward, done, _ = env.step(action)
        trajectory["rewards"].append(reward)

        state = get_state(observation)

        if visualize:
            env.render()
            time.sleep(0.5)

        if done:
            break

    return trajectory


class RandomAgent:
    def __init__(self, action_n: int):
        self.action_n = action_n

    def get_action(self, state):
        action = np.random.randint(self.action_n)  # from [0 to 3]
        return action


class CrossEntropyAgent:
    def __init__(self, state_n: int, action_n: int):
        self.state_n = state_n
        self.action_n = action_n
        self.model = np.ones((self.state_n, self.action_n)) / self.action_n

    def get_action(self, state: int):
        action = int(np.random.choice(np.arange(self.action_n), p=self.model[state]))
        return action

    def fit(self, elite_trajectories):
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


def random_agent_model(action_n, env):
    agent = RandomAgent(action_n=action_n)
    trajectory = get_trajectory(env, agent, max_iter=100, visualize=True)
    return trajectory


def cross_entropy_agent_model(state_n: int, action_n: int):
    agent = CrossEntropyAgent(action_n=action_n, state_n=state_n)

    quantile_param = 0.9
    iteration_n = 20
    trajectory_n = 50

    for n in range(iteration_n):
        # policy evaluation
        trajectories = [get_trajectory(env, agent, visualize=False) for _ in range(trajectory_n)]
        total_rewards = [np.sum(trajectory["rewards"]) for trajectory in trajectories]
        print(f"Iter: {n} | Total reward: {np.mean(total_rewards)}")  #  в график в зависимости от итерации

        # policy improvement
        gamma_quantile = np.quantile(total_rewards, quantile_param)
        elite_trajectories = []
        for trajectory in trajectories:
            if np.sum(trajectory["rewards"]) > gamma_quantile:
                elite_trajectories.append(trajectory)

        agent.fit(elite_trajectories)

    trajectory = get_trajectory(env, agent, visualize=True)
    print('total reward:', sum(trajectory['rewards']))
    print('model:\n', agent.model)


if __name__ == "__main__":
    env = gym.make('maze-sample-5x5-v0')
    state_n = 25
    action_n = 4

    cross_entropy_agent_model(state_n=state_n, action_n=action_n)

    # найти хорошее лямбда и сравнить графики с сглаживанием и без (стало лучше или нет)
    # сглаживание лапласа дало рез-т или нет ?
    # сглаживание политики дало рез-т или нет ?
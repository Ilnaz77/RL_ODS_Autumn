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
        self.deterministic_model = np.zeros_like(self.model)

    def sample_deterministic_model(self):
        self.deterministic_model = np.zeros_like(self.model)
        for state in range(state_n):
            action = int(np.random.choice(np.arange(self.action_n), p=self.model[state]))
            self.deterministic_model[state][action] = 1.0

    def get_action(self, state: int) -> int:
        assert np.sum(self.deterministic_model[state]) == 1
        assert max(self.deterministic_model[state]) == 1
        assert int(np.random.choice(np.arange(self.action_n), p=self.deterministic_model[state])) == int(np.argmax(self.deterministic_model[state]))
        action = int(np.argmax(self.deterministic_model[state]))
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
        deterministic_model_M: int = 10,
        max_iter: int = 200):
    agent = CrossEntropyAgent(action_n=action_n, state_n=state_n)

    for n in range(iterations_N):
        mean_rewards_over_deterministic_models = []
        pool_of_trajectories_over_deterministic_models = []
        # policy evaluation
        for m in range(deterministic_model_M):
            agent.sample_deterministic_model()
            trajectories = [get_trajectory(env, agent, max_iter, visualize=False) for _ in range(trajectories_K)]
            mean_reward_over_deterministic_model = np.mean([np.sum(trajectory["rewards"]) for trajectory in trajectories])
            pool_of_trajectories_over_deterministic_models.append(trajectories)
            mean_rewards_over_deterministic_models.append(mean_reward_over_deterministic_model)
        print(f"Iter: {n} | Total reward: {np.mean(mean_rewards_over_deterministic_models)}")  # в график в зависимости от итерации

        # policy improvement
        gamma_quantile = np.quantile(mean_rewards_over_deterministic_models, quantile_param)
        elite_trajectories = []
        for trajectory, total_reward in zip(pool_of_trajectories_over_deterministic_models, mean_rewards_over_deterministic_models):
            if total_reward > gamma_quantile:
                elite_trajectories.extend(trajectory)

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

    quantile_param = 0.6
    max_iter = 1000
    iterations_N = 10
    trajectories_K = 100
    deterministic_model_M = 10

    cross_entropy_agent_model(env=env,
                              state_n=state_n,
                              action_n=action_n,
                              quantile_param=quantile_param,
                              iterations_N=iterations_N,
                              trajectories_K=trajectories_K,
                              deterministic_model_M=deterministic_model_M,
                              max_iter=max_iter, )

    # найти хорошее лямбда и сравнить графики с сглаживанием и без (стало лучше или нет)
    # сглаживание лапласа дало рез-т или нет ?
    # сглаживание политики дало рез-т или нет ?

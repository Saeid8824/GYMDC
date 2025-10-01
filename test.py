import pickle
import numpy as np
import os
import random
import torch
from stable_baselines3.common.utils import set_random_seed
import gym


class Calculate_heuristics():
    """CalculatePerformance class to calculate performance of each model"""

    def __init__(self, matching_model_type, observation_model, action_model):
        """to calculate performance of each test we need to consider same random seed"""
        self.matching_model_type = matching_model_type
        self.action_model = action_model
        self.observation_model = observation_model
        self.episode_reward = True
        self.reward_upper_bound = 1000
        self.reward_lower_bound = 1

    def reward_step_action_fast_departure(self, random_list):

        reward_list = []
        for element in random_list:
            seed_value = element
            env = gym.make('gym_DC:basic-v0', fix_seed=True, action_model=self.action_model,
                           observation_model=self.observation_model, episode_reward=self.episode_reward,
                           seed=seed_value,
                           matching_model_type=self.matching_model_type,
                           reward_estimator="calculate_performance_simple_function",
                           reward_upper_bound=self.reward_upper_bound,
                           reward_lower_bound=self.reward_lower_bound).env

            env.reset()
            profit = 0
            is_done = False
            while not is_done:
                action = [0, 1, 0, 0, 0, 1, 0]
                next_state, rewards, is_done, info = env.step(action)
                profit = rewards + profit
            reward_list.append(profit)
        return reward_list

    def generate_heuristics(self, random_list):
        random_list_temp = random_list.copy()
        fast = self.reward_step_action_fast_departure(random_list_temp)

        return fast

    def seed_everything(self, seed, CPU):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if CPU:
            set_random_seed(seed=seed, using_cuda=False)
        else:
            set_random_seed(seed=seed, using_cuda=True)

    def check_log(self):
        CPU = True
        self.seed_everything(123, CPU)
        reward = {}
        increasing1 = [31547255, 35781137, 33440440, 34853038, 5011161, 47582526, 9957037, 42189407, 27955584, 49895682]
        reward["increasing1"] = self.generate_heuristics(increasing1)
        print(reward)


abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

calculator = Calculate_heuristics(matching_model_type="type_22",
                                  observation_model="observation_3",
                                  action_model="model_9")
calculator.check_log()
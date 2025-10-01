"""distribution-centre gym module to define environment"""
import random
import datetime
from datetime import timedelta
import gym
from gym import spaces
import numpy as np
from gym_DC.envs.simulator.simulation_parameters import SimulationDataParameters
from gym_DC.envs.simulator.simulator import DistributionCentre
from .reward_value import Reward_Value
from .observation import ObservationSwitch
from .action import ActionModelSwitch, ActionValueSwitch
import pickle
import os
import copy

class BasicDCEnv(gym.Env):
    """BasicDCEnv class to implement a basic version of distribution centre environment"""
    metadata = {'render.modes': ['human']}

    def __init__(self, fix_seed, action_model, observation_model, episode_reward,matching_model_type, seed ,reward_estimator,reward_upper_bound,reward_lower_bound, save_log= False):
        self.simulator = None
        self.save_log = save_log
        self.save_index = 0
        self.reward_estimator=reward_estimator
        self.reward_upper_bound= reward_upper_bound
        self.reward_lower_bound=reward_lower_bound
        input_dict = self.read_config_file()
        self.input_dict = input_dict
        self.num_packets = input_dict['num_packets']
        self.num_sp = input_dict['num_sp']
        self.num_days = input_dict['num_days']
        self.num_departures_sp = input_dict['num_departures_sp']
        self.num_destination = input_dict['num_destination']
        self.budget = input_dict['budget']
        self.penalty = input_dict['penalty']
        self.hyper_parameters = input_dict['hyper_parameters']
        self.start_date = input_dict['start_date']
        self.loading_to_storage_time = input_dict['loading_to_storage_time']
        self.window_time_for_service_providers = input_dict['window_time_for_service_providers']
        self.fixed_seed = False
        self.seed = seed
        self.matching_model_type = matching_model_type
        self.epsilon = input_dict['epsilon']
        self.packet_rate = input_dict['packet_rate']
        self.simulation_data_parameters = SimulationDataParameters(
            self.start_date,
            input_dict['mean_packet_weight'],
            input_dict['stddev_packet_weight'],
            input_dict['mean_packet_budget'],
            input_dict['stddev_packet_budget'],
            input_dict['mean_packet_penalty_late'],
            input_dict['stddev_packet_penalty_late'],
            input_dict['mean_packet_penalty_unmatched'],
            input_dict['stddev_packet_penalty_unmatched'],
            input_dict['mean_packet_time_to_deadline'],
            input_dict['stddev_packet_time_to_deadline'],
            input_dict['mean_packet_notification_time'],
            input_dict['stddev_packet_notification_time'],
            input_dict['mean_service_provider_price_per_kg'],
            input_dict['stddev_service_provider_price_per_kg'],
            input_dict['mean_service_provider_delivery_time'],
            input_dict['stddev_service_provider_delivery_time'],
            input_dict['loss_service_provider_max'],
            input_dict['loss_service_provider_min'],
            input_dict['mean_service_provider_capacity'],
            input_dict['stddev_service_provider_capacity'],
            input_dict['mean_storage_price_per_kg'],
            input_dict['stddev_storage_price_per_kg'],
            input_dict['mean_storage_capacity'],
            input_dict['stddev_storage_capacity'])
        self.list_of_times=[]
        self.simulation_time = self.input_dict['start_date'] + self.add_random_days(self.input_dict['num_days'])
        # Preparing Data
        self.max_episode_steps = (self.num_sp * self.num_departures_sp)
        self.var_directory = "/temp/"
        if fix_seed:
            random.seed(self.seed)
            np.random.seed(self.seed)
            self.fixed_seed = True
        """If action space is discrete , then we have to set the following variable"""

        action_switch = ActionModelSwitch()
        self.action_model  = action_model
        self.discrete_space, self.action_space = action_switch.action_space(action_model)
        """if we want to return reward in each episode instead of each time step , then we need to set the following variable"""
        if episode_reward:
            self.episode_reward = True
        else:
            self.episode_reward = False

        observation_switch = ObservationSwitch()
        self.observation_model , self.observation_space = observation_model , observation_switch.observation_space(observation_model)



    @staticmethod
    def read_config_file():
        """read config file to get simulation parameters"""
        # read contents of config file
        lines = []
        with open("config") as file_stream:
            for line in file_stream:
                lines.append(line)
        # create dictionary
        input_dict = {}
        # extract elements from list
        for i in range(8):
            (key, val) = lines[i].split()
            input_dict[key] = int(val)
        input_dict['start_date'] = datetime.datetime.strptime(
            str(lines[8][11:-1]), '%Y-%m-%d %H:%M:%S')
        for i in range(9, 25):
            (key, val) = lines[i].split()
            input_dict[key] = int(val)
        (key, val) = lines[25].split()
        input_dict[key] = float(val)
        (key, val) = lines[26].split()
        input_dict[key] = float(val)
        for i in range(27, 36):
            (key, val) = lines[i].split()
            input_dict[key] = int(val)
        (key, val) = lines[36].split()
        input_dict[key] = float(val)
        (key, val) = lines[37].split()
        input_dict[key] = val
        return input_dict

    def update_time_list(self):
        self.list_of_times.clear()
        self.list_of_times = self.simulator.packets_arrival_times + self.simulator.packets_deadline_times + self.simulator.packets_loading_times + self.simulator.packets_notification_times + self.simulator.service_providers_departures_times + self.simulator.packets_departure_times +  self.simulator.service_providers_in_window_times
        self.list_of_times.sort(reverse=True)

    def check_events(self, time):
        if len(self.simulator.packets_arrival_times):
            if time == self.simulator.packets_arrival_times[-1]:
                self.simulator.packets_arrival_event(self.simulator.packets_arrival_times.pop())
                self.update_time_list()
                return True
        if len(self.simulator.packets_deadline_times):
            if time ==self.simulator.packets_deadline_times[-1]:
                self.simulator.packets_deadline_event(self.simulator.packets_deadline_times.pop())
                self.update_time_list()
                return True
        if len(self.simulator.packets_loading_times):
            if time ==self.simulator.packets_loading_times[-1]:
                self.simulator.packets_loading_event(self.simulator.packets_loading_times.pop())
                self.update_time_list()
                return True
        if len(self.simulator.packets_departure_times):
            if time ==self.simulator.packets_departure_times[-1]:
                self.simulator.packets_departure_event(self.simulator.packets_departure_times.pop())
                self.update_time_list()
                return True
        if len(self.simulator.service_providers_in_window_times):
            if time ==self.simulator.service_providers_in_window_times[-1]:
                self.simulator.service_providers_in_window_event(self.simulator.service_providers_in_window_times.pop())
                self.update_time_list()
                return True
        return False

    def check_simulator_events(self):
        result = True
        while (self.simulation_time > self.list_of_times[-1]) and result:
            result = self.check_events(self.list_of_times[-1])

    def check_time_list(self):
        """we have to check list of times to run the next event"""
        self.check_simulator_events()
        if len(self.simulator.service_providers_departures_times):
            if self.list_of_times[-1] == self.simulator.service_providers_departures_times[-1]:
                #save statue before departure
                if self.save_log:
                    self.store_packets_sps_info(self.save_index)
                    self.save_index += 1
                self.simulator.service_providers_departures_event(
                    self.simulator.service_providers_departures_times.pop())
                self.list_of_times.pop()
                self.update_time_list()
                if self.save_log:
                    self.store_packets_sps_info(self.save_index)
                    self.save_index += 1

        self.check_simulator_events()
        if self.save_log:
            self.store_packets_sps_info(self.save_index)
            self.save_index += 1

    def write_var_to_file(self,file_name, var):
        cwd = os.getcwd()
        with open( cwd +"/"+ self.var_directory +str(file_name) ,'wb') as handle:
            pickle.dump(var, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ## create data  #################################################
    def store_packets_sps_info(self,index):
        info_dic = {}
        info_dic["packets_list"] = copy.deepcopy(self.simulator.packets_list)
        info_dic["successful_delivered_packets"] = copy.deepcopy(self.simulator.successful_delivered_packets)
        info_dic["unmatched_entrance_packets_list"] = copy.deepcopy(self.simulator.unmatched_entrance_packets_list)
        info_dic["unmatched_storage_packets_list"] = copy.deepcopy(self.simulator.unmatched_storage_packets_list)
        info_dic["unmatched_dropped_packets_list"] = copy.deepcopy(self.simulator.unmatched_dropped_packets_list)
        info_dic["storage_packets_list"] = copy.deepcopy(self.simulator.storage_packets_list)
        info_dic["matched_packets_list"] = copy.deepcopy(self.simulator.matched_packets_list)
        info_dic["sp_list"] = copy.deepcopy(self.simulator.service_providers_list)
        info_dic["matching_service_providers"] = copy.deepcopy(self.simulator.matching_service_providers)
        info_dic["service_providers_loss_values"] = copy.deepcopy(self.simulator.service_providers_loss_values)
        info_dic["service_providers_departed"] = copy.deepcopy(self.simulator.service_providers_departed)

        info_dic["evaluated_paired_packets"] = copy.deepcopy(self.simulator.evaluated_paired_packets)
        info_dic["evaluated_matched_packets"] = copy.deepcopy(self.simulator.evaluated_matched_packets)
        info_dic["evaluated_dropped_packets"] = copy.deepcopy(self.simulator.evaluated_dropped_packets)
        info_dic["evaluated_storage_packets"] = copy.deepcopy(self.simulator.evaluated_storage_packets)

        self.write_var_to_file(index, info_dic)

    @staticmethod
    def getDuration(start_date, time, interval="default"):
        # Returns a duration as specified by variable interval
        # Functions, except totalDuration, returns [quotient, remainder]
        duration = time - start_date  # For build-in functions
        duration_in_s = duration.total_seconds()

        def years():
            return divmod(duration_in_s, 31536000)  # Seconds in a year=31536000.

        def days(seconds=None):
            return divmod(seconds if seconds != None else duration_in_s, 86400)  # Seconds in a day = 86400

        def hours(seconds=None):
            return divmod(seconds if seconds != None else duration_in_s, 3600)  # Seconds in an hour = 3600

        def minutes(seconds=None):
            return divmod(seconds if seconds != None else duration_in_s, 60)  # Seconds in a minute = 60

        def seconds(seconds=None):
            if seconds != None:
                return divmod(seconds, 1)
            return duration_in_s

        def totalDuration():
            y = years()
            d = days(y[1])  # Use remainder to calculate next variable
            h = hours(d[1])
            m = minutes(h[1])
            s = seconds(m[1])
            return "Time between dates: {} years, {} days, {} hours, {} minutes and {} seconds".format(int(y[0]),
                                                                                                       int(d[0]),
                                                                                                       int(h[0]),
                                                                                                       int(m[0]),
                                                                                                       int(s[0]))

        return {
            'years': int(years()[0]),
            'days': int(days()[0]),
            'hours': int(hours()[0]),
            'minutes': int(minutes()[0]),
            'seconds': int(seconds()),
            'default': totalDuration()
        }[interval]

    @staticmethod
    def add_random_days(number_of_days):
        """add number of days to time """
        time = timedelta(days=number_of_days)
        return time

    @staticmethod
    def normal_function(value, a, b, c, d):
        if (b-a) == 0 :
            return 0
        if value>b:
            return d
        if value <a:
            return c
        result = c + ((d - c) / (b - a)) * (value - a)
        return result

    # calculating performance for the episode
    def step(self, action):
        """we have to check time step in this function"""
        "Information for reward and actions"
        a2 = (-1) * (self.input_dict['num_packets']) * ((self.input_dict['mean_packet_penalty_unmatched'] + (4 * self.input_dict['stddev_packet_penalty_unmatched'])))
        # we should consider the upper bound
        b2 = (self.input_dict['num_packets']) * (self.input_dict['mean_packet_weight'] + (4 * self.input_dict['stddev_packet_weight'])) * (self.input_dict['mean_packet_budget'] + (4 * self.input_dict['stddev_packet_budget']) - (self.input_dict['mean_service_provider_price_per_kg']))

        c2 = self.reward_lower_bound  # reward_lower_bound
        d2 = self.reward_upper_bound  # reward_upper_bound
        """if action space is discrete, then we have to change action value"""
        reward=0



        if len(self.list_of_times):
            action_value_switch = ActionValueSwitch()
            self.simulator.hyper_parameters = action_value_switch.action_value(self.action_model, action)

            # Run departure time
            self.check_time_list()
            state = self.observation()
            done = len(self.simulator.service_providers_departures_times)==0
            info = {'arrived_packets_list': len(self.simulator.arrived_packets_list),
                    'storage_packets_list': len(self.simulator.storage_packets_list),
                    'matched_packets_list': len(self.simulator.matched_packets_list),
                    'sent_packets_list': len(self.simulator.sent_packets_list),
                    'unmatched_entrance_packets_list': len(self.simulator.unmatched_entrance_packets_list),
                    'unmatched_storage_packets_list': len(self.simulator.unmatched_storage_packets_list),
                    'successful_delivered_packets': len(self.simulator.successful_delivered_packets),
                    'unsuccessful_delivered_packets': len(self.simulator.unsuccessful_delivered_packets),
                    'budget': self.simulator.budget, 'penalty': self.simulator.penalty}  # update notifications
            if done:
                reward_estimator = Reward_Value(True, self.simulator, self.input_dict,
                                                self.reward_estimator, a2, b2, c2, d2)
                reward = reward_estimator.reward_value()
                if self.save_log:
                    self.store_packets_sps_info(self.save_index)
                    self.save_index += 1
            else:
                reward = 0
            return state, reward, done, info



    def reset(self):
        """create distribution center object in this method"""
        self.simulator = DistributionCentre(
            self.num_packets,
            self.num_sp,
            self.num_days,
            self.num_departures_sp,
            self.num_destination,
            self.budget,
            self.penalty,
            self.hyper_parameters,
            self.window_time_for_service_providers,
            self.loading_to_storage_time,
            self.simulation_data_parameters, self.packet_rate,self.fixed_seed , self.seed, self.matching_model_type)
        self.simulator.generating_data(self.hyper_parameters)


        self.update_time_list()
        state = self.observation()
        # in this scenario we want to calculate the performance based on the results of packets. we only do simulation based on the half of the packets
        ########################################################
        sp_list = []
        for key in self.simulator.service_providers_list.keys():
            sp_list.append(self.simulator.service_providers_list[key])
        sp_list.sort(key=lambda x: x.departure_time, reverse=True)
        # find the sp departed in the middle of simulation time
        ########################################################

        return state

    def render(self, mode='human'):
        """render method in the gym framework methods"""

    def close(self):
        """close method in the gym framework methods"""

    def observation(self):
        state = 0
        if self.observation_model=="observation_1":
            state = self.observation_model_1()
        elif self.observation_model=="observation_2":
            state =self.observation_model_2()
        elif self.observation_model=="observation_3":
            state = self.observation_model_3()
        elif self.observation_model=="observation_4":
            state = self.observation_model_4()
        return state

    def observation_model_1(self):
        """observation method in the gym framework methods"""
        """we assumed that matching packets are only storage packets"""
        """trend_list"""
        trend_list = []
        for i in range(len(self.simulator.packets_trend)):
            trend_list.append(i)
        if len(self.simulator.packets_trend) > 3:
            trend, _line_parameter = np.polyfit(trend_list, self.simulator.packets_trend, 1)
        else:
            trend = 0
        if trend<0:
            trend_value = 0
        else:
            trend_value = 1
        """capacity"""
        capacity = self.simulator.storage.capacity
        storage_capacity_value = int(self.normal_function(capacity, 0, self.simulator.full_capacity, 0, 99))
        """rate_of_packets"""
        # total weight of packets / capacity of service providers
        # we assumed that matching packets are only storage packets
        total_weight, sp_capacity, packet_sp_rate_value = 0 ,0 , 0
        for key in self.simulator.storage_packets_list:
            total_weight += self.simulator.storage_packets_list[key].weight
        # for SP in self.matching_service_providers:
        for key, service_provider in self.simulator.matching_service_providers.items():
            sp_capacity += service_provider.capacity
        if sp_capacity ==0:
            rate =20
        else:
            rate = total_weight / sp_capacity
        if rate>10:
            packet_sp_rate_value = 9
        elif rate > 1 and rate <10:
            packet_sp_rate_value = int(self.normal_function(rate, 1, 10, 5, 9))
        elif rate<1:
            packet_sp_rate_value = int(self.normal_function(rate, 0, 1, 0, 4))

        observation = (trend_value * 1000) + (storage_capacity_value * 10) + (packet_sp_rate_value)
        observation = np.array([observation])
        return observation

    def observation_model_2(self):
        """observation method in the gym framework methods"""
        """we assumed that matching packets are only storage packets"""
        """trend_list"""
        trend_list = []
        for i in range(len(self.simulator.packets_trend)):
            trend_list.append(i)
        if len(self.simulator.packets_trend) > 3:
            trend, _line_parameter = np.polyfit(trend_list, self.simulator.packets_trend, 1)
        else:
            trend = 0
        if trend<0:
            trend_value = 0
        else:
            trend_value = 1
        """capacity"""
        capacity = self.simulator.storage.capacity
        storage_capacity_value = int(self.normal_function(capacity, 0, self.simulator.full_capacity, 0, 999))
        """rate_of_packets"""
        return storage_capacity_value

    def observation_model_3(self):
        """observation method in the gym framework methods"""
        """we assumed that matching packets are only storage packets"""
        """trend_list"""

        """capacity"""
        capacity = self.simulator.storage.capacity
        storage_capacity_value = capacity/self.simulator.full_capacity


        """rate_of_packets"""
        # total weight of packets / capacity of service providers
        # we assumed that matching packets are only storage packets
        profit, penalty = self.simulator.evaluate_greedy_action()
        if penalty>= profit:
            if profit == 0:
                rate = 0
            else:
                rate = (profit / penalty)
        else:
            if penalty== 0:
                rate = 1
            else:
                rate = (penalty /profit ) +1

        observation = [storage_capacity_value, rate]
        return observation

    def observation_model_4(self):
        """observation method in the gym framework methods"""
        """we assumed that matching packets are only storage packets"""
        """trend_list"""

        """capacity"""
        capacity = self.simulator.storage.capacity
        storage_capacity_value = capacity/self.simulator.full_capacity

        rate = 0
        """rate_of_packets"""
        # total weight of packets / capacity of service providers
        # we assumed that matching packets are only storage packets
        total_weight, sp_capacity, packet_sp_rate_value = 0, 0, 0
        for key in self.simulator.storage_packets_list:
            total_weight += self.simulator.storage_packets_list[key].weight
        # for SP in self.matching_service_providers:
        for key, service_provider in self.simulator.matching_service_providers.items():
            sp_capacity += service_provider.capacity

        if total_weight>sp_capacity:
            rate = 1
        elif sp_capacity==0:
            rate =0
        else:
            rate = total_weight/sp_capacity




        observation = [storage_capacity_value, rate]
        return observation

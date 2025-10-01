"""
Calculates the final episode reward using various estimation strategies.

This module provides the Reward_Value class, which is responsible for calculating
the total reward at the end of a simulation episode. It includes multiple methods
to estimate the value of packets that remain in the system (i.e., not yet
delivered or failed), as simply ignoring them would provide an incomplete performance
measure. These methods range from simple heuristics to machine learning-based predictions.
"""
from datetime import timedelta
import numpy as np
# Import various classifiers from scikit-learn for reward estimation.
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from collections import OrderedDict, deque

class Reward_Value():
    """
    A class dedicated to calculating the final reward for a simulation episode.
    """

    def __init__(self, return_episode_reward, simulator, input_dict, reward_estimator, a, b, c, d):
        """
        Initializes the Reward_Value calculator.

        Args:
            return_episode_reward (bool): Flag indicating if reward is calculated at the end of the episode.
            simulator (DistributionCentre): The main simulator object, containing the final state.
            input_dict (dict): The initial configuration dictionary for the simulation.
            reward_estimator (str): The name of the method to use for reward calculation.
            a (float): The lower bound of the input profit range for normalization.
            b (float): The upper bound of the input profit range for normalization.
            c (float): The lower bound of the output reward range (e.g., -1).
            d (float): The upper bound of the output reward range (e.g., +1).
        """
        self.return_episode_reward = return_episode_reward
        self.simulator = simulator
        self.input_dict = input_dict
        self.reward_estimator = reward_estimator
        # Parameters for the normalization function.
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    @staticmethod
    def getDuration(start_date, time, interval="default"):
        """
        Calculates the time duration between two datetime objects in various units.

        Args:
            start_date (datetime): The starting datetime.
            time (datetime): The ending datetime.
            interval (str): The desired unit ('years', 'days', 'hours', 'minutes', 'seconds').

        Returns:
            int or str: The calculated duration in the specified unit.
        """
        duration = time - start_date
        duration_in_s = duration.total_seconds()

        # Helper functions to calculate duration in different units.
        def years():
            return divmod(duration_in_s, 31536000)
        def days(seconds=None):
            return divmod(seconds if seconds is not None else duration_in_s, 86400)
        def hours(seconds=None):
            return divmod(seconds if seconds is not None else duration_in_s, 3600)
        def minutes(seconds=None):
            return divmod(seconds if seconds is not None else duration_in_s, 60)
        def seconds(seconds=None):
            return seconds if seconds is not None else duration_in_s
        def totalDuration():
            y = years()
            d = days(y[1])
            h = hours(d[1])
            m = minutes(h[1])
            s = divmod(m[1], 1)
            return f"Time between dates: {int(y[0])} years, {int(d[0])} days, {int(h[0])} hours, {int(m[0])} minutes and {int(s[0])} seconds"

        # Dictionary to map interval string to the corresponding function call.
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
        """Creates a timedelta object for a given number of days."""
        time = timedelta(days=number_of_days)
        return time

    @staticmethod
    def normal_function(value, a, b, c, d):
        """
        Performs min-max normalization to scale a value from one range [a, b] to another [c, d].
        
        Args:
            value (float): The input value to scale.
            a (float): The minimum of the input range.
            b (float): The maximum of the input range.
            c (float): The minimum of the output range.
            d (float): The maximum of the output range.
            
        Returns:
            float: The scaled value.
        """
        # Avoid division by zero if the input range is zero.
        if (b - a) == 0:
            return 0
        # Clip values that are outside the input range.
        if value > b:
            return d
        if value < a:
            return c
        # Perform linear interpolation.
        result = c + ((d - c) / (b - a)) * (value - a)
        return result

    def calculate_profit_without_remained_packets(self):
        """
        Calculates the net profit from packets that have completed their journey
        (delivered or failed), ignoring any packets still in the system.
        """
        # Get initial financial state.
        first_budget = self.input_dict['budget']
        first_penalty = self.input_dict['penalty']
        # Get final financial state from the simulator.
        last_budget = self.simulator.budget
        last_penalty = self.simulator.penalty
        # Calculate the net profit for the episode.
        profit = (last_budget - first_budget) - (last_penalty - first_penalty)
        return profit

    def calculate_performance_half_packets(self):
        """
        Calculates performance based only on packets that arrived before the
        midpoint of the simulation (defined by the median departure time of SPs).
        This is a heuristic to get a performance measure before all packets are processed.
        """
        # Create a sorted list of all departed service providers.
        sp_list = sorted(self.simulator.service_providers_departed, key=lambda x: x.departure_time, reverse=True)
        
        # Find the departure time of the service provider at the midpoint of the simulation.
        stop_time = sp_list[int(len(sp_list) / 2)].departure_time
        
        # Initialize budget and penalty counters.
        budget = 0
        penalty = 0

        # Sum profit only for successful packets that arrived before the stop_time.
        for packet in self.simulator.successful_delivered_packets:
            if packet.arrival < stop_time:
                price_per_kg = self.simulator.service_providers_loss_values[packet.status].price_per_kg
                profit = packet.budget - (packet.weight * price_per_kg)
                budget += profit

        # Sum penalty for failed packets that arrived before the stop_time.
        for packet in self.simulator.unmatched_storage_packets_list:
            if packet.arrival < stop_time:
                penalty += packet.penalty_unmatched
        for packet in self.simulator.unmatched_entrance_packets_list:
            if packet.arrival < stop_time:
                penalty += packet.penalty_unmatched

        # Calculate the net profit based on this subset of packets.
        net_profit = budget - penalty
        
        # Normalize the profit to get the final reward.
        reward = self.normal_function(net_profit, self.a, self.b, self.c, self.d)
        return reward

    def calculate_performance_ignore_remained(self):
        """
        Calculates reward based only on processed packets, ignoring all remaining ones.
        This is the simplest but least accurate method.
        """
        # Calculate the raw profit from processed packets.
        profit = self.calculate_profit_without_remained_packets()
        # Normalize the profit to get the reward.
        reward = self.normal_function(profit, self.a, self.b, self.c, self.d)
        return reward

    def calculate_average_profit(self):
        """Calculates the average profit per successfully delivered packet."""
        initial_budget = self.input_dict['budget']
        final_budget = self.simulator.budget
        num_successful = len(self.simulator.successful_delivered_packets)
        # Avoid division by zero.
        if num_successful > 0:
            return (final_budget - initial_budget) / num_successful
        else:
            return 0

    def calculate_average_penalty(self):
        """Calculates the average penalty per failed packet."""
        initial_penalty = self.input_dict['penalty']
        final_penalty = self.simulator.penalty
        num_failed = len(self.simulator.unmatched_storage_packets_list) + len(self.simulator.unmatched_entrance_packets_list)
        # Avoid division by zero.
        if num_failed > 0:
            return (final_penalty - initial_penalty) / num_failed
        else:
            return 0

    def calculate_performance_expire_day(self):
        """
        Estimates the value of remaining packets by applying a scaled penalty
        based on how close they are to their deadline. Packets closer to expiring
        incur a higher portion of their unmatched penalty.
        """
        extra_penalty = 0
        # Determine the end time of the simulation.
        last_date = self.input_dict['start_date'] + self.add_random_days(self.input_dict['num_days'])
        
        # Iterate through packets still in storage.
        for key, packet in self.simulator.storage_packets_list.items():
            # Calculate days remaining until the packet's deadline.
            days_remaining = self.getDuration(last_date, key, interval="days")
            # Apply a penalty scaled by the urgency.
            if days_remaining <= 1:
                extra_penalty += packet.penalty_unmatched
            elif days_remaining == 2:
                extra_penalty += (packet.penalty_unmatched * 4) / 5
            elif days_remaining == 3:
                extra_penalty += (packet.penalty_unmatched * 3) / 5
            elif days_remaining == 4:
                extra_penalty += (packet.penalty_unmatched * 2) / 5
            elif days_remaining == 5:
                extra_penalty += (packet.penalty_unmatched * 1) / 5
                
        # Calculate final profit by subtracting the estimated extra penalty.
        profit = self.calculate_profit_without_remained_packets() - extra_penalty
        # Normalize to get the reward.
        reward = self.normal_function(profit, self.a, self.b, self.c, self.d)
        return reward

    def calculate_performance_average_value(self):
        """
        Estimates the value of remaining packets by assuming they will achieve
        the average outcome (profit or penalty) of already processed packets.
        """
        # Calculate average profit and penalty from completed packets.
        average_profit = self.calculate_average_profit()
        average_penalty = self.calculate_average_penalty()
        
        # The opportunity cost of a packet in storage is the average profit minus average penalty.
        opportunity_cost = average_profit - average_penalty
        
        # Calculate the total estimated penalty for all packets still in storage.
        extra_penalty = len(self.simulator.storage_packets_list) * opportunity_cost
        
        # Calculate the final profit.
        profit = self.calculate_profit_without_remained_packets() - extra_penalty
        # Normalize to get the reward.
        reward = self.normal_function(profit, self.a, self.b, self.c, self.d)
        return reward

    def create_list_of_training(self):
        """
        Prepares training data (X, y) from processed packets to be used by ML classifiers.
        Features include budget, weight, destination, and time-to-deadline.
        Labels are 1 for success, 0 for failure.
        """
        X = [] # Features
        y = [] # Labels
        
        # Add successfully delivered packets to the dataset (label=1).
        for packet in self.simulator.successful_delivered_packets:
            time_to_deadline = self.getDuration(packet.arrival, packet.deadline, interval="days")
            sample = [packet.budget, packet.weight, packet.destination, time_to_deadline]
            X.append(sample)
            y.append(1)
            
        # Add failed (unmatched) packets to the dataset (label=0).
        for packet in self.simulator.unmatched_storage_packets_list:
            time_to_deadline = self.getDuration(packet.arrival, packet.deadline, interval="days")
            sample = [packet.budget, packet.weight, packet.destination, time_to_deadline]
            X.append(sample)
            y.append(0)
            
        # Calculate the average profit for use in prediction.
        average_profit = self.calculate_average_profit()
        
        # Convert lists to numpy arrays.
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        return X, y, average_profit

    def calculate_performance_Decision_Tree(self):
        """
        Estimates the value of remaining packets by training a Decision Tree classifier
        on processed packets and using it to predict the outcome of the remaining ones.
        """
        # Generate training data from processed packets.
        X, y, average_profit = self.create_list_of_training()
        extra_penalty = 0
        
        # Only proceed if there is data to train on.
        if len(X) > 0:
            # Initialize and train the classifier.
            # NOTE: Potential bug here. X is 4-dimensional, but X.reshape(-1, 1) collapses it.
            clf = DecisionTreeClassifier().fit(X, y)
            
            # Predict outcomes for packets in 'arrived' and 'storage' lists.
            for packet_list in [self.simulator.arrived_packets_list, self.simulator.storage_packets_list]:
                for packet in packet_list.values():
                    time_to_deadline = self.getDuration(packet.arrival, packet.deadline, interval="days")
                    sample = np.array([packet.budget, packet.weight, packet.destination, time_to_deadline])
                    
                    # Predict the outcome (0 for fail, 1 for success).
                    # NOTE: Potential bug - predict expects a 2D array.
                    result = clf.predict(sample.reshape(1, -1))
                    
                    # Adjust the extra penalty based on the prediction.
                    if result == 0:
                        extra_penalty += packet.penalty_unmatched
                    else:
                        extra_penalty -= average_profit
                        
        # Calculate final profit and normalize for the reward.
        profit = self.calculate_profit_without_remained_packets() - extra_penalty
        reward = self.normal_function(profit, self.a, self.b, self.c, self.d)
        return reward

    # Note: The following ML-based methods (KNeighbors, MLP, RandomForest) follow the
    # same pattern as the Decision Tree method, just with different classifiers.
    # Comments will be abbreviated for brevity.

    def calculate_performance_K_Neighbors(self):
        """Estimates reward using a K-Nearest Neighbors classifier."""
        X, y, average_profit = self.create_list_of_training()
        extra_penalty = 0
        if len(X) > 0:
            # NOTE: Potential bug with reshape.
            knn = KNeighborsClassifier().fit(X, y)
            for packet_list in [self.simulator.arrived_packets_list, self.simulator.storage_packets_list]:
                for packet in packet_list.values():
                    time_to_deadline = self.getDuration(packet.arrival, packet.deadline, interval="days")
                    sample = np.array([packet.budget, packet.weight, packet.destination, time_to_deadline])
                    result = knn.predict(sample.reshape(1, -1))
                    if result == 0:
                        extra_penalty += packet.penalty_unmatched
                    else:
                        extra_penalty -= average_profit
        profit = self.calculate_profit_without_remained_packets() - extra_penalty
        reward = self.normal_function(profit, self.a, self.b, self.c, self.d)
        return reward

    def calculate_performance_MLPClassifier(self):
        """Estimates reward using a Multi-Layer Perceptron classifier."""
        X, y, average_profit = self.create_list_of_training()
        extra_penalty = 0
        if len(X) > 0:
            # This fit call is correct (does not reshape X).
            mlpc = MLPClassifier().fit(X, y)
            for packet_list in [self.simulator.arrived_packets_list, self.simulator.storage_packets_list]:
                for packet in packet_list.values():
                    time_to_deadline = self.getDuration(packet.arrival, packet.deadline, interval="days")
                    sample = np.array([packet.budget, packet.weight, packet.destination, time_to_deadline])
                    result = mlpc.predict(sample.reshape(1, -1))
                    if result == 0:
                        extra_penalty += packet.penalty_unmatched
                    else:
                        extra_penalty -= average_profit
        profit = self.calculate_profit_without_remained_packets() - extra_penalty
        reward = self.normal_function(profit, self.a, self.b, self.c, self.d)
        return reward

    def calculate_performance_RandomForest(self):
        """Estimates reward using a Random Forest classifier."""
        X, y, average_profit = self.create_list_of_training()
        extra_penalty = 0
        if len(X) > 0:
            # NOTE: Potential bug with reshape.
            rfor = RandomForestClassifier().fit(X, y)
            for packet_list in [self.simulator.arrived_packets_list, self.simulator.storage_packets_list]:
                for packet in packet_list.values():
                    time_to_deadline = self.getDuration(packet.arrival, packet.deadline, interval="days")
                    sample = np.array([packet.budget, packet.weight, packet.destination, time_to_deadline])
                    result = rfor.predict(sample.reshape(1, -1))
                    if result == 0:
                        extra_penalty += packet.penalty_unmatched
                    else:
                        extra_penalty -= average_profit
        profit = self.calculate_profit_without_remained_packets() - extra_penalty
        reward = self.normal_function(profit, self.a, self.b, self.c, self.d)
        return reward

    def calculate_destination_information(self):
        """
        Analyzes departed service providers to calculate historical performance
        metrics (min/avg price and delivery time) for each destination.
        """
        # Initialize lists to hold data for each destination.
        num_dest = self.input_dict['num_destination'] + 1
        dest_min_budget = [[] for _ in range(num_dest)]
        dest_avg_budget = [[] for _ in range(num_dest)]
        dest_min_delivery = [[] for _ in range(num_dest)]
        dest_avg_delivery = [[] for _ in range(num_dest)]

        # Aggregate data from all departed service providers.
        for sp in self.simulator.service_providers_departed:
            for dest in sp.destination:
                dest_avg_budget[dest].append(sp.price_per_kg)
                delivery_days = self.getDuration(sp.departure_time, sp.delivery_time, interval="days")
                dest_avg_delivery[dest].append(delivery_days)

        # Calculate final metrics, using simulation-wide averages as a fallback.
        mean_price = self.input_dict['mean_service_provider_price_per_kg']
        mean_delivery = self.input_dict['mean_service_provider_delivery_time']
        
        final_min_budget = [min(d) if d else mean_price for d in dest_avg_budget]
        final_avg_budget = [sum(d)/len(d) if d else mean_price for d in dest_avg_budget]
        final_min_delivery = [min(d) if d else mean_delivery for d in dest_avg_delivery]
        final_avg_delivery = [sum(d)/len(d) if d else mean_delivery for d in dest_avg_delivery]
        
        return final_min_budget, final_avg_budget, final_min_delivery, final_avg_delivery

    def calculate_performance_simple_function(self):
        """
        Estimates the value of remaining packets using a simple heuristic function.
        It compares a packet's remaining time with historical delivery times for its
        destination to guess its potential profit or penalty.
        """
        # Get historical performance data for each destination.
        dest_min_budget, dest_avg_budget, dest_min_delivery, dest_avg_delivery = self.calculate_destination_information()
        simulation_end_time = self.input_dict['start_date'] + self.add_random_days(self.input_dict['num_days'])
        extra_penalty = 0

        # Iterate through all lists of remaining packets.
        packet_lists = [
            self.simulator.arrived_packets_list,
            self.simulator.storage_packets_list,
            self.simulator.matched_packets_list
        ]
        for packet_list in packet_lists:
            for packet in packet_list.values():
                time_to_expire = self.getDuration(simulation_end_time, packet.deadline, interval="days")
                dest = packet.destination
                
                # Heuristic logic:
                # If time remaining is less than the fastest-ever delivery, assume it will fail.
                if time_to_expire < dest_min_delivery[dest]:
                    extra_penalty += packet.penalty_unmatched
                # If there is ample time (e.g., >2x fastest), assume it will succeed with average profit.
                elif time_to_expire > (2 * dest_min_delivery[dest]):
                    est_profit = packet.budget - (packet.weight * dest_avg_budget[dest])
                    extra_penalty -= est_profit
                # If time is tight but possible, assume it will succeed with half the average profit.
                elif time_to_expire > dest_min_delivery[dest]:
                    est_profit = packet.budget - (packet.weight * dest_avg_budget[dest])
                    extra_penalty -= est_profit / 2

        # Calculate final profit and normalize.
        profit = self.calculate_profit_without_remained_packets() - extra_penalty
        reward = self.normal_function(profit, self.a, self.b, self.c, self.d)
        return reward

    def reward_value(self):
        """
        Dispatcher method that calls the appropriate reward calculation function
        based on the `reward_estimator` string specified during initialization.
        """
        # This check is somewhat redundant if only called at the end of an episode, but good practice.
        if self.return_episode_reward:
            # A mapping of estimator names to their respective methods.
            estimator_map = {
                "calculate_profit_without_remained_packets": self.calculate_profit_without_remained_packets,
                "calculate_performance_ignore_remained": self.calculate_performance_ignore_remained,
                "calculate_performance_expire_day": self.calculate_performance_expire_day,
                "calculate_performance_average_value": self.calculate_performance_average_value,
                "calculate_performance_Decision_Tree": self.calculate_performance_Decision_Tree,
                "calculate_performance_K_Neighbors": self.calculate_performance_K_Neighbors,
                "calculate_performance_MLPClassifier": self.calculate_performance_MLPClassifier,
                "calculate_performance_RandomForest": self.calculate_performance_RandomForest,
                "calculate_performance_simple_function": self.calculate_performance_simple_function,
                "calculate_performance_half_packets": self.calculate_performance_half_packets,
            }
            # Get the correct function from the map and call it.
            reward_func = estimator_map.get(self.reward_estimator)
            if reward_func:
                return reward_func()
        return 0 # Default reward if not end-of-episode.
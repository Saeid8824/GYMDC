"""
Generates simulated data for packets, service providers, and storage facilities
for the entire simulation period based on specified statistical distributions.
"""
from collections import deque
from datetime import timedelta
from random import normalvariate, randint, uniform, sample
from .packet import Packet
from .service_provider import ServiceProvider
from .storage import Storage
import numpy as np
import random


class GeneratingData():
    """
    A class to generate all necessary simulation data entities.

    This class takes a comprehensive set of simulation parameters to create
    lists of packets and service providers with randomized, yet statistically
    controlled, attributes over a defined time period.
    """

    def __init__(
            self,
            number_of_packets,
            number_of_service_providers,
            number_of_days,
            number_of_departures,
            number_of_destinations,
            simulation_parameters,
            window_time_for_service_providers, packet_rate, fixed_seed, seed):
        """
        Initializes the GeneratingData class with all simulation parameters.

        Args:
            number_of_packets (int): Total number of packets to generate for the simulation.
            number_of_service_providers (int): Total number of unique service provider companies.
            number_of_days (int): The duration of the simulation period in days.
            number_of_departures (int): The number of departure events for each service provider company.
            number_of_destinations (int): The number of possible destination locations.
            simulation_parameters (SimulationDataParameters): A dataclass object containing statistical parameters for data generation.
            window_time_for_service_providers (int): The time window (in days) before departure when a service provider becomes available for matching.
            packet_rate (str): A string defining the distribution of packet arrivals over time (e.g., "uniform", "decreasing").
            fixed_seed (bool): If True, use a fixed seed for random number generation for reproducibility.
            seed (int): The seed value to use if fixed_seed is True.
        """
        # --- Store core simulation numbers ---
        self.number_of_packets = number_of_packets
        self.number_of_service_providers = number_of_service_providers
        self.number_of_days = number_of_days
        self.number_of_departures = number_of_departures
        self.number_of_destinations = number_of_destinations
        self.window_time_for_service_providers = window_time_for_service_providers
        self.packet_rate = packet_rate
        self.start_date = simulation_parameters.start_date

        # --- Unpack Packet Generation Parameters from the parameters object ---
        self.mean_packet_weight = simulation_parameters.mean_packet_weight
        self.stddev_packet_weight = simulation_parameters.stddev_packet_weight
        self.mean_packet_budget = simulation_parameters.mean_packet_budget
        self.stddev_packet_budget = simulation_parameters.stddev_packet_budget
        self.mean_packet_penalty_late = simulation_parameters.mean_packet_penalty_late
        self.stddev_packet_penalty_late = simulation_parameters.stddev_packet_penalty_late
        self.mean_packet_penalty_unmatched = simulation_parameters.mean_packet_penalty_unmatched
        self.stddev_packet_penalty_unmatched = simulation_parameters.stddev_packet_penalty_unmatched
        self.mean_packet_time_to_deadline = simulation_parameters.mean_packet_time_to_deadline
        self.stddev_packet_time_to_deadline = simulation_parameters.stddev_packet_time_to_deadline
        self.mean_packet_notification_time = simulation_parameters.mean_packet_notification_time
        self.stddev_packet_notification_time = simulation_parameters.stddev_packet_notification_time

        # --- Unpack Service Provider Generation Parameters ---
        self.mean_service_provider_price_per_kg = \
            simulation_parameters.mean_service_provider_price_per_kg
        self.stddev_service_provider_price_per_kg = \
            simulation_parameters.stddev_service_provider_price_per_kg
        self.mean_service_provider_delivery_time = \
            simulation_parameters.mean_service_provider_delivery_time
        self.stddev_service_provider_delivery_time = \
            simulation_parameters.stddev_service_provider_delivery_time
        self.loss_service_provider_max = simulation_parameters.loss_service_provider_max
        self.loss_service_provider_min = simulation_parameters.loss_service_provider_min
        self.mean_service_provider_capacity = simulation_parameters.mean_service_provider_capacity
        self.stddev_service_provider_capacity = \
            simulation_parameters.stddev_service_provider_capacity

        # --- Unpack Storage Generation Parameters ---
        self.mean_storage_price_per_kg = simulation_parameters.mean_storage_price_per_kg
        self.stddev_storage_price_per_kg = simulation_parameters.stddev_storage_price_per_kg
        self.mean_storage_capacity = simulation_parameters.mean_storage_capacity
        self.stddev_storage_capacity = simulation_parameters.stddev_storage_capacity

        # --- Configure Random Seed for Reproducibility ---
        self.fixed_seed = fixed_seed
        self.seed = seed
        # If a fixed seed is requested, apply it to both random and numpy libraries.
        if (self.fixed_seed):
            random.seed(self.seed)
            np.random.seed(self.seed)

    @staticmethod
    def generate_data(mean, sigma):
        """
        Generates a single positive integer from a normal distribution.

        The result is rounded to the nearest integer and its absolute value is taken.
        1 is added to ensure the value is at least 1, preventing zero or negative values.

        Args:
            mean (float): The mean of the normal distribution.
            sigma (float): The standard deviation of the normal distribution.

        Returns:
            int: A randomly generated positive integer.
        """
        # Generate a value from a normal distribution.
        data = normalvariate(mean, sigma)
        # Convert to int (truncates), take absolute value, and ensure it's at least 1.
        data = abs(int(data + 0.5)) + 1
        return data

    @staticmethod
    def add_random_days(number_of_days):
        """
        Creates a timedelta object representing a given number of days plus a random time.
        This is used to add variability to datetimes within a specific day.

        Args:
            number_of_days (int): The base number of days for the timedelta.

        Returns:
            timedelta: A timedelta object with the specified days and a random H:M:S.ms offset.
        """
        # Start with the base number of days.
        time = timedelta(days=number_of_days) + \
               timedelta(hours=randint(0, 23)) + \
               timedelta(minutes=randint(0, 59)) + \
               timedelta(seconds=randint(0, 59)) + \
               timedelta(milliseconds=randint(0, 999))
        return time

    def generate_datetime_list(self, number):
        """
        Generates a sorted list of unique datetime objects with a uniform distribution.

        This method ensures the generated list has the exact length specified by 'number'
        by adding or removing items until the count is correct.

        Args:
            number (int): The desired number of unique datetime objects.

        Returns:
            list: A sorted list of unique datetime objects.
        """
        datetime_list = deque()
        # Initial generation of datetimes over the simulation period.
        for _index in range(number):
            days = randint(0, self.number_of_days - 1)
            datetime_list.append(self.start_date + self.add_random_days(days))
        
        # Ensure uniqueness by converting to a dictionary and back to a list.
        datetime_list = list(dict.fromkeys(datetime_list))
        
        # Calculate the difference between the current and desired length.
        difference = len(datetime_list) - number
        
        # Loop until the list has the exact required number of elements.
        while difference != 0:
            if difference > 0:  # List is too long.
                while len(datetime_list) > number:
                    # Randomly delete excess elements.
                    del datetime_list[randint(0, len(datetime_list) - 1)]
            elif difference < 0:  # List is too short.
                # Add new random datetimes to make up for the shortfall.
                for _ in range(-2 * difference): # Add more than needed to speed up uniqueness.
                    days = randint(0, self.number_of_days - 1)
                    datetime_list.append(self.start_date + self.add_random_days(days))
            
            # Re-check for uniqueness and update the difference.
            datetime_list = list(dict.fromkeys(datetime_list))
            difference = len(datetime_list) - number
            
        # Return the final, sorted list of datetimes.
        datetime_list = sorted(datetime_list)
        return datetime_list

    # --- Methods for Generating Packet Rates Based on Different Distributions ---

    def fix_list_length(self, input_list, expected_value, selected_day):
        """
        Adjusts the sum of elements in a list to match an expected total value.
        It randomly increments or decrements list elements until the sum is correct.

        Args:
            input_list (list): The list of integers (packet counts per day) to adjust.
            expected_value (int): The target sum (total number of packets).
            selected_day (int): Not currently used in the logic, but kept for signature consistency.

        Returns:
            list: The adjusted list where the sum of elements equals expected_value.
        """
        # Calculate the difference between the target sum and the current sum.
        difference = expected_value - np.sum(input_list)
        
        # Loop until the sum is correct.
        while difference != 0:
            if difference > 0:  # The sum is too small.
                # Increment random elements in the list to increase the sum.
                for _ in range(difference):
                    x = random.randint(0, len(input_list) - 1)
                    input_list[x] += 1
            if difference < 0:  # The sum is too large.
                # Decrement a random element, ensuring it doesn't go below zero.
                x = random.randint(0, len(input_list) - 1)
                if input_list[x] > 0:
                    input_list[x] -= 1
            # Recalculate the difference for the next iteration.
            difference = expected_value - np.sum(input_list)
        return input_list

    def decreasing_random_numbers(self, number_of_packets, number_of_days, minimum_per_day, selected_day, parameter):
        """
        Generates a list of day assignments for packets, with a decreasing trend over time.
        More packets will arrive on earlier days.

        Args:
            number_of_packets (int): Total number of packets.
            number_of_days (int): Total number of simulation days.
            minimum_per_day (int): The minimum number of packets to be assigned to any day.
            selected_day (int): Unused parameter.
            parameter (float): Parameter for the geometric distribution to shape the trend.

        Returns:
            list: A list where each element is a day number, representing an arrival day.
        """
        # Initialize each day with the minimum number of packets.
        x = np.ones((number_of_days,), dtype=int)
        x = [i * minimum_per_day for i in x]
        
        # Generate numbers from a geometric distribution to create a trend.
        temp = np.random.geometric(parameter, number_of_days * 8)
        # Take a fraction of the generated numbers to control the steepness.
        temp = np.sort(temp)[:len(temp) // 8]
        
        # Add the trend values to the base minimum counts.
        for i in range(len(x)):
            x[i] = x[i] + temp[i]
        
        # Reverse the array to create a decreasing trend (high values first).
        x = x[::-1]
        
        # Adjust the list to ensure the total packet count is exact.
        x = self.fix_list_length(x, number_of_packets, selected_day)
        
        # Sort again to enforce a smooth decreasing order.
        temp = np.sort(x)[::-1]
        
        # Convert the daily counts into a list of day assignments for each packet.
        return self.generate_day_list(temp)

    def increasing_random_numbers(self, number_of_packets, number_of_days, minimum_per_day, selected_day, parameter):
        """
        Generates a list of day assignments for packets, with an increasing trend over time.
        More packets will arrive on later days.
        
        Args:
            number_of_packets (int): Total number of packets.
            number_of_days (int): Total number of simulation days.
            minimum_per_day (int): The minimum number of packets to be assigned to any day.
            selected_day (int): Unused parameter.
            parameter (float): Parameter for the geometric distribution.

        Returns:
            list: A list where each element is a day number, representing an arrival day.
        """
        # Initialize each day with the minimum number of packets.
        x = np.ones((number_of_days,), dtype=int)
        x = [i * minimum_per_day for i in x]
        
        # Generate numbers from a geometric distribution.
        temp = np.random.geometric(parameter, number_of_days * 8)
        temp = np.sort(temp)[:len(temp) // 8] # Take a fraction of the sorted numbers.
        
        # Add the trend values to the base counts.
        for i in range(len(x)):
            x[i] = x[i] + temp[i]
            
        # Reverse the array to create a base trend.
        x = x[::-1]
        
        # Adjust the list to ensure the total packet count is exact.
        x = self.fix_list_length(x, number_of_packets, selected_day)
        
        # Sort to create the final increasing trend.
        temp = np.sort(x)
        
        # Convert daily counts into a list of day assignments.
        return self.generate_day_list(temp)

    def generate_day_list(self, input_list):
        """
        Converts a list of packet counts per day into a list of day assignments for each packet.
        Example: [2, 1] -> [0, 0, 1] (or a shuffled version).

        Args:
            input_list (np.array): An array where index is the day and value is the packet count.

        Returns:
            list: A list of day numbers, with a total length equal to the number of packets.
        """
        list_of_day = []
        # Process the list of daily counts from the last day to the first.
        while len(input_list):
            if input_list[-1] == 0:
                # If a day has no packets, remove it from consideration.
                input_list = np.delete(input_list, -1)
            else:
                # Get the current day number (which is the last index).
                value = len(input_list) - 1
                # Add this day number to the output list.
                list_of_day.append(value)
                # Decrement the count for this day.
                input_list[-1] -= 1
        return list_of_day

    def random_normal_integers(self, number_of_packets, number_of_days, minimum_per_day, selected_day, mu, sigma):
        """
        Generates packet arrival days following a normal (bell curve) distribution.
        
        Args:
            number_of_packets (int): Total number of packets.
            number_of_days (int): Total simulation days.
            minimum_per_day (int): Minimum packets per day.
            selected_day (int): Unused parameter.
            mu (float): The mean (center) of the normal distribution.
            sigma (float): The standard deviation of the normal distribution.

        Returns:
            list: A list of day numbers for packet arrivals.
        """
        # Initialize each day with the minimum number of packets.
        day_counts = np.ones((number_of_days,), dtype=int) * minimum_per_day
        remained_packets = number_of_packets - (number_of_days * minimum_per_day)
        
        # Generate random points from a normal distribution for the remaining packets.
        x = np.random.normal(mu, sigma, remained_packets)
        
        # Create a histogram to distribute these points across the days.
        h, _ = np.histogram(x, bins=number_of_days)
        
        # Add the histogram counts to the base minimums.
        for i in range(len(day_counts)):
            day_counts[i] += h[i]
            
        # Fix the total count to be exact.
        day_counts = self.fix_list_length(day_counts, number_of_packets, selected_day)
        
        # Convert counts to a list of day assignments.
        return self.generate_day_list(day_counts)

    def random_inverse_normal_integers(self, number_of_packets, number_of_days, minimum_per_day, selected_day, mu, sigma):
        """

        Generates packet arrival days following an inverse normal ('U-shaped') distribution.
        Arrivals are high at the beginning and end, and low in the middle.
        
        Args:
            number_of_packets (int): Total number of packets.
            number_of_days (int): Total simulation days.
            minimum_per_day (int): Minimum packets per day.
            selected_day (int): Unused parameter.
            mu (float): The mean (center) of the normal distribution.
            sigma (float): The standard deviation of the normal distribution.

        Returns:
            list: A list of day numbers for packet arrivals.
        """
        # Initialize each day with the minimum number of packets.
        day_counts = np.ones((number_of_days,), dtype=int) * minimum_per_day
        remained_packets = number_of_packets - (number_of_days * minimum_per_day)
        
        # Generate points from a normal distribution.
        x = np.random.normal(mu, sigma, remained_packets)
        
        # Create a histogram to group points into days.
        h, _ = np.histogram(x, bins=number_of_days)
        
        # Add histogram counts to the base.
        for i in range(len(day_counts)):
            day_counts[i] += h[i]
            
        # Invert the distribution by reversing the first and second halves.
        mid_point = int(len(day_counts) / 2)
        day_counts[0:mid_point] = day_counts[0:mid_point][::-1]
        day_counts[mid_point:len(day_counts)] = day_counts[mid_point:len(day_counts)][::-1]
        
        # Fix the total count to be exact.
        day_counts = self.fix_list_length(day_counts, number_of_packets, selected_day)
        
        # Convert counts to a list of day assignments.
        return self.generate_day_list(day_counts)

    def generate_packet_datetime_list(self, number):
        """
        Acts as a dispatcher to generate packet arrival datetimes based on the specified packet_rate.

        Args:
            number (int): The total number of datetimes to generate.

        Returns:
            list: A sorted list of packet arrival datetimes.
        """
        datetime_list = deque()
        # Parse the packet_rate string to get the distribution type and its parameters.
        (type, parameters) = self.packet_rate.split("_Parameters_")

        if type == "uniform":
            # For uniform distribution, call the standard datetime generator.
            datetime_list = self.generate_datetime_list(self.number_of_packets)
            return datetime_list
        elif type == "decreasing":
            # Parse parameters and call the decreasing trend generator.
            minimum_per_day, selected_day, parameter = parameters.split("_input_")
            minimum_per_day, selected_day, parameter = int(minimum_per_day), int(selected_day), float(parameter)
            list_of_days = self.decreasing_random_numbers(self.number_of_packets, self.number_of_days, minimum_per_day, selected_day, parameter)
        elif type == "increasing":
            # Parse parameters and call the increasing trend generator.
            minimum_per_day, selected_day, parameter = parameters.split("_input_")
            minimum_per_day, selected_day, parameter = int(minimum_per_day), int(selected_day), float(parameter)
            list_of_days = self.increasing_random_numbers(self.number_of_packets, self.number_of_days, minimum_per_day, selected_day, parameter)
        elif type == "normal":
            # Parse parameters and call the normal distribution generator.
            minimum_per_day, selected_day, mu, sigma = parameters.split("_input_")
            minimum_per_day, selected_day, mu, sigma = int(minimum_per_day), int(selected_day), int(mu), int(sigma)
            list_of_days = self.random_normal_integers(self.number_of_packets, self.number_of_days, minimum_per_day, selected_day, mu, sigma)
        elif type == "inverse_normal":
            # Parse parameters and call the inverse normal distribution generator.
            minimum_per_day, selected_day, mu, sigma = parameters.split("_input_")
            minimum_per_day, selected_day, mu, sigma = int(minimum_per_day), int(selected_day), int(mu), int(sigma)
            list_of_days = self.random_inverse_normal_integers(self.number_of_packets, self.number_of_days, minimum_per_day, selected_day, mu, sigma)
        elif type == "fluctuation":
            # Randomly choose one of the other distributions.
            minimum_per_day, selected_day, parameter, mu, sigma = parameters.split("_input_")
            minimum_per_day, selected_day, parameter, mu, sigma = int(minimum_per_day), int(selected_day), float(parameter), int(mu), int(sigma)
            type = random.randint(1, 5)
            if type == 1:  # increasing
                list_of_days = self.increasing_random_numbers(self.number_of_packets, self.number_of_days, minimum_per_day, selected_day, parameter)
            if type == 2:  # decreasing
                list_of_days = self.decreasing_random_numbers(self.number_of_packets, self.number_of_days, minimum_per_day, selected_day, parameter)
            if type == 3:  # normal
                list_of_days = self.random_normal_integers(self.number_of_packets, self.number_of_days, minimum_per_day, selected_day, mu, sigma)
            if type == 4:  # inverse_normal
                list_of_days = self.random_inverse_normal_integers(self.number_of_packets, self.number_of_days, minimum_per_day, selected_day, mu, sigma)
            if type == 5:  # uniform
                datetime_list = self.generate_datetime_list(self.number_of_packets)
                return datetime_list

        # Convert the generated list of day numbers into specific datetime objects.
        for _index in range(number):
            days = list_of_days[_index]
            datetime_list.append(self.start_date + self.add_random_days(days))
            
        # Ensure the final list has the exact number of unique datetimes.
        datetime_list = list(dict.fromkeys(datetime_list))
        difference = len(datetime_list) - number
        while difference != 0:
            if difference > 0:
                while (len(datetime_list) > number):
                    del datetime_list[randint(0, len(datetime_list) - 1)]
            elif difference < 0:
                for _ in range(-2 * difference):
                    days = randint(0, self.number_of_days - 1)
                    datetime_list.append(self.start_date + self.add_random_days(days))
            datetime_list = list(dict.fromkeys(datetime_list))
            difference = len(datetime_list) - number
            
        # Return the final sorted list.
        datetime_list = sorted(datetime_list)
        return datetime_list

    def generate_packet(self, index, arrival, deadline):
        """
        Generates a single packet with randomized attributes.

        Args:
            index (int): The unique ID for the packet.
            arrival (datetime): The arrival datetime for the packet.
            deadline (datetime): The delivery deadline for the packet.

        Returns:
            Packet: A newly created Packet object.
        """
        # Generate weight from a normal distribution.
        weight = self.generate_data(
            self.mean_packet_weight,
            self.stddev_packet_weight)
        # Budget is based on a normal distribution, scaled by weight.
        budget = (self.generate_data(self.mean_packet_budget, self.stddev_packet_budget)) * weight
        # Assign a random destination.
        destination = abs(int(uniform(1, self.number_of_destinations + 1)))
        # Generate late penalty from a normal distribution.
        penalty_late = self.generate_data(
            self.mean_packet_penalty_late,
            self.stddev_packet_penalty_late)
        # Generate unmatched penalty from a normal distribution.
        penalty_unmatched = self.generate_data(
            self.mean_packet_penalty_unmatched,
            self.stddev_packet_penalty_unmatched)
        # Generate notification time.
        notification_time = self.generate_data(
            self.mean_packet_notification_time,
            self.stddev_packet_notification_time)
        
        # Create and return the Packet object with initial status values.
        random_packet = Packet(
            packet_id=index,
            budget=budget,
            weight=weight,
            arrival=arrival,
            deadline=deadline,
            destination=destination,
            penalty_late=penalty_late,
            penalty_unmatched=penalty_unmatched,
            status=-2,  # -2 indicates 'unmatched' initial state.
            assigned_sp=0,
            departure=0,
            notification_time=notification_time,
            delivery_report=0,
            action_number=0)
        return random_packet

    def generate_packets(self):
        """
        Generates the full list of packets for the entire simulation.

        Returns:
            tuple: A tuple containing:
                - packets_list (dict): A dictionary of {arrival_time: Packet}.
                - packet_arrival_times (list): A sorted list of arrival times for the event queue.
                - packet_deadline_times (list): A sorted list of deadline times for the event queue.
        """
        # Generate the list of arrival times based on the selected packet rate distribution.
        datetime_list = self.generate_packet_datetime_list(self.number_of_packets)
        
        # Initialize containers for packets and their event times.
        packets_list = {}
        packet_arrival_times = sorted(datetime_list, reverse=True) # Sorted for pop() efficiency.
        packet_deadline_times = deque()
        
        # Iterate to create each packet.
        for index in range(self.number_of_packets):
            # Get the next arrival time from the pre-generated list.
            arrival = datetime_list.pop(0) # Use pop(0) for sorted list.
            
            # Calculate the deadline by adding a random duration to the arrival time.
            number_of_days_to_deadline = self.generate_data(
                self.mean_packet_time_to_deadline,
                self.stddev_packet_time_to_deadline)
            deadline = arrival + self.add_random_days(number_of_days_to_deadline)
            packet_deadline_times.append(deadline)
            
            # Generate the packet object with these times.
            random_packet = self.generate_packet(index, arrival, deadline)
            
            # Store the packet in a dictionary keyed by its arrival time.
            packets_list[arrival] = random_packet
            
        # Sort the deadline times for the event queue.
        packet_deadline_times = sorted(packet_deadline_times, reverse=True)
        return packets_list, packet_arrival_times, packet_deadline_times

    def generate_service_provider(self, counter, service_provider_company_info, departure_time):
        """
        Generates a single service provider instance for a specific departure.

        Args:
            counter (int): A unique ID for this service provider instance.
            service_provider_company_info (ServiceProvider): A template object containing company-level info (company ID, price, loss rate).
            departure_time (datetime): The specific departure time for this instance.

        Returns:
            ServiceProvider: A newly created ServiceProvider object.
        """
        # Calculate a random delivery duration.
        delivery_day = self.generate_data(
            self.mean_service_provider_delivery_time,
            self.stddev_service_provider_delivery_time)
        # Calculate the delivery time based on the departure time.
        delivery_time = departure_time + self.add_random_days(delivery_day)
        
        # Generate a random capacity for this departure.
        capacity = self.generate_data(
            self.mean_service_provider_capacity,
            self.stddev_service_provider_capacity)
            
        # Randomly select which destinations this service covers.
        destination_list = [i + 1 for i in range(self.number_of_destinations)]
        destination = sample(destination_list, randint(1, self.number_of_destinations))
        
        # Create and return the ServiceProvider object.
        random_service_provider = ServiceProvider(
            service_provider_number=counter,
            service_provider_company=service_provider_company_info.service_provider_company,
            price_per_kg=service_provider_company_info.price_per_kg,
            capacity=capacity,
            departure_time=departure_time,
            delivery_time=delivery_time,
            loss_rate=service_provider_company_info.loss_rate,
            delivered_packets=0,
            total_packets=0,
            destination=destination)
        return random_service_provider

    def generate_service_providers(self):
        """
        Generates all service provider companies and their individual departure instances.

        Returns:
            tuple: A tuple containing:
                - service_providers_list (dict): All SP instances, keyed by departure time.
                - service_provider_loss_values (list): Company-level SP info.
                - service_providers_departures_times (list): Sorted departure times for the event queue.
                - service_providers_in_window (list): Sorted times for when SPs become available for matching.
        """
        # Initialize containers.
        service_providers_list = {}
        service_provider_loss_values = deque() # Holds company-level info.
        service_providers_departures_times = deque()
        service_providers_in_window = deque()
        
        # Generate a pool of unique departure datetimes for all services.
        total_departures = self.number_of_service_providers * self.number_of_departures
        datetime_list = self.generate_datetime_list(total_departures)
        
        # Counter for unique service provider instance IDs.
        counter = 0
        
        # Loop through each service provider company.
        for company_id in range(self.number_of_service_providers):
            # Generate company-level attributes that are constant for all its departures.
            loss_rate = uniform(self.loss_service_provider_min, self.loss_service_provider_max)
            price_per_kg = self.generate_data(self.mean_service_provider_price_per_kg,
                                              self.stddev_service_provider_price_per_kg)
            # Create a template object for the company.
            company_template = ServiceProvider(0, company_id, price_per_kg, 0, 0, 0, loss_rate, 0, 0, [1])
            service_provider_loss_values.append(company_template)
            
            # Create the specified number of departure instances for this company.
            for _ in range(self.number_of_departures):
                # Pick a random, unique departure time from the pool.
                index = random.randint(0, len(datetime_list) - 1)
                departure_time = datetime_list.pop(index)
                
                # Generate the full service provider instance.
                random_service_provider = self.generate_service_provider(counter, company_template, departure_time)
                counter += 1
                
                # Store the instance and its key event times.
                service_providers_list[departure_time] = random_service_provider
                service_providers_departures_times.append(departure_time)
                
                # Calculate when this SP enters the matching window.
                in_window_time = departure_time - timedelta(days=self.window_time_for_service_providers)
                service_providers_in_window.append(in_window_time)
                
        # Sort the event time lists for the simulation queue.
        service_providers_departures_times = sorted(service_providers_departures_times, reverse=True)
        service_providers_in_window = sorted(service_providers_in_window, reverse=True)
        
        return service_providers_list, service_provider_loss_values, service_providers_departures_times, service_providers_in_window

    def generate_storages(self):
        """
        Generates the storage facility for the simulation.

        Returns:
            Storage: A Storage object with a randomized capacity.
        """
        # Generate capacity from a normal distribution.
        capacity = int(normalvariate(self.mean_storage_capacity, self.stddev_storage_capacity) + 0.5)
        
        # Create and return the Storage object.
        storage = Storage(1, capacity)
        return storage
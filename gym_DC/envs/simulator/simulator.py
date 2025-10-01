"""simulator module implements main simulation events"""
from collections import OrderedDict, deque
import random
from datetime import timedelta
from .generating_simulation_data import GeneratingData
import mcdm
import copy

class DistributionCentre:
    """DistributionCentre class implements main simulation methods"""
    def __init__(
            self,
            num_packets,
            num_sp,
            num_days,
            num_departures_sp,
            num_destination,
            budget,
            penalty,
            hyper_parameters,
            window_time_for_service_providers,
            loading_time,
            data_parameters, packet_rate, fixed_seed, seed, matching_model_type):
        self.num_packets = num_packets
        self.num_sp = num_sp
        self.num_days = num_days
        self.num_departures_sp = num_departures_sp
        self.num_destination = num_destination
        self.budget = budget
        self.penalty = penalty
        self.hyper_parameters = hyper_parameters
        self.window_time_for_service_providers = window_time_for_service_providers
        self.loading_time = loading_time
        self.data_parameters = data_parameters
        self.packet_rate = packet_rate
        self.full_capacity = 0
        self.fixed_seed = fixed_seed
        self.seed = seed
        if (self.fixed_seed):
            random.seed(self.seed)
        self.matching_model_type= matching_model_type
        # storage information
        self.storage = 0
        # different lists for packets
        # list of all packets in the beginning of simulation
        self.packets_list = {}
        # list of packet which are arrived but not loaded
        self.arrived_packets_list = {}
        # list of packets which are loaded to the storage space
        self.storage_packets_list = {}
        # list of packets which are matched but they are still in the storage
        self.matched_packets_list = {}
        # list of packets which are assigned to service providers and left the
        # DC
        self.sent_packets_list = {}
        # list of packets which left DC unmatched because of storage capacity
        # when they arrived
        self.unmatched_entrance_packets_list = deque()
        # list of packets which left DC unmatched because they remained in the
        # storage and the deadline is passed
        self.unmatched_storage_packets_list = deque()
        # list of packets which drooped from storage by decision making system
        # and left the system unmatched
        self.unmatched_dropped_packets_list = deque()
        # list of successful delivered Packets
        self.successful_delivered_packets = deque()
        # list of unsuccessful delivered Packets
        self.unsuccessful_delivered_packets = deque()
        self.packets_trend = []
        # different lists for service providers
        self.service_providers_list = {}
        # store the loss value for service providers to use later for
        # notification
        self.service_providers_loss_values = []
        self.service_providers_departed = deque()
        self.matching_service_providers = OrderedDict()
        # list of times
        self.packets_arrival_times = []
        self.packets_deadline_times = []
        self.packets_loading_times = []
        self.packets_notification_times = []
        self.packets_departure_times = []
        self.service_providers_departures_times = []
        self.service_providers_in_window_times = []

        self.evaluated_paired_packets = []
        self.evaluated_matched_packets = []
        self.evaluated_dropped_packets = []
        self.evaluated_storage_packets  = []


    @staticmethod
    def sort_packets_penalty_ascending_dict(input):
        """Sort list of packets based on their budget"""
        result = OrderedDict(
            sorted(
                input.items(),
                key=lambda x: (
                    x[1].penalty_unmatched/x[1].weight,
                    x[1].deadline),
                reverse=False))
        return result
    @staticmethod
    def sort_packets_budget_ascending_dict(input):
        """Sort list of packets based on their budget"""
        result = OrderedDict(
            sorted(
                input.items(),
                key=lambda x: (
                    x[1].budget /
                    x[1].weight,
                    x[1].deadline),
                reverse=False))
        return result
    @staticmethod
    def sort_packets_deadline_ascending_dict(input):
        """Sort list of packets based on their budget"""
        result = OrderedDict(
            sorted(
                input.items(),
                key=lambda x: (
                    x[1].deadline,
                    x[1].budget /
                    x[1].weight),
                reverse=True))
        return result
    @staticmethod
    def sort_service_provides_price_per_kg_ascending_dict(input):
        """Sort list of service providers based on their cost"""
        result = OrderedDict( sorted( input.items(),key=lambda x:(x[1].price_per_kg, x[1].departure_time),reverse=False))
        return result
    @staticmethod
    def sort_service_provides_delivery_time_ascending_dict(input):
        """Sort list of service providers based on their delivery time"""
        result = OrderedDict(
            sorted(
                input.items(),
                key=lambda x: (
                    x[1].departure_time,
                    x[1].price_per_kg),
                reverse=False))
        return result
    @staticmethod
    def pair_packet_sp(packet, service_provider, hyper_parameters):
        """pair one packet with one service_provider"""
        service_provider.capacity -= packet.weight
        packet.status = service_provider.service_provider_company
        packet.assigned_sp = service_provider.service_provider_number
        packet.departure = service_provider.departure_time
        packet.action_number = hyper_parameters
        return packet, service_provider
    @staticmethod
    def unpair_packet_sp(packet, service_provider):
        """unpair one packet with one service_provider"""
        service_provider.capacity += packet.weight
        packet.status = -1
        packet.assigned_sp = -1
        packet.departure = 0
        return packet, service_provider

    def matching_model(self, matching_model, time,check_packet_budget,check_dropped_packets, packet_criteria, check_packet_penalty):
        if matching_model==1:
            self.matching_model_1(time, check_packet_budget, check_dropped_packets)
        elif matching_model==2:
            self.matching_model_2(time,check_packet_budget,check_dropped_packets, packet_criteria, check_packet_penalty)
        elif matching_model == 3:
            self.matching_model_3(time,check_packet_budget,check_dropped_packets, packet_criteria, check_packet_penalty)

    def matching_model_1(self, time, check_packet_budget, check_dropped_packets):
        """ if we want to match a packet from matched_packets_list
         then we need to change its status and capacity of service provider
          In this model we only assume that the priority will be given
         to the packets with the highest budget we only match loaded
         packets and stored packets ( not arrived packets ) so we do
         not change the storage capacity"""
        matching_packets = OrderedDict(self.storage_packets_list)
        self.packets_trend.append(len(matching_packets))
        if len(self.packets_trend) > 5:
            del self.packets_trend[0]
        self.storage_packets_list.clear()
        # sort vectors based on the selected action
        matching_packets = self.sorting_action_dict(matching_packets)
        while len(matching_packets):  # check the length of list
            packet = matching_packets.popitem()
            packet = packet[1]
            check_not_matched = True
            """find the cheapest service provider for this packet the Matching_service_provides is sorted based on 
            the cost if weight of the packet is not matched to any service provider then we should check the storage 
            If the storage does not have capacity then the packet will leave the system unmatched"""
            for key, service_provider in self.matching_service_providers.items():
                if check_packet_budget:
                    condition = (service_provider.capacity >= packet.weight) and (packet.destination in service_provider.destination) and (service_provider.delivery_time <= packet.deadline) and (packet.budget > ((service_provider.price_per_kg) * (packet.weight)))
                else :
                    condition = (service_provider.capacity >= packet.weight) and (packet.destination in service_provider.destination) and (service_provider.delivery_time <= packet.deadline)
                if condition:
                    packet, service_provider = self.pair_packet_sp(packet, service_provider, self.hyper_parameters)
                    departure_time = service_provider.departure_time
                    while departure_time in self.matched_packets_list.keys():
                        departure_time = departure_time + \
                            timedelta(milliseconds=random.randint(0, 999))
                    self.packets_departure_times.append(departure_time)
                    self.packets_departure_times = sorted(
                        self.packets_departure_times, reverse=True)
                    self.matched_packets_list[departure_time] = packet
                    check_not_matched = False
                    self.matching_service_providers[key] = service_provider
                    break
            if check_not_matched:
                if check_dropped_packets:
                    #we check if the expire date for this packet is before time_window so we need to drop it
                    # we should check if there is not any service provider in time and the packet will expire in that window , then we should drop it

                    time = time + timedelta(days=self.window_time_for_service_providers)
                    if (packet.deadline <= time):
                        self.storage.capacity += packet.weight
                        self.penalty = self.penalty + packet.penalty_unmatched
                        self.unmatched_storage_packets_list.append(packet)

                    else:
                        # we subtract packet weight from storage capacity before this
                        # step
                        self.storage_packets_list[packet.deadline] = packet
                else:
                    self.storage_packets_list[packet.deadline] = packet

    def matching_model_2(self, time,check_packet_budget,check_dropped_packets, packet_criteria, check_packet_penalty):
        """ if we want to match a packet from matched_packets_list
         then we need to change its status and capacity of service provider
          In this model we only assume that the priority will be given
         to the packets with the highest budget we only match loaded
         packets and stored packets ( not arrived packets ) so we do
         not change the storage capacity"""
        matching_packets = OrderedDict(self.storage_packets_list)
        self.packets_trend.append(len(matching_packets))
        if len(self.packets_trend) > 5:
            del self.packets_trend[0]
        self.storage_packets_list.clear()
        # sort vectors based on the selected action
        if packet_criteria == 1:
            matching_packets = self.sorting_action_dict(matching_packets)
        if packet_criteria == 2:
            matching_packets = self.sorting_action_penalty_dict(matching_packets)
        # firstly we check if packets will match with one sp or not
        temp_matching_packets = copy.deepcopy(matching_packets)
        temp_matching_service_providers = copy.deepcopy(self.matching_service_providers)
        if check_packet_penalty:
            paired_packets,matched_packets, dropped_packets, storage_packets = self.evaluate_packets_penalty_model_2(temp_matching_packets, temp_matching_service_providers, check_packet_budget, check_dropped_packets, time)
        else:
            paired_packets, matched_packets, dropped_packets, storage_packets = self.evaluate_packets_penalty_model_1(temp_matching_packets, temp_matching_service_providers, check_packet_budget, check_dropped_packets, time)


        while len(matching_packets):  # check the length of list
            packet = matching_packets.popitem()
            packet = packet[1]
            if packet.packet_id in paired_packets:
                service_provider = self.matching_service_providers[time]
                packet, service_provider = self.pair_packet_sp(packet, service_provider, self.hyper_parameters)
                departure_time = service_provider.departure_time
                while departure_time in self.matched_packets_list.keys():
                    departure_time = departure_time + \
                                     timedelta(milliseconds=random.randint(0, 999))
                self.packets_departure_times.append(departure_time)
                self.packets_departure_times = sorted(
                    self.packets_departure_times, reverse=True)
                self.matched_packets_list[departure_time] = packet
                self.matching_service_providers[time] = service_provider
            elif packet.packet_id in matched_packets:
                self.storage_packets_list[packet.deadline] = packet
            elif packet.packet_id in storage_packets:
                self.storage_packets_list[packet.deadline] = packet
            elif packet.packet_id in dropped_packets:
                self.storage.capacity += packet.weight
                self.penalty = self.penalty + packet.penalty_unmatched
                self.unmatched_storage_packets_list.append(packet)

    def matching_model_3(self, time,check_packet_budget,check_dropped_packets, packet_criteria, check_packet_penalty):
        """ if we want to match a packet from matched_packets_list
         then we need to change its status and capacity of service provider
          In this model we only assume that the priority will be given
         to the packets with the highest budget we only match loaded
         packets and stored packets ( not arrived packets ) so we do
         not change the storage capacity"""
        matching_packets = OrderedDict(self.storage_packets_list)
        self.packets_trend.append(len(matching_packets))
        if len(self.packets_trend) > 5:
            del self.packets_trend[0]
        self.storage_packets_list.clear()
        # sort vectors based on the selected action
        if packet_criteria == 1:
            matching_packets = self.sorting_action_dict(matching_packets)
        if packet_criteria == 2:
            matching_packets = self.sorting_mcdm_action(matching_packets)
        # firstly we check if packets will match with one sp or not
        temp_matching_packets = copy.deepcopy(matching_packets)
        temp_matching_service_providers = copy.deepcopy(self.matching_service_providers)
        if check_packet_penalty:
            paired_packets,matched_packets, dropped_packets, storage_packets = self.evaluate_packets_penalty_model_2(temp_matching_packets, temp_matching_service_providers, check_packet_budget, check_dropped_packets, time)
        else:
            paired_packets, matched_packets, dropped_packets, storage_packets = self.evaluate_packets_penalty_model_1(temp_matching_packets, temp_matching_service_providers, check_packet_budget, check_dropped_packets, time)


        while len(matching_packets):  # check the length of list
            packet = matching_packets.popitem()
            packet = packet[1]
            if packet.packet_id in paired_packets:
                service_provider = self.matching_service_providers[time]
                packet, service_provider = self.pair_packet_sp(packet, service_provider, self.hyper_parameters)
                departure_time = service_provider.departure_time
                while departure_time in self.matched_packets_list.keys():
                    departure_time = departure_time + \
                                     timedelta(milliseconds=random.randint(0, 999))
                self.packets_departure_times.append(departure_time)
                self.packets_departure_times = sorted(
                    self.packets_departure_times, reverse=True)
                self.matched_packets_list[departure_time] = packet
                self.matching_service_providers[time] = service_provider
            elif packet.packet_id in matched_packets:
                self.storage_packets_list[packet.deadline] = packet
            elif packet.packet_id in storage_packets:
                self.storage_packets_list[packet.deadline] = packet
            elif packet.packet_id in dropped_packets:
                self.storage.capacity += packet.weight
                self.penalty = self.penalty + packet.penalty_unmatched
                self.unmatched_storage_packets_list.append(packet)

    def evaluate_packets_penalty_model_1(self, temp_matching_packets, temp_matching_service_providers, check_packet_budget, check_dropped_packets, departure_time):
        paired_packets = []
        matched_packets = []
        dropped_packets = []
        storage_packets = []
        self.evaluated_paired_packets.clear()
        self.evaluated_matched_packets.clear()
        self.evaluated_dropped_packets.clear()
        self.evaluated_storage_packets.clear()
        while len(temp_matching_packets):  # check the length of list
            packet = temp_matching_packets.popitem()
            packet = packet[1]
            check_not_matched = True
            """find the cheapest service provider for this packet the Matching_service_provides is sorted based on 
            the cost if weight of the packet is not matched to any service provider then we should check the storage 
            If the storage does not have capacity then the packet will leave the system unmatched"""
            for key, service_provider in temp_matching_service_providers.items():
                if check_packet_budget:
                    condition = (service_provider.capacity >= packet.weight) and (packet.destination in service_provider.destination) and (service_provider.delivery_time <= packet.deadline) and (packet.budget > ((service_provider.price_per_kg) * (packet.weight)))
                else:
                    condition = (service_provider.capacity >= packet.weight) and (packet.destination in service_provider.destination) and (service_provider.delivery_time <= packet.deadline)
                if condition:
                    packet, service_provider = self.pair_packet_sp(packet, service_provider, self.hyper_parameters)
                    check_not_matched = False
                    temp_matching_service_providers[key] = service_provider
                    if key == departure_time:
                        paired_packets.append(packet.packet_id)
                        self.evaluated_paired_packets.append(packet)
                    else:
                        matched_packets.append(packet.packet_id)
                        self.evaluated_matched_packets.append(packet)
                    break
            if check_not_matched:
                if check_dropped_packets:
                    #we check if the expire date for this packet is before time_window so we need to drop it
                    # we should check if there is not any service provider in time and the packet will expire in that window , then we should drop it
                    time = departure_time + timedelta(days=self.window_time_for_service_providers)
                    if (packet.deadline <= time):
                        dropped_packets.append(packet.packet_id)
                        self.evaluated_dropped_packets.append(packet)
                    else:
                        # we subtract packet weight from storage capacity before this
                        # step
                        storage_packets.append(packet.packet_id)
                        self.evaluated_storage_packets.append(packet)
                else:
                    # we subtract packet weight from storage capacity before this
                    # step
                    storage_packets.append(packet.packet_id)
                    self.evaluated_storage_packets.append(packet)
        return paired_packets,matched_packets, dropped_packets, storage_packets

    def evaluate_packets_penalty_model_2(self, temp_matching_packets, temp_matching_service_providers, check_packet_budget, check_dropped_packets, departure_time):
        "assign packet to service provider if the penalty of matching is higher than penalty of being unmatched"
        paired_packets = []
        matched_packets = []
        dropped_packets = []
        storage_packets = []
        self.evaluated_paired_packets.clear()
        self.evaluated_matched_packets.clear()
        self.evaluated_dropped_packets.clear()
        self.evaluated_storage_packets.clear()
        while len(temp_matching_packets):  # check the length of list
            packet = temp_matching_packets.popitem()
            packet = packet[1]
            check_not_matched = True
            """find the cheapest service provider for this packet the Matching_service_provides is sorted based on 
            the cost if weight of the packet is not matched to any service provider then we should check the storage 
            If the storage does not have capacity then the packet will leave the system unmatched"""
            for key, service_provider in temp_matching_service_providers.items():
                condition_1 = (service_provider.capacity >= packet.weight) and (
                                packet.destination in service_provider.destination) and (
                                            service_provider.delivery_time <= packet.deadline) and (
                                            packet.budget > ((service_provider.price_per_kg) * (packet.weight)))
                condition_2 = (service_provider.capacity >= packet.weight) and (
                                packet.destination in service_provider.destination) and (
                                            service_provider.delivery_time <= packet.deadline)
                condition_3 = ((abs(packet.budget - ((service_provider.price_per_kg) * (packet.weight))))< packet.penalty_unmatched )
                if condition_1:
                    condition = True
                elif condition_2:
                    if condition_3:
                        condition = True
                    else:
                        condition = False
                else:
                    condition = False
                if condition:
                    packet, service_provider = self.pair_packet_sp(packet, service_provider, self.hyper_parameters)
                    check_not_matched = False
                    temp_matching_service_providers[key] = service_provider
                    if key == departure_time:
                        paired_packets.append(packet.packet_id)
                        self.evaluated_paired_packets.append(packet)
                    else:
                        matched_packets.append(packet.packet_id)
                        self.evaluated_matched_packets.append(packet)
                    break
            if check_not_matched:
                if check_dropped_packets:
                    # we check if the expire date for this packet is before time_window so we need to drop it
                    # we should check if there is not any service provider in time and the packet will expire in that window , then we should drop it
                    time = departure_time + timedelta(days=self.window_time_for_service_providers)
                    if (packet.deadline <= time):
                        dropped_packets.append(packet.packet_id)
                        self.evaluated_dropped_packets.append(packet)
                    else:
                        # we subtract packet weight from storage capacity before this
                        # step
                        storage_packets.append(packet.packet_id)
                        self.evaluated_storage_packets.append(packet)
                else:
                    # we subtract packet weight from storage capacity before this
                    # step
                    storage_packets.append(packet.packet_id)
                    self.evaluated_storage_packets.append(packet)
        return paired_packets, matched_packets, dropped_packets, storage_packets

    # preparing data
    def generating_data(self, action_number_length):
        """generates list of packets and service_providers for the simulation period"""
        distribution_centre = GeneratingData(
            self.num_packets,
            self.num_sp,
            self.num_days,
            self.num_departures_sp,
            self.num_destination,
            self.data_parameters,
            self.window_time_for_service_providers, self.packet_rate,self.fixed_seed, self.seed)
        self.packets_list, self.packets_arrival_times, self.packets_deadline_times\
            = distribution_centre.generate_packets()
        self.service_providers_list, self.service_providers_loss_values,\
        self.service_providers_departures_times, self.service_providers_in_window_times\
            = distribution_centre.generate_service_providers()
        self.storage = distribution_centre.generate_storages()
        self.full_capacity = self.storage.capacity

    #######################################################################
    # normal matching

    def packets_arrival_event(self, time):
        """ simulation period"""
        if self.packets_list[time].weight <= self.storage.capacity:
            self.storage.capacity -= self.packets_list[time].weight
            loading_datetime = time + \
                timedelta(hours=self.loading_time)
            self.arrived_packets_list[loading_datetime] = self.packets_list.pop(
                time, None)
            self.packets_loading_times.append(loading_datetime)
            self.packets_loading_times = sorted(
                self.packets_loading_times, reverse=True)
        else:
            packet = self.packets_list.pop(time, None)
            self.penalty = self.penalty + packet.penalty_unmatched
            self.unmatched_entrance_packets_list.append(packet)

    def packets_deadline_event(self, time):
        """ simulation period"""
        if time in self.storage_packets_list.keys():
            self.storage.capacity += self.storage_packets_list[time].weight
            packet = self.storage_packets_list.pop(time, None)
            self.penalty = self.penalty + packet.penalty_unmatched
            self.unmatched_storage_packets_list.append(packet)

    def packets_loading_event(self, time):
        """Packets_list is sorted based on the time, we check packets
         which are arrived and loaded before time = T
        we have to check first element to compare the time and use pop left"""
        if time in self.arrived_packets_list.keys():
            self.arrived_packets_list[time].status = -1
            key = self.arrived_packets_list[time].deadline
            self.storage_packets_list[key] = (
                self.arrived_packets_list.pop(time, None))

    def service_providers_departures_event(self, time):
        """if it is not loaded before , we should add to matching list"""
        if time in self.service_providers_list.keys():
            self.matching_service_providers.append(
                self.service_providers_list.pop(time, None))
        # Send list for matching process at time = T
        # if there is no packet or SP, we do not need to match anything
        if len(self.storage_packets_list) and len(self.matching_service_providers):
            if self.matching_model_type == "type_1":
                self.matching_model(matching_model=1, time=time, check_packet_budget=False, check_dropped_packets=False, packet_criteria=None,check_packet_penalty=None)
            elif self.matching_model_type == "type_2":
                self.matching_model(matching_model=1, time=time, check_packet_budget=False, check_dropped_packets=True, packet_criteria=None,check_packet_penalty=None)
            elif self.matching_model_type == "type_3":
                self.matching_model(matching_model=1, time=time, check_packet_budget=True, check_dropped_packets=False,
                                    packet_criteria=None, check_packet_penalty=None)
            elif self.matching_model_type == "type_4":
                self.matching_model(matching_model=1, time=time, check_packet_budget=True, check_dropped_packets=True,
                                    packet_criteria=None, check_packet_penalty=None)
            ################################################################################################################
            elif self.matching_model_type == "type_5":
                self.matching_model(matching_model=2, time=time, check_packet_budget=False, check_dropped_packets=False,
                                    packet_criteria=1, check_packet_penalty=False)
            elif self.matching_model_type == "type_6":
                self.matching_model(matching_model=2, time=time, check_packet_budget=False, check_dropped_packets=True,
                                    packet_criteria=1, check_packet_penalty=False)
            elif self.matching_model_type == "type_7":
                self.matching_model(matching_model=2, time=time, check_packet_budget=True, check_dropped_packets=False,
                                    packet_criteria=1, check_packet_penalty=False)
            elif self.matching_model_type == "type_8":
                self.matching_model(matching_model=2, time=time, check_packet_budget=True, check_dropped_packets=True,
                                    packet_criteria=1, check_packet_penalty=False)
            ################################################################################################################
            elif self.matching_model_type == "type_9":
                self.matching_model(matching_model=2, time=time, check_packet_budget=False, check_dropped_packets=False,
                                    packet_criteria=2, check_packet_penalty=False)
            elif self.matching_model_type == "type_10":
                self.matching_model(matching_model=2, time=time, check_packet_budget=False, check_dropped_packets=True,
                                    packet_criteria=2, check_packet_penalty=False)
            elif self.matching_model_type == "type_11":
                self.matching_model(matching_model=2, time=time, check_packet_budget=True, check_dropped_packets=False,
                                    packet_criteria=2, check_packet_penalty=False)
            elif self.matching_model_type == "type_12":
                self.matching_model(matching_model=2, time=time, check_packet_budget=True, check_dropped_packets=True,
                                    packet_criteria=2, check_packet_penalty=False)
            ################################################################################################################
            elif self.matching_model_type == "type_13":
                self.matching_model(matching_model=2, time=time, check_packet_budget=True, check_dropped_packets=False,
                                    packet_criteria=1, check_packet_penalty=True)
            elif self.matching_model_type == "type_14":
                self.matching_model(matching_model=2, time=time, check_packet_budget=True, check_dropped_packets=True,
                                    packet_criteria=1, check_packet_penalty=True)
            elif self.matching_model_type == "type_15":
                self.matching_model(matching_model=2, time=time, check_packet_budget=True, check_dropped_packets=False,
                                    packet_criteria=2, check_packet_penalty=True)
            elif self.matching_model_type == "type_16":
                self.matching_model(matching_model=2, time=time, check_packet_budget=True, check_dropped_packets=True,
                                    packet_criteria=2, check_packet_penalty=True)
            ################################################################################################################
            elif self.matching_model_type == "type_17" or self.matching_model_type == "type_18" or self.matching_model_type == "type_19" or self.matching_model_type == "type_20" or self.matching_model_type == "type_21" or self.matching_model_type == "type_22":
                self.matching_model(matching_model=3, time=time, check_packet_budget=True, check_dropped_packets=True,
                                    packet_criteria=2, check_packet_penalty=True)

        # after matching process , we add sp to departed list
        # check for service providers which are departed
        self.service_providers_departed.append(
            self.matching_service_providers[time])
        del self.matching_service_providers[time]

    def packets_departure_event(self, time):
        """ simulation period"""
        if time in self.matched_packets_list.keys():
            packet = self.matched_packets_list.pop(time, None)
            self.storage.capacity += packet.weight
            self.service_providers_loss_values[packet.status].total_packets += 1
            self.service_providers_loss_values[packet.status].delivered_packets += 1
            profit = packet.budget - \
                (packet.weight * self.service_providers_loss_values[packet.status].price_per_kg)
            self.budget += profit
            self.successful_delivered_packets.append(packet)

    def service_providers_in_window_event(self, time):
        """ simulation period"""
        time = time + \
            timedelta(days=self.window_time_for_service_providers)
        if time in self.service_providers_list.keys():
            self.matching_service_providers[time] = self.service_providers_list.pop(
                time, None)

    def sorting_action_dict(self, matching_packets):
        """ simulation period"""
        packet_rank_score = self.hyper_parameters[0]
        sp_rank_score = self.hyper_parameters[1]
        p_budget = self.sort_packets_budget_ascending_dict(matching_packets)
        p_time = self.sort_packets_deadline_ascending_dict(matching_packets)


        sp_budget = self.sort_service_provides_price_per_kg_ascending_dict(self.matching_service_providers).copy()
        sp_time = self.sort_service_provides_delivery_time_ascending_dict(self.matching_service_providers).copy()

        list_p_budget=[]
        list_p_time=[]
        for key, value in p_budget.items():
            list_p_budget.append(value)
        for key, value in p_time.items():
            list_p_time.append(value)
        # find list of index
        packet_rank_list = []
        for i in range(len(list_p_budget)):
            index = list_p_time.index(list_p_budget[i])
            packet_rank_list.append([i , index])
        rows = 1
        columns = len(packet_rank_list)
        for i in range(len(packet_rank_list)):
            for j in range(len(packet_rank_list[i])):
                packet_rank_list[i][j] = self.normal_function(packet_rank_list[i][j], 0, columns-1, 0, 1)

        rank = mcdm.rank(packet_rank_list, w_vector=[packet_rank_score, 1-packet_rank_score], s_method="TOPSIS").copy()
        packet_list = []
        for i in range(len(rank)):
            element = rank.pop()
            index = int(element[0].replace("a", ""))
            packet_list.append(list_p_budget[index - 1])
            #packet_list.append(list_p_budget[index-1])

        list_sp_budget = []
        list_sp_time = []
        for key, value in sp_budget.items():
            list_sp_budget.append(value)
        for key, value in sp_time.items():
            list_sp_time.append(value)
        # find list of index
        sp_rank_list = []
        for i in range(len(list_sp_budget)):
            index = list_sp_time.index(list_sp_budget[i])
            sp_rank_list.append([i, index])
        rows = 1
        columns = len(sp_rank_list)
        for i in range(len(sp_rank_list)):
            for j in range(len(sp_rank_list[i])):
                sp_rank_list[i][j] = self.normal_function(sp_rank_list[i][j], 0, columns - 1, 0, 1)

        rank = mcdm.rank(sp_rank_list, w_vector=[sp_rank_score, 1 - sp_rank_score], s_method="TOPSIS").copy()
        sp_list = []
        for i in range(len(rank)):
            element = rank.pop()
            index = int(element[0].replace("a", ""))
            sp_list.append(list_sp_budget[index-1] )
            #sp_list.append(list_sp_budget[index-1])


        self.matching_service_providers.clear()
        for i in range(len(sp_list)):
            self.matching_service_providers[sp_list[i].departure_time]=sp_list[i]

        matching_packets.clear()
        for i in range(len(packet_list)):
            matching_packets[packet_list[i].deadline]=packet_list[i]
        #matching_packets = OrderedDict(sorted(matching_packets.items(), reverse=True))
        return matching_packets

    def sorting_action_penalty_dict(self, matching_packets):
        """ simulation period"""
        packet_rank_score = self.hyper_parameters[0]
        sp_rank_score = self.hyper_parameters[1]
        p_budget = self.sort_packets_budget_ascending_dict(matching_packets)
        p_penalty = self.sort_packets_penalty_ascending_dict(matching_packets)


        sp_budget = self.sort_service_provides_price_per_kg_ascending_dict(self.matching_service_providers).copy()
        sp_time = self.sort_service_provides_delivery_time_ascending_dict(self.matching_service_providers).copy()

        list_p_budget=[]
        list_p_time=[]
        for key, value in p_budget.items():
            list_p_budget.append(value)
        for key, value in p_penalty.items():
            list_p_time.append(value)
        # find list of index
        packet_rank_list = []
        for i in range(len(list_p_budget)):
            index = list_p_time.index(list_p_budget[i])
            packet_rank_list.append([i , index])
        rows = 1
        columns = len(packet_rank_list)
        for i in range(len(packet_rank_list)):
            for j in range(len(packet_rank_list[i])):
                packet_rank_list[i][j] = self.normal_function(packet_rank_list[i][j], 0, columns-1, 0, 1)

        rank = mcdm.rank(packet_rank_list, w_vector=[packet_rank_score, 1-packet_rank_score], s_method="TOPSIS").copy()
        packet_list = []
        for i in range(len(rank)):
            element = rank.pop()
            index = int(element[0].replace("a", ""))
            packet_list.append(list_p_budget[index - 1])
            #packet_list.append(list_p_budget[index-1])

        list_sp_budget = []
        list_sp_time = []
        for key, value in sp_budget.items():
            list_sp_budget.append(value)
        for key, value in sp_time.items():
            list_sp_time.append(value)
        # find list of index
        sp_rank_list = []
        for i in range(len(list_sp_budget)):
            index = list_sp_time.index(list_sp_budget[i])
            sp_rank_list.append([i, index])
        rows = 1
        columns = len(sp_rank_list)
        for i in range(len(sp_rank_list)):
            for j in range(len(sp_rank_list[i])):
                sp_rank_list[i][j] = self.normal_function(sp_rank_list[i][j], 0, columns - 1, 0, 1)

        rank = mcdm.rank(sp_rank_list, w_vector=[sp_rank_score, 1 - sp_rank_score], s_method="TOPSIS").copy()
        sp_list = []
        for i in range(len(rank)):
            element = rank.pop()
            index = int(element[0].replace("a", ""))
            sp_list.append(list_sp_budget[index-1] )
            #sp_list.append(list_sp_budget[index-1])


        self.matching_service_providers.clear()
        for i in range(len(sp_list)):
            self.matching_service_providers[sp_list[i].departure_time]=sp_list[i]

        #self.matching_service_providers=OrderedDict(sorted(self.matching_service_providers.items(), reverse=True))

        matching_packets.clear()
        for i in range(len(packet_list)):
            matching_packets[packet_list[i].deadline]=packet_list[i]
        #matching_packets = OrderedDict(sorted(matching_packets.items(), reverse=True))
        return matching_packets

    @staticmethod
    def normal_function(value, a, b, c, d):
        if value == 0:
            return 0
        if (b - a) == 0:
            return 0
        result = c + ((d - c) / (b - a)) * (value - a)
        return result

    def evaluate_greedy_action(self):

        matching_packets = OrderedDict(self.storage_packets_list)
        # sort vectors based on the selected action
        packets = copy.deepcopy(matching_packets)
        profit, penalty = 0, 0
        for key in packets.keys():
            penalty += packets[key].penalty_unmatched
        service_providers = copy.deepcopy(self.matching_service_providers)
        if len(packets)==0 or len(service_providers)==0:
            return 0,0
        packets, service_providers  =self.sorting_action_dict_greedy_policy( packets,service_providers)
        while len(packets):  # check the length of list
            packet = packets.popitem()
            packet = packet[1]
            check_matched = True
            """find the cheapest service provider for this packet the Matching_service_provides is sorted based on 
            the cost if weight of the packet is not matched to any service provider then we should check the storage 
            If the storage does not have capacity then the packet will leave the system unmatched"""
            for key, service_provider in service_providers.items():
                if (service_provider.capacity >= packet.weight) \
                        and (packet.destination in service_provider.destination)\
                        and (service_provider.delivery_time <= packet.deadline)\
                        and (packet.budget > ((service_provider.price_per_kg) * (packet.weight))):
                    packet, service_provider = self.pair_packet_sp(packet, service_provider, self.hyper_parameters)
                    profit += packet.budget - (packet.weight * service_provider.price_per_kg)
                    service_providers[key] = service_provider
                    break
        return profit, penalty

    def sorting_action_dict_greedy_policy(self, packets,service_providers):
        """ simulation period"""
        packet_rank_score = 1
        sp_rank_score = 1
        p_budget = self.sort_packets_budget_ascending_dict(packets)
        p_time = self.sort_packets_deadline_ascending_dict(packets)


        sp_budget = self.sort_service_provides_price_per_kg_ascending_dict(service_providers).copy()
        sp_time = self.sort_service_provides_delivery_time_ascending_dict(service_providers).copy()

        list_p_budget=[]
        list_p_time=[]
        for key, value in p_budget.items():
            list_p_budget.append(value)
        for key, value in p_time.items():
            list_p_time.append(value)
        # find list of index
        packet_rank_list = []
        for i in range(len(list_p_budget)):
            index = list_p_time.index(list_p_budget[i])
            packet_rank_list.append([i , index])
        rows = 1
        columns = len(packet_rank_list)
        for i in range(len(packet_rank_list)):
            for j in range(len(packet_rank_list[i])):
                packet_rank_list[i][j] = self.normal_function(packet_rank_list[i][j], 0, columns-1, 0, 1)

        rank = mcdm.rank(packet_rank_list, w_vector=[packet_rank_score, 1-packet_rank_score], s_method="TOPSIS").copy()
        packet_list = []
        for i in range(len(rank)):
            element = rank.pop()
            index = int(element[0].replace("a", ""))
            packet_list.append(list_p_budget[index - 1])

        list_sp_budget = []
        list_sp_time = []
        for key, value in sp_budget.items():
            list_sp_budget.append(value)
        for key, value in sp_time.items():
            list_sp_time.append(value)
        # find list of index
        sp_rank_list = []
        for i in range(len(list_sp_budget)):
            index = list_sp_time.index(list_sp_budget[i])
            sp_rank_list.append([i, index])
        rows = 1
        columns = len(sp_rank_list)
        for i in range(len(sp_rank_list)):
            for j in range(len(sp_rank_list[i])):
                sp_rank_list[i][j] = self.normal_function(sp_rank_list[i][j], 0, columns - 1, 0, 1)

        rank = mcdm.rank(sp_rank_list, w_vector=[sp_rank_score, 1 - sp_rank_score], s_method="TOPSIS").copy()
        sp_list = []
        for i in range(len(rank)):
            element = rank.pop()
            index = int(element[0].replace("a", ""))
            sp_list.append(list_sp_budget[index-1] )
            #sp_list.append(list_sp_budget[index-1])


        service_providers.clear()
        for i in range(len(sp_list)):
            service_providers[sp_list[i].departure_time]=sp_list[i]

        #self.matching_service_providers=OrderedDict(sorted(self.matching_service_providers.items(), reverse=True))

        packets.clear()
        for i in range(len(packet_list)):
            packets[packet_list[i].deadline]=packet_list[i]
        #matching_packets = OrderedDict(sorted(matching_packets.items(), reverse=True))
        return packets, service_providers

    #######################################################################
    # sorting mcdm

    def sorting_mcdm_action(self, matching_packets):
        """ ranking packets"""
        rank,list_p_budget = self.ranking_packets(matching_packets)
        packet_list = []
        for i in range(len(rank)):
            element = rank.pop()
            index = int(element[0].replace("a", ""))
            packet_list.append(list_p_budget[index - 1])

        matching_packets.clear()
        for i in range(len(packet_list)):
            matching_packets[packet_list[i].deadline]=packet_list[i]

        """ ranking service providers"""
        sp_list = self.ranking_service_providers()
        self.matching_service_providers.clear()
        for i in range(len(sp_list)):
            self.matching_service_providers[sp_list[i].departure_time] = sp_list[i]



        return matching_packets

    def ranking_packets(self, matching_packets):
        """ simulation period"""
        w_budget = self.hyper_parameters[0]
        w_time = self.hyper_parameters[1]
        w_unmatched = self.hyper_parameters[2]
        w_late = self.hyper_parameters[3]

        sum = w_budget + w_time + w_unmatched +w_late
        if sum ==0:
            w_budget , w_time , w_unmatched , w_late = 0.25, 0.25, 0.25, 0.25
        else:
            w_budget, w_time, w_unmatched, w_late = w_budget/sum , w_time/sum , w_unmatched/sum , w_late/sum


        p_budget = self.sort_packets_budget(matching_packets)
        p_time = self.sort_packets_deadline(matching_packets)
        p_unmatched = self.sort_packets_penalty_unmatched(matching_packets)
        p_late = self.sort_packets_penalty_late(matching_packets)



        list_p_budget = []
        list_p_time = []
        list_p_unmatched = []
        list_p_late = []
        for key, value in p_budget.items():
            list_p_budget.append(value)
        for key, value in p_time.items():
            list_p_time.append(value)
        for key, value in p_unmatched.items():
            list_p_unmatched.append(value)
        for key, value in p_late.items():
            list_p_late.append(value)
        # find list of index
        packet_rank_list = []
        for i in range(len(list_p_budget)):
            index_time = list_p_time.index(list_p_budget[i])
            index_unmatched = list_p_unmatched.index(list_p_budget[i])
            index_late = list_p_late.index(list_p_budget[i])
            packet_rank_list.append([i, index_time, index_unmatched,index_late])
        rows = 1
        columns = len(packet_rank_list)
        for i in range(len(packet_rank_list)):
            for j in range(len(packet_rank_list[i])):
                packet_rank_list[i][j] = self.normal_function(packet_rank_list[i][j], 0, columns - 1, 0, 1)


        rank = mcdm.rank(packet_rank_list, w_vector=[w_budget, w_time, w_unmatched ,w_late],s_method="TOPSIS").copy()
        return rank,list_p_budget

    def ranking_service_providers(self):
        w_budget = self.hyper_parameters[4]
        w_departure_time = self.hyper_parameters[5]
        w_delivery_time = self.hyper_parameters[6]

        sum = w_budget + w_departure_time + w_delivery_time

        if sum == 0:
            w_budget , w_departure_time , w_delivery_time = 1/3 , 1/3, 1/3
        else:
            w_budget , w_departure_time , w_delivery_time =  w_budget/sum , w_departure_time/sum , w_delivery_time/sum

        sp_budget = self.sort_sp_price_per_kg(self.matching_service_providers).copy()
        sp_departure_time = self.sort_sp_departure_time(self.matching_service_providers).copy()
        sp_delivery_time = self.sort_sp_delivery_time(self.matching_service_providers).copy()

        list_sp_budget = []
        list_sp_departure_time = []
        list_sp_delivery_time = []

        for key, value in sp_budget.items():
            list_sp_budget.append(value)
        for key, value in sp_departure_time.items():
            list_sp_departure_time.append(value)
        for key, value in sp_delivery_time.items():
            list_sp_delivery_time.append(value)
        # find list of index
        sp_rank_list = []
        for i in range(len(list_sp_budget)):
            index_departure_time = list_sp_departure_time.index(list_sp_budget[i])
            index_delivery_time = list_sp_delivery_time.index(list_sp_budget[i])
            sp_rank_list.append([i, index_departure_time, index_delivery_time])

        columns = len(sp_rank_list)
        for i in range(len(sp_rank_list)):
            for j in range(len(sp_rank_list[i])):
                sp_rank_list[i][j] = self.normal_function(sp_rank_list[i][j], 0, columns - 1, 0, 1)
        rank = mcdm.rank(sp_rank_list, w_vector=[w_budget, w_departure_time, w_delivery_time], s_method="TOPSIS").copy()
        sp_list = []
        for i in range(len(rank)):
            element = rank.pop()
            index = int(element[0].replace("a", ""))
            sp_list.append(list_sp_budget[index - 1])
            # sp_list.append(list_sp_budget[index-1])

        return sp_list

    #packets
    @staticmethod
    def sort_packets_budget(input):
        """Sort list of packets based on their budget"""
        result = OrderedDict(
            sorted(
                input.items(),
                key=lambda x: (
                    x[1].budget /
                    x[1].weight,
                    x[1].deadline),
                reverse=False))
        return result
    @staticmethod
    def sort_packets_deadline(input):
        """Sort list of packets based on their budget"""
        result = OrderedDict(
            sorted(
                input.items(),
                key=lambda x: (
                    x[1].deadline,
                    x[1].budget /
                    x[1].weight),
                reverse=True))
        return result
    @staticmethod
    def sort_packets_penalty_unmatched(input):
        """Sort list of packets based on their budget"""
        result = OrderedDict(
            sorted(
                input.items(),
                key=lambda x: (
                    x[1].penalty_unmatched /
                    x[1].weight,
                    x[1].deadline),
                reverse=False))
        return result
    @staticmethod
    def sort_packets_penalty_late(input):
        """Sort list of packets based on their budget"""
        result = OrderedDict(
            sorted(
                input.items(),
                key=lambda x: (
                    x[1].penalty_late /
                    x[1].weight,
                    x[1].deadline),
                reverse=False))
        return result

    #service providers
    @staticmethod
    def sort_sp_price_per_kg(input):
        """Sort list of service providers based on their cost"""
        result = OrderedDict( sorted( input.items(),key=lambda x:(x[1].price_per_kg, x[1].departure_time),reverse=False))
        return result
    @staticmethod
    def sort_sp_departure_time(input):
        """Sort list of service providers based on their delivery time"""
        result = OrderedDict(
            sorted(
                input.items(),
                key=lambda x: (
                    x[1].departure_time,
                    x[1].price_per_kg),
                reverse=False))
        return result
    @staticmethod
    def sort_sp_delivery_time(input):
        """Sort list of service providers based on their delivery time"""
        result = OrderedDict(
            sorted(
                input.items(),
                key=lambda x: (
                    x[1].delivery_time,
                    x[1].price_per_kg),
                reverse=False))
        return result

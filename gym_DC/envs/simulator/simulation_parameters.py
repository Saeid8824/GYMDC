"""
Defines the dataclass for storing all simulation configuration parameters.

This module centralizes all statistical inputs for data generation, making it
easy to configure and modify simulation scenarios.
"""
from dataclasses import dataclass
import datetime


@dataclass
class SimulationDataParameters:
    """
    A container for all simulation data generation parameters.

    This class uses a dataclass to hold the mean and standard deviation for various
    attributes of packets, service providers, and storage. This allows for easy
    configuration of different simulation scenarios from a single object.
    
    Using __slots__ optimizes memory usage.
    """
    # Define slots to pre-allocate space for attributes, saving memory.
    __slots__ = [
        'start_date',
        'mean_packet_weight',
        'stddev_packet_weight',
        'mean_packet_budget',
        'stddev_packet_budget',
        'mean_packet_penalty_late',
        'stddev_packet_penalty_late',
        'mean_packet_penalty_unmatched',
        'stddev_packet_penalty_unmatched',
        'mean_packet_time_to_deadline',
        'stddev_packet_time_to_deadline',
        'mean_packet_notification_time',
        'stddev_packet_notification_time',
        'mean_service_provider_price_per_kg',
        'stddev_service_provider_price_per_kg',
        'mean_service_provider_delivery_time',
        'stddev_service_provider_delivery_time',
        'loss_service_provider_max',
        'loss_service_provider_min',
        'mean_service_provider_capacity',
        'stddev_service_provider_capacity',
        'mean_storage_price_per_kg',
        'stddev_storage_price_per_kg',
        'mean_storage_capacity',
        'stddev_storage_capacity']
        
    # --- General Simulation Parameters ---
    start_date: datetime.date
    
    # --- Packet Attribute Parameters (Mean & Standard Deviation) ---
    mean_packet_weight: int
    stddev_packet_weight: int
    mean_packet_budget: int
    stddev_packet_budget: int
    mean_packet_penalty_late: int
    stddev_packet_penalty_late: int
    mean_packet_penalty_unmatched: int
    stddev_packet_penalty_unmatched: int
    mean_packet_time_to_deadline: int
    stddev_packet_time_to_deadline: int
    mean_packet_notification_time: int
    stddev_packet_notification_time: int
    
    # --- Service Provider Attribute Parameters (Mean, StdDev, Min/Max) ---
    mean_service_provider_price_per_kg: int
    stddev_service_provider_price_per_kg: int
    mean_service_provider_delivery_time: int
    stddev_service_provider_delivery_time: int
    loss_service_provider_max: float
    loss_service_provider_min: float
    mean_service_provider_capacity: int
    stddev_service_provider_capacity: int
    
    # --- Storage Attribute Parameters (Mean & Standard Deviation) ---
    mean_storage_price_per_kg: int
    stddev_storage_price_per_kg: int
    mean_storage_capacity: int
    stddev_storage_capacity: int
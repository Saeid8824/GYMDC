"""
Defines the dataclass for a Service Provider in the simulation.

This module contains the data structure for representing a single transport service
available at a specific time.
"""
from dataclasses import dataclass


@dataclass
class ServiceProvider:
    """
    Represents a single service provider instance for a specific departure.

    Each object corresponds to a unique scheduled departure by a transport company.
    It includes details about cost, capacity, schedule, and performance metrics.

    Attributes:
        service_provider_number (int): A unique identifier for this specific service instance.
        service_provider_company (int): An identifier for the parent transport company.
        price_per_kg (int): The cost to transport one kilogram of weight.
        capacity (int): The remaining available weight capacity for this departure.
        departure_time (int): The scheduled departure datetime from the distribution center.
        delivery_time (int): The scheduled arrival datetime at the destination.
        loss_rate (float): The statistical probability of a packet being lost during transit.
        delivered_packets (int): A counter for successfully delivered packets by this provider.
        total_packets (int): A counter for all packets assigned to this provider.
        destination (list): A list of destination IDs that this service can deliver to.
    """
    # Use __slots__ for memory efficiency, as many ServiceProvider objects may be created.
    __slots__ = [
        'service_provider_number',
        'service_provider_company',
        'price_per_kg',
        'capacity',
        'departure_time',
        'delivery_time',
        'loss_rate',
        'delivered_packets',
        'total_packets',
        'destination']
        
    # --- Attribute Definitions ---
    service_provider_number: int
    service_provider_company: int
    price_per_kg: int
    capacity: int
    departure_time: int
    delivery_time: int
    loss_rate: float
    delivered_packets: int
    total_packets: int
    destination: list
"""
Defines the dataclass for a Packet in the simulation.

This module contains the data structure representing a single shipment request.
"""
from dataclasses import dataclass


@dataclass
class Packet:
    """
    Represents a single packet to be processed by the distribution center.

    This class holds all information related to a packet, including its physical
    attributes, financial details, scheduling constraints, and current status
    within the simulation.

    Attributes:
        packet_id (int): A unique identifier for the packet.
        budget (int): The total budget allocated for shipping this packet.
        weight (int): The weight of the packet in kilograms.
        arrival (int): The datetime the packet arrives at the distribution center.
        deadline (int): The latest datetime the packet must be delivered by.
        destination (int): An identifier for the packet's destination.
        penalty_late (int): The financial penalty incurred if the packet is delivered after its deadline.
        penalty_unmatched (int): The financial penalty incurred if the packet is not shipped at all.
        status (int): The current status of the packet. Can represent the assigned service provider company ID or a state like 'unmatched' (-2) or 'in storage' (-1).
        assigned_sp (int): The unique ID of the service provider instance this packet is matched with.
        departure (int): The scheduled departure datetime from the center.
        notification_time (int): Time parameter related to delivery notifications.
        delivery_report (int): A flag or status for the final delivery report.
        action_number (int): An identifier for the decision-making action that processed this packet.
    """
    # Use __slots__ for memory optimization.
    __slots__ = [
        'packet_id',
        'budget',
        'weight',
        'arrival',
        'deadline',
        'destination',
        'penalty_late',
        'penalty_unmatched',
        'status',
        'assigned_sp',
        'departure',
        'notification_time',
        'delivery_report',
        'action_number']
        
    # --- Attribute Definitions ---
    packet_id: int
    budget: int
    weight: int
    arrival: int
    deadline: int
    destination: int
    penalty_late: int
    penalty_unmatched: int
    status: int
    assigned_sp: int
    departure: int
    notification_time: int
    delivery_report: int
    action_number: int
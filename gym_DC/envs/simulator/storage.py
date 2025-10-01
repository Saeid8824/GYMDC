"""
Defines the dataclass for a storage facility in the simulation.

This module contains the data structure for representing the distribution center's storage.
"""
from dataclasses import dataclass


@dataclass
class Storage:
    """
    Represents a storage facility with a unique ID and capacity.

    Using a dataclass provides a concise way to group data without boilerplate code.
    The __slots__ attribute is used for memory optimization by preventing the creation
    of an instance __dict__, which is useful when creating many objects.

    Attributes:
        id_storage (int): A unique identifier for the storage facility.
        capacity (int): The total weight capacity of the storage facility in kilograms.
    """
    # Define slots to pre-allocate space for attributes, saving memory.
    __slots__ = ['id_storage', 'capacity']
    
    # --- Attribute Definitions ---
    id_storage: int
    capacity: int
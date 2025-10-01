"""
Defines the observation (state) spaces for the Gym environment.

This module provides a factory class, ObservationSwitch, to generate different
OpenAI Gym observation spaces based on a selected model name. Each observation
space defines the structure of the state information that the agent receives
from the environment at each step.
"""
from gym import spaces
import numpy as np

class ObservationSwitch:
    """
    A factory class to generate different observation space configurations.
    
    This class uses the getattr pattern to dynamically call a method corresponding
    to the provided `observation_model` string, returning the appropriate Gym space.
    """
    def observation_space(self, observation_model):
        """
        Selects and returns the observation space for the given model name.

        Args:
            observation_model (str): The name of the observation model (e.g., "observation_1").

        Returns:
            gym.spaces.Space: The corresponding observation space object.
        """
        # Set a default message for an invalid model name.
        default = "Incorrect type"
        # Dynamically get the method for the observation_model and call it.
        return getattr(self, observation_model, lambda: default)()

    def observation_1(self):
        """Defines a discrete observation space with 2000 possible states."""
        # This typically means the state is encoded as a single integer.
        observation_space = spaces.Discrete(2000)
        return observation_space

    def observation_2(self):
        """Defines a discrete observation space with 1000 possible states."""
        observation_space = spaces.Discrete(1000)
        return observation_space

    def observation_3(self):
        """Defines a 2D continuous observation space."""
        # The state is a vector of two floating-point numbers.
        # The first value is between 0.0 and 1.0.
        # The second value is between 0.0 and 2.0.
        high_o = np.array([1.0, 2.0], dtype=np.float32)
        low_o = np.array([0, 0], dtype=np.float32)
        observation_space = spaces.Box(low=low_o, high=high_o, dtype=np.float32)
        return observation_space

    def observation_4(self):
        """Defines a 2D continuous observation space where both values are normalized between 0.0 and 1.0."""
        # The state is a vector of two floating-point numbers.
        high_o = np.array([1.0, 1.0], dtype=np.float32)
        low_o = np.array([0, 0], dtype=np.float32)
        observation_space = spaces.Box(low=low_o, high=high_o, dtype=np.float32)
        return observation_space
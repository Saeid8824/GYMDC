"""
Defines the action spaces and action value interpretations for the Gym environment.

This module provides two main classes:
1. ActionModelSwitch: A factory for creating different OpenAI Gym action spaces
   (e.g., Discrete, Box) based on a selected model name.
2. ActionValueSwitch: A class to translate the raw action from an RL agent
   into the specific list of hyperparameter values required by the simulator.
"""
from gym import spaces
import numpy as np

class ActionModelSwitch:
    """
    A factory class to generate different action space configurations.
    
    This class uses the getattr pattern to dynamically call a method corresponding
    to the provided `action_model` string, returning the appropriate Gym action space.
    """
    def action_space(self, action_model):
        """
        Selects and returns the action space for the given model name.

        Args:
            action_model (str): The name of the action model (e.g., "model_1").

        Returns:
            A tuple (bool, gym.spaces.Space) representing (is_discrete, action_space_object).
        """
        # Set a default message for an invalid model name.
        default = "Incorrect type"
        # Dynamically get the method corresponding to action_model and call it.
        return getattr(self, action_model, lambda: default)()

    def model_1(self):
        """Defines a discrete action space with 2 possible actions."""
        discrete_space = True
        action_space = spaces.Discrete(2)
        return discrete_space, action_space

    def model_2(self):
        """Defines a 2-dimensional continuous action space with integer values [0, 1]."""
        discrete_space = False
        # Note: np.int is deprecated; np.int_ should be used in newer numpy versions.
        action_space = spaces.Box(
            low=np.array([0, 0]).astype(np.int),
            high=np.array([1, 1]).astype(np.int),
        )
        return discrete_space, action_space

    def model_3(self):
        """Defines a discrete action space with 100 possible actions."""
        discrete_space = True
        action_space = spaces.Discrete(100)
        return discrete_space, action_space

    def model_4(self):
        """Defines a 4-dimensional continuous action space with float values [0.0, 1.0]."""
        discrete_space = False
        high = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        low = np.array([0, 0, 0, 0], dtype=np.float32)
        action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return discrete_space, action_space

    def model_5(self):
        """Defines a 5-dimensional continuous action space with float values [0.0, 1.0]."""
        discrete_space = False
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        low = np.array([0, 0, 0, 0, 0], dtype=np.float32)
        action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return discrete_space, action_space

    def model_6(self):
        """Defines a 6-dimensional continuous action space with float values [0.0, 1.0]."""
        discrete_space = False
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        low = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
        action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return discrete_space, action_space

    def model_7(self):
        """Defines a 5-dimensional continuous action space with float values [0.0, 1.0]."""
        discrete_space = False
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        low = np.array([0, 0, 0, 0, 0], dtype=np.float32)
        action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return discrete_space, action_space

    def model_8(self):
        """Defines a 6-dimensional continuous action space with float values [0.0, 1.0]."""
        discrete_space = False
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        low = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
        action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return discrete_space, action_space

    def model_9(self):
        """Defines a 7-dimensional continuous action space with float values [0.0, 1.0]."""
        discrete_space = False
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        low = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return discrete_space, action_space

    def model_10(self):
        """Defines a discrete action space with 2 possible actions."""
        discrete_space = True
        action_space = spaces.Discrete(2)
        return discrete_space, action_space

    def model_11(self):
        """Defines a discrete action space with 2187 (3^7) actions, mapping to a list of 7 params."""
        discrete_space = True
        action_space = spaces.Discrete(2187)
        return discrete_space, action_space

    def model_12(self):
        """Defines a discrete action space with 20 pre-defined hyperparameter sets."""
        discrete_space = True
        action_space = spaces.Discrete(20)
        return discrete_space, action_space


class ActionValueSwitch:
    """
    A factory class to translate a raw agent action into simulator-ready hyperparameters.
    """
    def action_value(self, action_model, action_value):
        """
        Selects the correct interpretation logic for the given action model.

        Args:
            action_model (str): The name of the action model (e.g., "model_1").
            action_value (any): The raw action from the agent (e.g., an int or numpy array).

        Returns:
            list: A list of hyperparameter values for the simulator.
        """
        default = "Incorrect type"
        # Store the raw action value to be used by the specific model method.
        self.action_value = action_value
        # Dynamically call the correct interpretation method.
        return getattr(self, action_model, lambda: default)()

    def model_1(self):
        """Maps a discrete action (0 or 1) to a simple hyperparameter list."""
        if self.action_value == 0:
            action = [0, 0] # Represents one strategy
        else:
            action = [1, 1] # Represents another strategy
        return action

    def model_2(self):
        """Passes through the continuous action value directly."""
        return self.action_value

    def model_3(self):
        """Maps a discrete action to a simple hyperparameter list (same as model_1)."""
        if self.action_value == 0:
            action = [0, 0]
        else:
            action = [1, 1]
        return action

    def model_4(self):
        """Maps a 4-element continuous action to a 7-element hyperparameter list, inserting zeros."""
        action = self.action_value
        return [action[0], action[1], 0, 0, action[2], action[3], 0]

    def model_5(self):
        """Maps a 5-element continuous action to a 7-element hyperparameter list."""
        action = self.action_value
        return [action[0], action[1], action[2], 0, action[3], action[4], 0]

    def model_6(self):
        """Maps a 6-element continuous action to a 7-element hyperparameter list."""
        action = self.action_value
        return [action[0], action[1], action[2], action[3], action[4], action[5], 0]

    def model_7(self):
        """Maps a 5-element continuous action to a 7-element hyperparameter list."""
        action = self.action_value
        return [action[0], action[1], 0, 0, action[2], action[3], action[4]]

    def model_8(self):
        """Maps a 6-element continuous action to a 7-element hyperparameter list."""
        action = self.action_value
        return [action[0], action[1], action[2], 0, action[3], action[4], action[5]]

    def model_9(self):
        """Passes through the 7-element continuous action value directly."""
        action = self.action_value
        return [action[0], action[1], action[2], action[3], action[4], action[5], action[6]]

    def model_10(self):
        """Maps a discrete action (0 or 1) to one of two pre-defined 7-element lists."""
        action_list = [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0]]
        return action_list[self.action_value]

    def model_11(self):
        """Maps a large discrete action (0-2186) to a unique 7-element list using base-3 conversion."""
        return int_to_list_mapping(self.action_value)

    def model_12(self):
        """Maps a discrete action (0-19) to one of 20 pre-defined 7-element lists."""
        # This is a lookup table of hand-picked hyperparameter configurations.
        action_list = [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0],
                       [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1, 1, 1], [0, 1, 0.5, 0, 0.5, 1, 0],
                       [1, 0.5, 0, 0, 1, 0, 0], [0.5, 1, 0, 0, 0, 1, 0],
                       [1, 0, 0.5, 0, 1, 0, 0], [0, 1, 0.5, 0.5, 0, 1, 0],
                       [1, 0, 0, 0.5, 1, 0, 0], [0, 1, 0.5, 0, 0, 1, 0.5],
                       [1, 0, 0, 0, 1, 0, 0], [0.5, 1, 0.5, 0.5, 0, 1, 0],
                       [1, 0, 0, 0, 1, 0.5, 0], [1, 1, 0, 1, 0, 1, 0],
                       [1, 0, 0, 0, 1, 0, 0.5], [0, 1, 1, 1, 1, 1, 0],
                       [1, 0, 0, 0, 1, 0.5, 0.5], [0, 1, 1, 1, 0, 1, 0]]
        return action_list[self.action_value]

def int_to_list_mapping(n):
    """
    Converts an integer to a unique list of 7 elements, where each element is 0, 0.5, or 1.
    This function effectively treats the integer `n` as a number in base-3.

    Args:
        n (int): An integer in the range [0, 2186] (i.e., 0 to 3^7 - 1).

    Returns:
        list: A 7-element list corresponding to the base-3 representation of n.
    """
    # Check if the input integer is within the valid range.
    if n < 0 or n >= 3**7:
        raise ValueError("Input must be in the range 0 to 2186 inclusive.")

    # Convert the integer to its base-3 string representation, padding with leading zeros to ensure a length of 7.
    base3_str = np.base_repr(n, base=3).zfill(7)

    # Map each base-3 digit ('0', '1', '2') to the corresponding value (0, 0.5, 1).
    mapping = {'0': 0, '1': 0.5, '2': 1}
    result = [mapping[digit] for digit in base3_str]

    return result
"""
Initializes the gym_DC.envs package.

This file makes the BasicDCEnv class available for import when the gym_DC.envs
package is imported, allowing for registration with the OpenAI Gym framework.
"""
# Import the main environment class to make it accessible at the package level.
from gym_DC.envs.distribution_centre_env import BasicDCEnv
"""
AirHockey v2 - A modular air hockey physics environment
"""

__version__ = "2.0.0"

# Make the main environment easily importable
from .envs.PhysicsAirHockey import PhysicsAirHockeyEnv

__all__ = ["PhysicsAirHockeyEnv"]
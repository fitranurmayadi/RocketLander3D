
print("DEBUG: Top Level Start", flush=True)
import gymnasium as gym
import numpy as np
import pybullet as p
import math
import time
import argparse
import sys
import os
from enum import Enum
print("DEBUG: Imports Done", flush=True)

# Force local import
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(curr_dir)
sys.path.insert(0, root_dir)

import rocket_lander
print("DEBUG: RocketLander Imported", flush=True)

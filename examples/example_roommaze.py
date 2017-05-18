import numpy as np
from UCRL.envs.toys import FourRoomsMaze
import UCRL.envs.RewardDistributions as RewardDistributions


dimension = 2
reward_distribution_states = RewardDistributions.ConstantReward(0)
reward_distribution_target = RewardDistributions.ConstantReward(dimension)

env = FourRoomsMaze(dimension=dimension,
                    initial_position=[dimension-1, dimension-1],
                    reward_distribution_states=reward_distribution_states,
                    reward_distribution_target=reward_distribution_target,
                    target_coordinates= [0,0],
                    success_probability=0.8)
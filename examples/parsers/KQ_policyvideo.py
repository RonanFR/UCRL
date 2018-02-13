import numpy as np
from UCRL.envs.toys import ResourceCollection
import json
import time


with open('/home/matteo/Desktop/KQ_0.01_20180205_101433/SCAL_KQ_c1/log_policy_ep_2000.json', 'r') as f:
    data = json.load(f)


policy = data['policy']
print(policy)

env = ResourceCollection()
fps = 2
env.reset()
s = env.state
env.render(mode='human')
for i in range(1000):
    # a = env.optimal_policy_indices[s]
    a = policy[s]
    env.execute(a)
    print("{} {:.3f}".format(a, env.reward))
    env.render(mode='human')
    time.sleep(1.0 / fps)
    s = env.state

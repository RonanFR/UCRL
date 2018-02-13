import numpy as np
import UCRL.envs.RewardDistributions as rdists



def test_bernoullireward(maxv=10, proba = 0.1):
    reward = rdists.BernouilliReward(r_max=maxv,proba=proba)

    n_samples = 500000
    L = [0] * n_samples
    for i in range(n_samples):
        L[i] = reward.generate()
    mean = np.mean(L)
    var = np.var(L)
    delta = 0.9
    epsilon = np.sqrt(2*var*np.log(2.0/delta) / n_samples) + 7 * np.log(2.0/delta) / (3*(n_samples-1))
    print(epsilon)
    assert np.isclose(reward.mean, mean, atol=epsilon, rtol=0.), '{}, {}'.format(reward.mean, mean)



if __name__ == '__main__':
    test_bernoullireward()

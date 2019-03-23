import numpy as np
import math
import time
from .evi import EVI, py_LPPROBA_bernstein, py_LPPROBA_hoeffding
from . import Ucrl as ucrl
from collections import namedtuple
import cvxpy as cp

MDP = namedtuple("MDP", "P R state_actions")


class BKIA(ucrl.UcrlMdp):
    """
    Optimistic Linear Programming

    Ref.
    ----
    Tewari, Ambuj, and Peter L. Bartlett.
    "Optimistic linear programming gives logarithmic regret for irreducible MDPs."
    Advances in Neural Information Processing Systems. 2008.
    """

    def __init__(self, environment, r_max,
                 alpha_r=None, alpha_p=None, bound_type_p="bernstein", bound_type_rew="bernstein",
                 verbose=0, logger=ucrl.default_logger,
                 random_state=None, known_reward=False):

        assert bound_type_p in ["KL", "hoeffding", "bernstein"]
        assert bound_type_rew in ["KL", "hoeffding", "bernstein"]

        super(BKIA, self).__init__(
            environment=environment, r_max=r_max,
            alpha_r=alpha_r, alpha_p=alpha_p,
            solver=None,
            bound_type_p=bound_type_p, bound_type_rew=bound_type_rew,
            verbose=verbose,
            logger=logger, random_state=random_state, known_reward=known_reward)

        self.nb_state_observations = np.zeros((self.environment.nb_states,))

    def _stopping_rule(self, curr_state, curr_act_idx, next_state):
        return False # episodes are of length 1

    def update_at_episode_end(self):
        super(BKIA, self).update_at_episode_end()
        self.nb_state_observations = self.nb_observations.sum(axis=1)

    def MDP_reduced_space(self, curr_state):
        S = self.environment.nb_states
        A = self.environment.max_nb_actions_per_state

        nlog = np.log(self.nb_state_observations+1)**2
        D = self.nb_observations >= nlog[:, np.newaxis]

        A_k = np.max(np.sum(D, axis=1))
        P_k = np.zeros((S, A_k, S))
        R_k = np.zeros((S, A_k))
        state_actions_k = [None] * S
        A_original_index = []

        for s in range(S):
            A_s = []
            a_counter = 0
            for a_idx, a in enumerate(self.environment.state_actions[s]):
                if self.nb_state_observations[s] == 0 or D[s, a_idx]:
                    # valid action
                    A_s.append(a)
                    P_k[s, a_counter] = self.P[s, a_idx]
                    R_k[s, a_counter] = self.estimated_rewards[s, a_idx]
                    a_counter += 1
                    if s == curr_state:
                        A_original_index.append(a_idx)
            state_actions_k[s] = A_s

        return MDP(P_k, R_k, state_actions_k), A_original_index

    # def solve_MDP(self, mdp, epsilon=1e-8):
    #         ns, na = mdp.R.shape
    #
    #         tau = 0.9
    #         t0 = time.perf_counter()
    #         gain, bias = value_iteration(self.policy_indices, self.policy,
    #                         ns, mdp.state_actions, mdp.P, mdp.R,
    #                         epsilon, tau)
    #         span_value = (np.max(bias) - np.min(bias)) / tau
    #
    #         if span_value < 0:
    #             raise ucrl.EVIException(error_value=span_value)
    #
    #         return span_value, gain, bias

    def solve_MDP(self, mdp, epsilon=1e-8):
            ns, na = mdp.R.shape

            t0 = time.perf_counter()
            solver = EVI(nb_states=ns,
                         actions_per_state=mdp.state_actions,
                         bound_type="bernstein",
                         random_state=np.random.randint(1, 1000000, 1),
                         gamma=1.
                         )
            Z = np.zeros((ns, na))
            span_value = solver.run(
                self.policy_indices, self.policy,
                mdp.P,
                mdp.R,
                Z,
                Z, np.zeros((ns, na, ns)), Z,
                self.tau_max,
                self.r_max,
                self.tau,
                self.tau_min,
                epsilon
            )
            t1 = time.perf_counter()
            tn = t1 - t0
            self.solver_times.append(tn)

            span_value *= self.tau / self.r_max

            u1, u2 = solver.get_uvectors()
            gain = 0.5 * (max(u2 - u1) + min(u2 - u1))

            if span_value < 0:
                raise ucrl.EVIException(error_value=span_value)

            return span_value, gain, u2

    def compute_optimal_actions(self, mdp, s, h, remap_actions, epsilon=1e4):
        optimal_actions = None
        maxv = 0
        n = 0

        for ai, _ in enumerate(mdp.state_actions[s]):
            v = mdp.R[s, ai] + np.dot(mdp.P[s, ai], h)
            if ai == 0 or np.abs(v-maxv) > epsilon:
                optimal_actions = [remap_actions[ai]]
                maxv = v
                n = 1
            elif np.abs(v-maxv) <= epsilon:
                optimal_actions += [remap_actions[ai]]
                maxv = (n * maxv + v) / (n+1)
                n += 1

        return optimal_actions

    def max_index(self, bias, curr_state):
        S = self.environment.nb_states
        curr_act_idx = None
        if self.bound_type_p == "KL":
            # convex optimization
            u_max = -np.inf
            for a_idx, _ in enumerate(self.environment.state_actions[curr_state]):
                q = cp.Variable(shape=S)
                p_hat = self.P[curr_state, a_idx]
                R = cp.kl_div(p_hat, q)
                gamma = np.log(self.total_time + 1) / self.nb_observations[curr_state, a_idx]
                objective = cp.Maximize(self.estimated_rewards[curr_state, a_idx] + p_hat*bias)
                constraints = [q >=0,
                               cp.sum(q) == 1,
                               R <= gamma]
                prob = cp.Problem(objective, constraints)
                prob.solve()
                print(prob.value, q.value)
                if prob.value > u_max or a_idx == 0:
                    curr_act_idx = a_idx
                    u_max = prob.value
        else:
            beta_r = self.beta_r()
            beta_p = self.beta_p()

            r_optimal = np.minimum(self.r_max, self.estimated_rewards[curr_state] + beta_r[curr_state])

            u_max = -np.inf
            asc_sort = np.argsort(bias).astype(np.int)
            opt_v = None
            for a_idx, _ in enumerate(self.environment.state_actions[curr_state]):
                if self.bound_type_p == "hoeffding":
                    opt_v = py_LPPROBA_hoeffding(v=bias, p=self.P[curr_state, a_idx],
                                                 asc_sorted_idx=asc_sort, beta=np.asscalar(beta_p[curr_state, a_idx]),
                                                 reverse=False)

                elif self.bound_type_p == "bernstein":
                    opt_v = py_LPPROBA_bernstein(v=bias, p=self.P[curr_state, a_idx],
                                                 asc_sorted_idx=asc_sort, beta=beta_p[curr_state, a_idx],
                                                 bplus=0, reverse=False)
                else:
                    raise ValueError("unknown bound type: {}".format(self.bound_type_p))
                u = opt_v + r_optimal[a_idx]
                if u > u_max or a_idx == 0:
                    curr_act_idx = a_idx
                    u_max = u

        return curr_act_idx

    def solve_optimistic_model(self, curr_state=None):

        epsilon =1e-8
        mdp_k, A_original_index = self.MDP_reduced_space(curr_state=curr_state)
        span_k, gain_k, bias_k = self.solve_MDP(mdp_k, epsilon=1e-8)

        S = self.environment.nb_states
        A = self.environment.max_nb_actions_per_state

        # Optimal actions (for the current problem) that are about to become "undersampled"
        optimal_set_k = self.compute_optimal_actions(mdp_k, curr_state, bias_k, A_original_index, 2 * epsilon)
        # note that the actions have already the original index when inserted in the optimal_set_k

        gamma_1 = [a_idx for a_idx in optimal_set_k if
                   self.nb_observations[curr_state, a_idx] < math.log(self.nb_state_observations[curr_state] + 1)]

        if len(optimal_set_k) == len(gamma_1):
            curr_act_idx = np.random.choice(gamma_1, 1)
            # na = len(self.environment.state_actions[curr_state])
            # curr_act_idx = np.argmin(self.nb_observations[curr_state, 0:na])
        else:
            curr_act_idx = self.max_index(bias=bias_k, curr_state=curr_state)

        self.policy_indices[curr_state] = curr_act_idx
        self.policy[curr_state] = self.environment.state_actions[curr_state][curr_act_idx]

        return span_k


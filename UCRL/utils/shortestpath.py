import numpy as np
import heapq


def dijkstra(P, state_actions, source):
    q = []
    parents = []
    distances = []
    start_weight = float("inf")

    n_vertex = len(state_actions)
    for i in range(n_vertex):
        weight = 0 if i == source else start_weight
        distances.append(weight)
        parents.append(np.nan)
        heapq.heappush(q, (weight, i))

    while q:
        u_tuple = heapq.heappop(q)  # get item with lowest priority
        u = u_tuple[1]

        for v in range(n_vertex):
            if v in [x[1] for x in q]:
                alt = distances[u]
                for e in state_actions[u]:
                    if P[u, e, v] > 0:
                        alt += 1 / P[u, e, v]
                if distances[u] < alt < distances[v]:
                    distances[v] = alt
                    parents[v] = u
                    # primitive but effective negative cycle detection
                    if alt < -1000:
                        raise Exception("Negative cycle detected")
                    heapq.heappush(q, (distances[v], v))
    return distances, parents

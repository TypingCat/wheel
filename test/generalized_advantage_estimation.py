#!/usr/bin/env python3

def discount_cumsum(arr, discount_factor):
    for i in range(len(arr)-2, -1, -1):
        arr[i] += discount_factor * arr[i+1]
    return arr

if __name__ == '__main__':
    gamma = 0.99
    lamda = 0.95
    val = [0., 0., 0., 0., -1., 0., 0., 1.]     # Predicted by critic
    rew = [0., 0., 0., 0., -0.5, 0., 0., 0.8]   # Observation results

    # Generalized Advantage Estimation(GAE)
    delta = [rew[i] + gamma*val[i+1] - val[i] for i in range(len(rew)-1)]
    adv = discount_cumsum(delta, gamma*lamda)
    ret = discount_cumsum(rew, gamma)

    # Results
    print(adv)
    print(ret)
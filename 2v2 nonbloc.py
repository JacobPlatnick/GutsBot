import nashpy as nash
import numpy as np
import matplotlib.pyplot as plt


def helper(p1, p2, p3, p4):
    value = np.array([0, 0, 0, 0])
    #case 1: h d d d
    temp = (1 - p1) * p2 * p3 * p4 * np.array([3, -1, -1, -1])
    value = value + temp
    #case 2: d h d d
    temp = p1 * (1 - p2) * p3 * p4 * np.array([-1, 3, -1, -1])
    value = value + temp
    # case 3: d d h d
    temp = p1 * p2 * (1 - p3) * p4 * np.array([-1, -1, 3, -1])
    value = value + temp
    # case 4: d d d h
    temp = p1 * p2 * p3 * (1 - p4) * np.array([-1, -1, -1, 3])
    value = value + temp

    # case 5: h h d d fair play
    temp = (1 - p2) * (1 - p2) * p3 * p4 * np.array([0, 0, 0, 0])
    value = value + temp
    # case 6: h h d d player 2 wins
    temp = (p2 - p1) * (1 - p2) * p3 * p4 * np.array([-4, 4, 0, 0])
    value = value + temp
    # case 7: h d h d fair play
    temp = (1 - p3) * p2 * (1 - p3) * p4 * np.array([0, 0, 0, 0])
    value = value + temp
    # case 8: h d h d player 3 wins
    temp = (p3 - p1) * p2 * (1 - p3) * p4 * np.array([-4, 0, 4, 0])
    value = value + temp
    # case 9: h d d h fair play
    temp = (1 - p4) * p2 * p3 * (1 - p4) * np.array([0, 0, 0, 0])
    value = value + temp
    # case 10: h d d h player 4 wins
    temp = (p4 - p1) * p2 * p3 * (1 - p4) * np.array([-4, 0, 0, 4])
    value = value + temp

    #case 11: d h d h fair play
    temp = p1 * (1 - p4)* p3 * (1 - p4) * np.array([0, 0, 0, 0])
    value = value + temp
    #case 12:
    temp = p1 * (p4 - p2) * p3 * (1 - p4) * np.array([0, -4, 0, 4])
    value = value + temp
    #case 13: d h h d
    temp = p1 * (1 - p3) * (1 - p3) * p4 * np.array([0, 0, 0, 0])
    value = value + temp
    #case 14:
    temp = p1 * ( p3 - p2) * (1 - p3) * p4 * np.array([0, -4, 4, 0])
    value = value + temp
    #case 15: d d h h
    temp = p1 * p2 * (1 -p4)* (1 - p4)* np.array([0, 0, 0, 0])
    value = value + temp
    #case 16:
    temp = p1 * p2 * (p4 - p3)* (1 - p4) * np.array([0, 0, -4, 4])
    value = value + temp

    # case 17: h h h d fair play
    temp = (1 - p3) * (1 - p3) * (1 - p3) * p4 * np.array([-1/3, -1/3,-1/3, 1 ])
    value = value + temp

    # case 18: h h h d player 3 wins
    temp = (p3 - p1) * (p3 - p2) * (1 - p3) * p4 * np.array([-3, -3, 5, 1])
    value = value + temp

    # case 19: h h h d player 2 loses, fair play between 1 and 3
    temp = (1 - p3) * (p3 - p2) * (1 - p3) * p4 * np.array([1, -3, 1, 1])
    value = value + temp
    # case 20: h h h d player 1 loses, fair play between 2 and 3
    temp = (p3 - p1) * (1 - p3) * (1 - p3) * p4 * np.array([-3, 1, 1, 1])
    value = value + temp

    # case 21: h d h h fair play
    temp = (1 - p4) * p2 * (1 - p4) * (1 - p4) * np.array([-1 / 3, 1, -1 / 3, -1/3])
    value = value + temp

    # case 22: h d h h player 4 wins
    temp = (p4 - p1) * p2 * (p4 - p3) * (1 - p4) * np.array([-3, 1, -3, 5])
    value = value + temp

    #case 23: h d h h player 3 loses
    temp = (1 - p4) * p2 * (p4 - p3) * (1 - p4) * np.array([1, 1, -3, 1])
    value = value + temp

    #case 24: h d h h player 1 loses
    temp = (p4 - p1) * p2 * (1 - p4) * (1 - p4) * np.array([-3, 1, 1, 1])
    value = value + temp

    #case 25: h h d h fair play
    temp = (1 - p4) * (1 - p4) * p3 * (1 - p4) * np.array([-1 / 3, -1/3, 1, -1 / 3])
    value = value + temp

    # case 26: h h d h player 4 wins
    temp = (p4 - p1) * (p4 - p2) * p3 * (1 - p4) * np.array([-3, -3, 1, 5])
    value = value + temp

    # case 27: h h d h player 2 loses
    temp = (1 - p4) * (p4 - p2) * p3 * (1 - p4) * np.array([1, -3, 1, 1])
    value = value + temp

    #case 28: h h d h player 1 loses
    temp = (p4 - p1) * (1 - p4) * p3 * (1 - p4) * np.array([-3, 1, 1, 1])
    value = value + temp

    #case 29: d h h h fair play
    temp = p1 * (1 - p4) * (1 - p4) * (1 - p4) * np.array([1, -1 / 3, -1/3, -1 / 3])
    value = value + temp
    # case 30: d h h h player 3 loses
    temp = p1 * (1 - p4) * (p4 - p3) * (1 - p4) * np.array([1, 1, -3, 1])
    value = value + temp
    #case 31: d h h h player 2 loses
    temp = p1 * (p4 - p2) * (1 - p4) * (1 - p4) * np.array([1, -3, 1, 1])
    value = value + temp

    #case 32: d h h h player 4 wins
    temp = p1 * (p4 - p2) * (p4 - p3) * (1 - p4) * np.array([1, -3, -3, 5])
    value = value + temp

    #case 33: h h h h fair play
    temp = (1 - p4) * (1 - p4) * (1 - p4) * (1 - p4) * np.array([0, 0, 0,0])
    value = value + temp
    #case 34: h h h h player 4 wins
    temp = (p4 - p1) * (p4- p2) * (p4 - p3) * (1 - p4) * np.array([-2, -2, -2, 6])
    value = value + temp
    #case 35: h h h h player 3 loses
    temp = (1 - p4) * (1 - p4) * (p4 - p3) * (1 - p4) * np.array([2/3, 2/3, -2, 2/3])
    value = value + temp
    #case 36: h h h h player 2 loses
    temp = (1 - p4) * (p4 - p2) * (1 - p4) * (1 - p4) * np.array([2 / 3, -2, 2/3, 2 / 3])
    value = value + temp
    #case 37: h h h h player 1 loses
    temp = (p4 - p1) * (1 - p4) * (1 - p4) * (1 - p4) * np.array([-2, 2 / 3, 2/3, 2 / 3])
    value = value + temp
    #case 38: h h h h player 3 and 2 lose
    temp = (1 - p4) * (p4 - p2) * (p4 - p3) * (1 - p4) * np.array([2, -2, -2, 2])
    value = value + temp
    #case 39: h h h h player 3 and 1 lose
    temp = (p4 - p1) * (1 - p4) * (p4 - p3) * (1 - p4) * np.array([-2, 2, -2, 2])
    value = value + temp
    #case 40: h h h h player 1 and 2 lose
    temp = (p4 - p1) * (p4 - p2) * (1 - p4) * (1 - p4) * np.array([-2, -2, 2, 2])
    value = value + temp

    return value

def alpha(p1, p2, p3, p4):
    strats = [p1, p2, p3, p4]
    strats.sort()

    value = helper(strats[0], strats[1], strats[2], strats[3])
    p1index = strats.index(p1)
    p2index = strats.index(p2)
    p3index = strats.index(p3)
    p4index = strats.index(p4)
    return [value[p1index], value[p2index], value[p3index], value[p4index]]

N = 21
M = 21
A = [[alpha((i // N)/ (N - 1),(i % N)/(N - 1), (j // M) / (M - 1), (j % M) / (M - 1))[0] + alpha((i // N)/ (N - 1),(i % N)/(N - 1), (j // M) / (M - 1), (j % M) / (M - 1))[1]
      for j in range(M ** 2)] for i in range(N**2)]

for iterations in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
    gts = nash.Game(A)
    np.random.seed(0)
    play_counts = tuple(gts.fictitious_play(iterations=iterations))

    row = play_counts[-1][0]
    col = play_counts[-1][1]
    row_halfway = play_counts[int(iterations / 2)][0]
    col_halfway = play_counts[int(iterations / 2)][1]
    p1_strats = []
    p1_usage = []
    opponent_strats = []
    opponent_usage = []
    threshold = 0.1
    for i in range(N * N):
        if row[i] - row_halfway[i] > threshold:
            p1_strats.append(((i % N) / (N - 1), (i - (i % N)) / (N * (N - 1))))
            p1_usage.append(row[i] - row_halfway[i])
    for i in range(M * M):
        if col[i] - col_halfway[i] > threshold:
            opponent_strats.append(((i % M) / (M - 1), (i - (i % M)) / (M * (M - 1))))
            opponent_usage.append(col[i] - col_halfway[i])
    print(iterations)
    print(p1_strats)
    print(p1_usage)
    print(opponent_strats)
    print(opponent_usage)


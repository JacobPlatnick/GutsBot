#fp for 3 player
#using bestResponse function
#"""Code to carry out fictitious learning"""
import numpy as np

def alpha(p1, p2, p3):
    large = max(p2, p3)
    small = min(p2, p3)
    if p1 <= small:
        return 2 * p1 - small - large + large ** 3 + 3 * small ** 2 * large - 4 * p1 * small * large
    if p1 >= large:
        return 2 * p1 - small - large - 2 * (p1 ** 3) + 2 * p1 * large * small

    return 2 * p1 - small - large + large ** 3 - 3 * (p1 ** 2) * large + 2 * p1 * p2 * p3

N = 101
A = np.array([[alpha(i / (N - 1), (j // N) / (N - 1), (j % N) / (N- 1)) for j in range(N ** 2)] for i in range(N)])

def get_best_response_to_play_count(A, play_count):
    """
    Returns the best response to a belief based on the playing distribution of the opponent
    """
    utilities = A @ play_count
    return np.random.choice(
        np.argwhere(utilities == np.max(utilities)).transpose()[0]
    )


def update_play_count(play_count, play):
    """
    Update a belief vector with a given play
    """
    extra_play = np.zeros(play_count.shape)
    extra_play[play] = 1
    return play_count + extra_play

def update_coalition_count(coalition_count, play1, play2, n):
    """
    Update the coalition belief vector with given play
    """
    coalition_count[play1 * n + play2] += 1
    return coalition_count



def three_player_fictitious_play(A, B, C, iterations, play_counts=None):
    """
    Implement fictitious play
    A = matrix for p1 against coalition p2, p3
    B = matrix for p2 against coalition p1, p3
    C = matrix for p3 against coalition p1, p2
    """
    p1stratcount = A.shape[0]
    p2stratcount = B.shape[0]
    p3stratcount = C.shape[0]
    if play_counts is None:
        play_counts = [
            np.zeros(p1stratcount),
            np.zeros(p2stratcount),
            np.zeros(p3stratcount),
            np.zeros(A.shape[1]), #players 2+3
            np.zeros(B.shape[1]), #players 1+3
            np.zeros(C.shape[1]), #players 1+2
        ]

    yield play_counts

    for repetition in range(iterations):

        plays = [
            get_best_response_to_play_count(matrix, play_count)
            for matrix, play_count in zip((A, B, C), play_counts[3:])
        ]

        play_counts[0:3] = [
            update_play_count(play_count, play)
            for play_count, play in zip(play_counts[0:3], plays)
        ]
        play_counts[3:] = [
            update_coalition_count(play_counts[3], plays[1], plays[2], p2stratcount),
            update_coalition_count(play_counts[4], plays[0], plays[2], p1stratcount),
            update_coalition_count(play_counts[5], plays[0], plays[1], p1stratcount)
        ]
        yield play_counts

iterations = 10000
play_counts = tuple(three_player_fictitious_play(A, A, A, iterations))
p1 = play_counts[-1][0] / iterations
p2 = play_counts[-1][1] / iterations
p3 = play_counts[-1][2] / iterations

p1_strats = []
p1_usage = []
p2_strats = []
p2_usage = []
p3_strats = []
p3_usage = []

for i in range(N):
    if p1[i] > 0.01:
        p1_strats.append(i/(N - 1))
        p1_usage.append(p1[i])
    if p2[i] > 0.01:
        p2_strats.append(i/(N - 1))
        p2_usage.append(p2[i])
    if p3[i] > 0.01:
        p3_strats.append(i/(N - 1))
        p3_usage.append(p3[i])
print(p1_strats, p1_usage)
print(p2_strats, p2_usage)
print(p3_strats, p3_usage)
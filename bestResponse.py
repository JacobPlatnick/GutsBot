# finds best response for player 1 against fixed strategies (could be blended) of players 2-N
import numpy as np
from itertools import combinations
import scipy
from scipy.optimize import minimize
def findPlayers(n,dropPlayers):
    players = []
    for i in range(n):
        if i not in dropPlayers:
            players = players + [i]
    return players
def helper(orderedStrats, n):
    value = 0
    indexList = np.array([i for i in range(n)])
    for dropNum in range(n):
        dropPlayerCombs = combinations(indexList, dropNum)
        holdNum = n - dropNum
        for dropPlayers in dropPlayerCombs:
            players = findPlayers(n, dropPlayers)
            maxStrat = orderedStrats[players[-1]]
            for loserNum in range(holdNum):
                fairNum = holdNum - loserNum
                loserPlayerCombs = combinations(players[:-1], loserNum)
                for losers in loserPlayerCombs:
                    temp = np.zeros(n)
                    prob = (1 - maxStrat) ** fairNum
                    for dropPlayer in dropPlayers:
                        prob *= orderedStrats[dropPlayer]
                        temp[dropPlayer] = -1 + holdNum - 1
                    for loser in losers:
                        prob *= (maxStrat - orderedStrats[loser])
                        temp[loser] = -1 - n + holdNum - 1
                    for player in players:
                        if player not in losers:
                            temp[player] = (-1 + n + holdNum - 1 + (fairNum - 1) * (-1 - n + holdNum - 1)) / fairNum
                    value = value + prob * temp
    return value
def alpha(strats):
    n = len(strats)

    orderedStrats = sorted(strats)
    value = helper(orderedStrats, n)
    sortedValue = np.zeros(n)
    for i in range(n):
        p = strats[i]
        pIndex = orderedStrats.index(p)
        sortedValue[i] = value[pIndex]
    return sortedValue
def beta(strats):
    n = len(strats)
    value = np.prod(strats)
    indexList = np.array([i for i in range(n)])
    for dropNum in range(n - 1):
        dropPlayerCombs = combinations(indexList, dropNum)
        holdnum = n - dropNum
        for dropPlayers in dropPlayerCombs:
            prob = 1
            players = findPlayers(n, dropPlayers)
            for dropPlayer in dropPlayers:
                prob *= strats[dropPlayer]
            for player in players:
                prob *= (1 - strats[player])
            value += (holdnum - 1) * prob
    return value

N = 101
x0 = [1/N for i in range(N)]

def con1(X):
    return np.sum(X) - 1
#n is total number of players, strats is a list of strategies played by the (n - 1) players
def bestResponse(n, strats, stratProbs, Value):
    m = n - 1
    Alpha = np.zeros(N)
    Beta = np.zeros(N)
    for i in range(N):
        a = 0
        b = 0
        for j in range(len(stratProbs)):
            x = strats[j]
            temp = alpha(np.concatenate(([i/(N - 1)], x)))
            a = a + temp[0] * stratProbs[j]
            b = b + beta(np.concatenate(([i/(N - 1)], x))) * stratProbs[j]
        Alpha[i] = a
        Beta[i] = b
    def fun(X):
        return -(np.dot(Alpha, X) + np.dot(Beta, X) * Value)

    cons = [{'type': 'eq', 'fun': con1}]
    bounds = [[0, 1] for i in range(N)]
    value = scipy.optimize.minimize(fun, x0, method='SLSQP', bounds=bounds, constraints=cons)
    return value

response = bestResponse(3, [[0.67, 0.67], [0, 0.86]], [6.3/7.3, 1/7.3], - 1)
x = response.x
strats = []
probs = []
for i in range(N):
    if x[i] > 0.01:
        strats.append(i/(N - 1))
        probs.append(x[i])
print(strats, probs)


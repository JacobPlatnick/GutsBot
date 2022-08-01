import numpy as np
import nashpy as nash
from itertools import combinations
import joblib
def findPlayers(n,dropPlayers):
    players = []
    for i in range(n):
        if i not in dropPlayers:
            players = players + [i]
    return players
def helper(orderedStrats, n, weenie):
    value = 0
    if weenie:
        smallest = orderedStrats[0]
        for i in range(1, n):
            temp = np.repeat(1,n)
            temp[i] = 1 - n
            prob = smallest ** (n - 1) * (orderedStrats[i] - smallest)
            value = value + prob * temp
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

#weenie = True if weenie rule is activated
def generalizedAlpha(strats, weenie = False):
    n = len(strats)

    orderedStrats = sorted(strats)
    value = helper(orderedStrats, n, weenie)
    sortedValue = np.zeros(n)
    for i in range(n):
        p = strats[i]
        pIndex = orderedStrats.index(p)
        sortedValue[i] = value[pIndex]
    return sortedValue[0]
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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 13:22:53 2022

@author: jacobplatnick
"""
"""This should now use V correctly
"""
import nashpy as nash
import numpy as np
import matplotlib.pyplot as plt
import joblib
from itertools import combinations
block_size=3


def alpha(p1, p2, p3):
  return multialpha([p1,p2]+[p3 for i in range(block_size)])

def beta(p1,p2,p3):
  return multibeta([p1,p2]+[p3 for i in range(block_size)])
def Val(p1,p2,p3,Vlength):
  V=0
  for i in range(Vlength):
      V = alpha(p1,p2,p3)+beta(p1,p2,p3)*V
  return V

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
def multialpha(strats):
    n = len(strats)

    orderedStrats = sorted(strats)
    value = helper(orderedStrats, n)
    sortedValue = np.zeros(n)
    for i in range(n):
        p = strats[i]
        pIndex = orderedStrats.index(p)
        sortedValue[i] = value[pIndex]
    return sortedValue[0]
def multibeta(strats):
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

def getV(p1,p2,p3,Vlast):
  return alpha(p1,p2,p3)+beta(p1,p2,p3)*Vlast
def getOppV(p1,p2,p3,Vlast):
  return -1*alpha(p1,p2,p3)+beta(p1,p2,p3)*Vlast

def play(Alphas,Betas,N=201,M=201,iterations=10000,Vprevious=0,VpreviousOpp=0,number=0,saving=False,threshold=0.1):
  A=Alphas+Vprevious*Betas
  B=-1*Alphas+VpreviousOpp*Betas
  s1 = [0 for _ in range(N)]
  s2 = [0 for _ in range(M)]
  s1[-2] = 1
  s2[-1] = 1
  
  
  gts = nash.Game(A,B)
  
  np.random.seed(0)
  play_counts = tuple(gts.fictitious_play(iterations = iterations))
  #print(play_counts[-1])
  
  #plt.figure()
  #probabilities = [row_play_counts / (np.sum(row_play_counts) + 1) for row_play_counts, col_play_counts in play_counts]
  #for number, strategy in enumerate(zip(*probabilities)):
  #    plt.plot(strategy, label = f"$s_{number}$")
  #plt.show()
  
  #tempr = -1
  #outputr = 0
  row = play_counts[-1][0]
  col = play_counts[-1][1]
  row_halfway=play_counts[int(iterations/2)][0]
  col_halfway=play_counts[int(iterations/2)][1]
  p1_strats=[]
  p1_usage=[]
  opponent_strats=[]
  opponent_usage=[]
  p1_all_strats=[]
  p1_all_usage=[]
  opponent_all_strats=[]
  opponent_all_usage=[]
  if threshold>=1:
      for i in range(N):
          p1_all_strats.append(i/(N - 1))
          p1_all_usage.append(row[i]-row_halfway[i])
          if row[i]-row_halfway[i] > threshold:
              #tempr = row[i]
              p1_strats.append( i/(N - 1))
              p1_usage.append(row[i]-row_halfway[i])
      for i in range(M*M):
          opponent_all_strats.append(((i%M)/(M-1),(i-(i%M))/(M*(M-1))))
          opponent_all_usage.append(col[i]-col_halfway[i]) 
          if col[i]-col_halfway[i] > threshold:
              #opponent_strats.append((i//M)/(M - 1))
              #opponent_strats.append((i%M)/(M - 1))
              opponent_strats.append(((i%M)/(M-1),(i-(i%M))/(M*(M-1))))
              opponent_usage.append(col[i]-col_halfway[i]) 
  else:
      for i in range(N):
          if row[i]-row_halfway[i] > threshold:
              #tempr = row[i]
              p1_strats.append( i/(N - 1))
              p1_usage.append(row[i]-row_halfway[i])
      for i in range(M*M):
          if col[i]-col_halfway[i] > threshold:
              #opponent_strats.append((i//M)/(M - 1))
              #opponent_strats.append((i%M)/(M - 1))
              opponent_strats.append(((i%M)/(M-1),(i-(i%M))/(M*(M-1))))
              opponent_usage.append(col[i]-col_halfway[i]) 
  if saving:
      joblib.dump(play_counts, "play_counts"+str(number)+".txt")
  print("round number: "+str(number))  
  print(p1_strats)
  print(p1_usage)
  print(opponent_strats)
  print(opponent_usage)
  Vsum=0
  Asum=0
  Bsum=0
  if threshold>=1:
      for i in range(len(p1_all_strats)):
        for j in range(len(opponent_all_strats)):
          Vsum+=getV(p1_all_strats[i],opponent_all_strats[j][0],opponent_all_strats[j][1],Vprevious)*p1_all_usage[i]*opponent_all_usage[j]
          Asum+=alpha(p1_all_strats[i],opponent_all_strats[j][0],opponent_all_strats[j][1])*p1_all_usage[i]*opponent_all_usage[j]
          Bsum+=beta(p1_all_strats[i],opponent_all_strats[j][0],opponent_all_strats[j][1])*p1_all_usage[i]*opponent_all_usage[j]
  else:
      for i in range(len(p1_strats)):
        for j in range(len(opponent_strats)):
          Vsum+=getV(p1_strats[i],opponent_strats[j][0],opponent_strats[j][1],Vprevious)*p1_usage[i]*opponent_usage[j]
          Asum+=alpha(p1_strats[i],opponent_strats[j][0],opponent_strats[j][1])*p1_usage[i]*opponent_usage[j]
          Bsum+=beta(p1_strats[i],opponent_strats[j][0],opponent_strats[j][1])*p1_usage[i]*opponent_usage[j]
  #Vsum=Vsum/(int(iterations/2)**2)
  Asum=Asum/(int(iterations/2)**2)
  Bsum=Bsum/(int(iterations/2)**2)
  OppVsum=0
  if threshold>=1:
        for i in range(len(p1_all_strats)):
          for j in range(len(opponent_all_strats)):
            OppVsum+=getOppV(p1_all_strats[i],opponent_all_strats[j][0],opponent_all_strats[j][1],VpreviousOpp)*p1_all_usage[i]*opponent_all_usage[j]
  else:
        for i in range(len(p1_strats)):
          for j in range(len(opponent_strats)):
            OppVsum+=getOppV(p1_strats[i],opponent_strats[j][0],opponent_strats[j][1],VpreviousOpp)*p1_usage[i]*opponent_usage[j]

  Vsum=Vsum/(int(iterations/2)**2)
  OppVsum=OppVsum/(int(iterations/2)**2)
  print("average val: "+str(Vsum))
  print("average alpha: "+str(Asum))
  print("average beta: "+str(Bsum))
  print("Opponent average val: "+str(OppVsum))  
  return(Vsum,OppVsum,"round number: "+str(number),p1_strats,p1_usage,opponent_strats,opponent_usage,"average val: "+str(Vsum),"Opponent average val: "+str(OppVsum),"average val: "+str(Vsum),"average alpha: "+str(Asum),"average beta: "+str(Bsum))

#loaded_play_counts = joblib.load("play_counts.txt")

rounds=1000
curV=-1
oppV=-1*(block_size+1)
filename="nothin.txt"
N=21
M=21
open(filename, 'w').close()
A = np.array([[alpha(i/(N - 1), (j%M)/(M - 1),(j//M)/(M - 1)) for j in range(M**2)] for i in range(N)])
print("zoom")
B = np.array([[beta(i/(N - 1), (j%M)/(M - 1), (j//M)/(M - 1)) for j in range(M**2)] for i in range(N)])
print("whee")

for roundnum in range(rounds):
  results=play(A,B,Vprevious=curV,VpreviousOpp=oppV,number=roundnum,iterations=10000,N=N,M=M)
  curV=results[0]
  oppV=results[1]
  file_object=open(filename, "a+")
  for i in range(2,len(results)):
      file_object.write(str(results[i]))
      file_object.write("\n")
  file_object.close()
  if curV<=-1:
    file_object=open(filename, "a+")
    print("player 1 leaves on round "+str(roundnum))
    file_object.write("player 1 leaves on round "+str(roundnum))
    file_object.close()
    break


#print(output)
#print(gts[s1, s2])
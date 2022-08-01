#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 13:37:28 2022

@author: jacobplatnick
"""
import itertools
def victor(hand,opp):
    if hand[0] in opp or hand[1] in opp:
        return -1
    hpair=hand[0][0]==hand[1][0]
    opair=opp[0][0]==opp[1][0]
    if hpair != opair:
        if hpair:
            return 0
        return 1
    if hand[0][0]>opp[0][0]:
        return 0
    if hand[0][0]<opp[0][0]:
        return 1
    if hand[1][0]>opp[1][0]:
        return 0
    if hand[1][0]<opp[1][0]:
        return 1
    if hand[0][1]>opp[0][1]:
        return 0
    if hand[0][1]<opp[0][1]:
        return 1
    raise(ValueError("oops lol"))

def findsubsets(S,m):
    return set(itertools.combinations(S, m))
cardlist=[(i,j) for i in range(13) for j in range(4)]
cardset=set(cardlist)
prehands=findsubsets(cardset,2)
hands=[]
for hand in prehands:
    if hand[0][0]>hand[1][0]:
        hands.append(hand)
    elif hand[0][0]<hand[1][0]:
        hands.append((hand[1],hand[0]))
    elif hand[0][1]>hand[1][1]:
        hands.append(hand)
    elif hand[0][1]<hand[1][1]:
        hands.append((hand[1],hand[0]))
    else:
        raise(ValueError("oops2"))
handvictdict={}
for hand in hands:
    vict=0
    for opp in hands:
        if victor(hand,opp)==0:
            vict+=1
    handvictdict[hand]=vict
print("here")
for h1 in hands:
    for h2 in hands:
        if victor(h1,h2)==0 and handvictdict[h1]<=handvictdict[h2]:
            print(h1)
            print(h2)
            print(handvictdict[h1])
            print(handvictdict[h2])
        if victor(h1,h2)==1 and handvictdict[h2]<=handvictdict[h1]:
            print(h1)
            print(h2)
            print(handvictdict[h1])
            print(handvictdict[h2])


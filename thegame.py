import random
discretization=4
oppstrats_list=[False,([1,[0.5]],),([.132,[0,0.86]],[1,[0.67,0.68]]),([0.2084,[0,0.89,0.89]],[1,[0.76,0.76,0.76]]),([0.25,[0,0.91,0.91,0.91]],[1,[0.81,0.81,0.81,0.81]]),([0.254,[0,.93,.93,.93,.93]],[1,[.84,.84,.84,.84,.84]])]
opponent_count=int(input("How many opponents would you like? Select a number from 1 to "+str(len(oppstrats_list)-1)+". "))
cycle_length=0
balances=[-1.0,-1.0*opponent_count]
pot=opponent_count+1.0
names=["You"]+["Player "+str(i+2) for i in range(opponent_count)]
current_strats=[0 for i in range(opponent_count+1)]
while True:
    print("")
    print("Current pot: "+str(pot))
    print("Your balance: "+str(balances[0]))
    print("Total opponent balance: "+str(balances[1]))
    print("")
    hands=[round(random.uniform(0, 1),discretization) for i in range(opponent_count+1)]
    if cycle_length<=0:
        read=input("Your hand is "+str(hands[0])+". That means that it is better than that proportion of all hands. Enter 0 to hold, 1 to fold, or i to input a strategy and simulate future rounds. ")
        if read=="i":
            current_strats[0]=float(input("input a strategy: a number from 0 to 1. You will hold on any hand better than that proportion of hands. "))
            cycle_length=int(input("For how many hands would you like to play that strategy? "))
        else:
            current_strats[0]=float(read)
    cycle_length-=1
    print("")
    possible_strats=oppstrats_list[opponent_count]
    index=random.uniform(0,1)
    for s in possible_strats:
        if index<=s[0]:
            for i in range(len(s[1])):
                current_strats[i+1]=s[1][i]
            break
    holders=[current_strats[i]<=hands[i] for i in range(len(hands))]
    if True not in holders:
        print("Everyone folds.")
    else:
        winhand=-1
        winner=-1
        for i in range(len(holders)):
            if holders[i]:
                if hands[i]>winhand:
                    winhand=hands[i]
                    winner=i
                if i>0:    
                    print(names[i]+" holds with "+str(hands[i]))
                else:
                    print(names[i]+" hold with "+str(hands[i]))
            else:
                if i>0:    
                    print(names[i]+" folds")
                else:
                    print(names[i]+" fold")
        print("")
        if winner>0:
            print(names[winner]+" wins!")
        else:
            print(names[winner]+" win!")
        balances[min(1,winner)]+=pot
        oldpot=pot
        pot=0
        for i in range(len(holders)):
            if holders[i] and i!= winner:
                balances[min(1,i)]-=oldpot
                pot+=oldpot
        if pot==0:
            print("All players add antes to the pot.")
            pot=opponent_count+1
            balances[0]-=1
            balances[1]-=opponent_count
        


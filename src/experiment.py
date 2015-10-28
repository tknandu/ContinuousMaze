# 
# Copyright (C) 2008, Brian Tanner
# 
#http://rl-glue-ext.googlecode.com/
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys
import numpy as np
import math
import pickle
import rlglue.RLGlue as RLGlue

from sys import argv
#script, goal = argv

gamma       = 1
NO_RUNS     = 50

NO_EPISODES = 100

TIME_LIMIT  = 100000


whichEpisode=0
def runEpisode(stepLimit):
    # stepLimit of 0 implies no limit
    global whichEpisode
    terminal=RLGlue.RL_episode(stepLimit)
    totalSteps=RLGlue.RL_num_steps()
    totalReward=RLGlue.RL_return()
    
    print "Episode "+str(whichEpisode)+"\t "+str(totalSteps)+ " steps \t" + str(totalReward) + " total reward\t " 
    
    whichEpisode=whichEpisode+1

RLGlue.RL_init()
#RLGlue.RL_env_message("dumptmatrix tmatrixperfect.dat")
#RLGlue.RL_env_message("printabstractstates")
for i in xrange(NO_EPISODES):
    runEpisode(0)
 
'''
returnVsEpisode = np.zeros(NO_EPISODES)
timeVsEpisode = np.zeros(NO_EPISODES)

def calculateCoords(state):
    return [state%12,state/12]

policy = [12*[4*[0]] for i in xrange(12)]

for run in xrange(NO_RUNS):
    print "Run: "+str(run+1)
    RLGlue.RL_init()

    for episode in xrange(NO_EPISODES):
        #if not i==0:
        #   RLGlue.RL_agent_message("load_policy results.dat");
        currentReturn = 0.0
        startResponse = RLGlue.RL_start()

        stepResponse = RLGlue.RL_step()
        currentReturn += stepResponse.r

        timeStep = 1
        while (stepResponse.terminal != 1 and timeStep<=TIME_LIMIT):
            stepResponse = RLGlue.RL_step()
            currentReturn += (stepResponse.r * math.pow(gamma,timeStep))
            timeStep += 1

        currentSteps = RLGlue.RL_num_steps()
        #totalReward = RLGlue.RL_return()
        print "\tEpisode "+str(episode+1)+"\t "+str(currentSteps)+ " steps \t" + str(currentReturn/timeStep) + " average reward\t "
        #RLGlue.RL_agent_message("save_policy results.dat");

        returnVsEpisode[episode] += (currentReturn/timeStep)
        timeVsEpisode[episode] += currentSteps

    """
    RLGlue.RL_agent_message("save_policy value_function.dat");
    theFile = open("value_function.dat", "r")
    value_function=pickle.load(theFile)
    theFile.close()

    theFile = open("/Users/krishnamurthythangavel/Documents/8-Sem/RL/Assignment2/Assignment_2/question1/results/"+goal+"/average_reward_"+str(run)+".dat", "w")
    pickle.dump(returnVsEpisode, theFile)
    theFile.close()

    theFile = open("/Users/krishnamurthythangavel/Documents/8-Sem/RL/Assignment2/Assignment_2/question1/results/"+goal+"/average_time_"+str(run)+".dat", "w")
    pickle.dump(timeVsEpisode, theFile)
    theFile.close()
    """

    for s in xrange(144):
        a = value_function[s].index(max(value_function[s]))
        coords = calculateCoords(s)
        policy[coords[0]][coords[1]][a]+=1

    RLGlue.RL_cleanup()


print "\nSummary Resuls:\n"
for episode in xrange(NO_EPISODES):
    timeVsEpisode[episode] /= NO_RUNS
    returnVsEpisode[episode] = float(returnVsEpisode[episode])/NO_RUNS
    print "\tEpisode "+str(episode+1)+"\t "+str(timeVsEpisode[episode])+ " steps \t" + str(returnVsEpisode[episode]) + " average return\t "

"""
theFile = open("/Users/krishnamurthythangavel/Documents/8-Sem/RL/Assignment2/Assignment_2/question1/results/"+goal+"/final_average_reward.dat", "w")
pickle.dump(returnVsEpisode, theFile)
theFile.close()

theFile = open("/Users/krishnamurthythangavel/Documents/8-Sem/RL/Assignment2/Assignment_2/question1/results/"+goal+"/final_average_time.dat", "w")
pickle.dump(timeVsEpisode, theFile)
theFile.close()

optimal_policy = [12*['o'] for i in xrange(12)]

print "\nOptimal Policy:\n"
for i in xrange(12):
    for j in xrange(12):
        a = policy[i][j].index(max(policy[i][j]))
        if a==0:
            optimal_policy[i][j] = 'L'
        elif a==1:
            optimal_policy[i][j] = 'R'
        elif a==2:
            optimal_policy[i][j] = 'U'
        elif a==3:
            optimal_policy[i][j] = 'D'
    print optimal_policy[i]

theFile = open("/Users/krishnamurthythangavel/Documents/8-Sem/RL/Assignment2/Assignment_2/question1/results/"+goal+"/final_policy.dat", "w")
pickle.dump(optimal_policy, theFile)
theFile.close()

# Plotting 

episodes = [i for i in range(NO_EPISODES)]
import matplotlib.pyplot as plt 

f1 = plt.figure(1)
plt.xlabel("Episodes")
plt.ylabel("Average Time Steps per Episode")
plt.plot(episodes,timeVsEpisode)
f1.show()

f2 = plt.figure(2)
plt.xlabel("Episodes")
plt.ylabel("Average Reward per Episode")
plt.plot(episodes,returnVsEpisode)
f2.show()

raw_input()
"""


RLGlue.RL_init()
for i in xrange(NO_EPISODES):
    runEpisode(0)
'''
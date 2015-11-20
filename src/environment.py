import random
import sys
from rlglue.environment.Environment import Environment
from rlglue.environment import EnvironmentLoader as EnvironmentLoader
from rlglue.types import Observation
from rlglue.types import Action
from rlglue.types import Reward_observation_terminal
import pickle
import numpy as np
import math 
from scipy.spatial import distance

# This is a very simple discrete-state, episodic grid world that has 
# exploding mines in it.  If the agent steps on a mine, the episode
# ends with a large negative reward.
# 
# The reward per step is -1, with +10 for exiting the game successfully
# and -100 for stepping on a mine.


# TO USE THIS Environment [order doesn't matter]
# NOTE: I'm assuming the Python codec is installed an is in your Python path
#   -  Start the rl_glue executable socket server on your computer
#   -  Run the SampleSarsaAgent and SampleExperiment from this or a
#   different codec (Matlab, Python, Java, C, Lisp should all be fine)
#   -  Start this environment like:
#   $> python sample_mines_environment.py

class threeroom_environment(Environment):
    
    SIZE_WORLD = 10  
    FIXED_DISTANCE = 1

    randGenerator = random.Random()

    def env_init(self):
        self.END_STATE = [0.6*self.SIZE_WORLD, 0.6*self.SIZE_WORLD]

        #The Python task spec parser is not yet able to build task specs programmatically
        return "VERSION RL-Glue-3.0 PROBLEMTYPE episodic DISCOUNTFACTOR 0.9 OBSERVATIONS DOUBLES ([times-to-repeat-this-tuple=2]) (0 10) ACTIONS DOUBLES (0 7) REWARDS (-3.0 10.0) EXTRA SampleMinesEnvironment(C/C++) by Brian Tanner."

    def env_start(self):
        self.setStartState()
        returnObs=Observation()
        returnObs.doubleArray=[self.agentRow, self.agentCol]
        return returnObs
        
    def env_step(self,thisAction):
        #print self.agentRow, self.agentCol
        hitBoundary = self.updatePosition(thisAction.doubleArray[0])

        theObs=Observation()
        theObs.doubleArray=[self.agentRow, self.agentCol]

        returnRO=Reward_observation_terminal()
        returnRO.r=self.calculateReward(hitBoundary)
        returnRO.o=theObs
        returnRO.terminal=self.checkCurrentTerminal()

        return returnRO

    def env_cleanup(self):
        pass

    def env_message(self,inMessage):
        #   Message Description
        #   'print-state'
        #   Action: Print the map and the current agent location
        if inMessage.startswith("print-state"):
            self.printState()
            return "Message understood.  Printed the state."

        return "SamplesMinesEnvironment(Python) does not respond to that message."

    #TODO: Set random start state
    def setStartState(self):
        x = 0.2 * self.SIZE_WORLD
        y = 0.2 * self.SIZE_WORLD
        self.setAgentState(x,y)

    def setAgentState(self,row, col):
        self.agentRow=row
        self.agentCol=col

        assert self.checkValid(row,col)

    def checkValid(self,row, col):
        return row > 0 and row < self.SIZE_WORLD and col > 0 and col < self.SIZE_WORLD

    def checkTerminal(self,row,col):
        return distance.euclidean([self.agentRow,self.agentCol], self.END_STATE)<0.5*self.FIXED_DISTANCE

    def checkCurrentTerminal(self):
        return self.checkTerminal(self.agentRow,self.agentCol)

    def updatePosition(self, theAction):
        # When the move would result in hitting an obstacles, the agent simply doesn't move 

        newRow = self.agentRow
        newCol = self.agentCol

        newRow += self.FIXED_DISTANCE*math.cos(theAction)
        newCol += self.FIXED_DISTANCE*math.sin(theAction)

        #Check if new position is out of bounds or inside an obstacle 
        if self.checkValid(newRow,newCol):
            self.agentRow = newRow
            self.agentCol = newCol
            return False
        else:
            return True

    def calculateReward(self, hitBoundary):
        if hitBoundary:
            return -0.5
        if(distance.euclidean([self.agentRow,self.agentCol], self.END_STATE)<0.5*self.FIXED_DISTANCE):
            return 10.0
        return 0.0

if __name__=="__main__":
    EnvironmentLoader.loadEnvironment(threeroom_environment())

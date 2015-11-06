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
        self.END_STATE = [0.8*self.SIZE_WORLD, 0.8*self.SIZE_WORLD]

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
        # 'set-random-start-state'
        #Action: Set flag to do random starting states (the default)
#       if inMessage.startswith("set-random-start-state"):
#           self.fixedStartState=False;
#           return "Message understood.  Using random start state.";

        #   Message Description
        # 'set-start-state X Y'
        # Action: Set flag to do fixed starting states (row=X, col=Y)
        """
        if inMessage.startswith("set-start-state"):
            splitString=inMessage.split(" ");
            self.startRow=int(splitString[1]);
            self.startCol=int(splitString[2]);
            self.fixedStartState=True;
            return "Message understood.  Using fixed start state.";
        """

        #   Message Description
        #   'print-state'
        #   Action: Print the map and the current agent location
        if inMessage.startswith("print-state"):
            self.printState()
            return "Message understood.  Printed the state."

        if inMessage.startswith("printabstractstates"):
            self.printAbstractStates()
         
        if inMessage.startswith("dumptmatrix"):

            tmatrix = np.zeros((len(self.validstates),len(self.validstates)),dtype=np.float)
            for (state_i,state) in enumerate(self.validstates):
                valid_successors = []
                (row,col) = self.calculateCoordsFromFlatState(state)
                if self.checkValid(row,col+1):
                    valid_successors.append(self.calculateFlatState(row,col+1))
                if self.checkValid(row,col-1):
                    valid_successors.append(self.calculateFlatState(row,col-1))
                if self.checkValid(row+1,col):
                    valid_successors.append(self.calculateFlatState(row+1,col))
                if self.checkValid(row-1,col):
                    valid_successors.append(self.calculateFlatState(row-1,col))
                for vs in valid_successors:
                    tmatrix[state_i][self.validstates.index(vs)] = float(1)/len(valid_successors)
 
            splitstring = inMessage.split()
            outfile = open(splitstring[1],'w')
            pickle.dump(tmatrix,outfile)

        if inMessage.startswith("dumppmatrix"):

            pmatrix = np.zeros((len(self.validstates),4,len(self.validstates)),dtype=np.float)
            for (state_i,state) in enumerate(self.validstates):
                (row,col) = self.calculateCoordsFromFlatState(state)
                if self.checkValid(row,col-1): #Left
                    pmatrix[state_i][0][self.validstates.index(self.calculateFlatState(row,col-1))] = 1
                else:
                    pmatrix[state_i][0][state_i] = 1

                if self.checkValid(row,col+1): #Right
                    pmatrix[state_i][1][self.validstates.index(self.calculateFlatState(row,col+1))] = 1
                else:
                    pmatrix[state_i][1][state_i] = 1
 
                if self.checkValid(row-1,col): #Up
                    pmatrix[state_i][2][self.validstates.index(self.calculateFlatState(row-1,col))] = 1
                else:
                    pmatrix[state_i][2][state_i] = 1
 
                if self.checkValid(row+1,col): #Down
                    pmatrix[state_i][3][self.validstates.index(self.calculateFlatState(row+1,col))] = 1
                else:
                    pmatrix[state_i][3][state_i] = 1
 
 
            print 'HH'
            splitstring = inMessage.split()
            outfile = open(splitstring[1],'w')
            pickle.dump(pmatrix,outfile)

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
            return 1.0
        return 0.0
        
    def printState(self):
        numRows=len(self.worldmap)
        numCols=len(self.worldmap[0])

        print "Agent is at: "+str(self.agentRow)+","+str(self.agentCol)
        print "Columns:0-10                10-17"
        print "Col    ",
        for col in range(0,numCols):
            print col%10,
            
        for row in range(0,numRows):
            print
            print "Row: "+str(row)+" ",
            for col in range(0,numCols):
                if self.agentRow==row and self.agentCol==col:
                    print "A",
                else:
                    if self.map[row][col] == self.GOAL:
                        print "G",
                    if self.map[row][col] == self.WALL:
                        print "X",
                    if self.map[row][col] == self.START:
                        print "S",
                    if self.map[row][col] == self.FREE:
                        print " ",
        print

    def printAbstractStates(self):
        argmaxes = [2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0]
        numRows=len(self.worldmap)
        numCols=len(self.worldmap[0])
        for row in range(0,numRows):
            print
            print "Row: "+str(row)+" ",
            for col in range(0,numCols):
                if self.checkValid(row,col):
                    flat = self.calculateFlatState(row,col)
                    flat_i = self.validstates.index(flat)
                    print unicode(argmaxes[flat_i]),
                else:
                    print "X",

                """
                if self.map[row][col] == self.GOAL:
                    print "G",
                if self.map[row][col] == self.WALL:
                    print "X",
                if self.map[row][col] == self.START:
                    print "S",
                if self.map[row][col] == self.FREE:
                    print " ",
                """
        print

if __name__=="__main__":
    EnvironmentLoader.loadEnvironment(threeroom_environment())

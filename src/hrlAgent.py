# 
# Copyright (C) 2008, Brian Tanner
# 
#http://rl-glue-ext.googlecode.com/
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
#  $Revision: 1011 $
#  $Date: 2009-02-11 22:29:54 -0700 (Wed, 11 Feb 2009) $
#  $Author: brian@tannerpages.com $
#  $HeadURL: http://rl-library.googlecode.com/svn/trunk/projects/packages/examples/mines-q-python/sample_q_agent.py $

import random
import sys
import copy
import pickle
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3
from random import Random, shuffle
from operator import itemgetter
from sys import argv

# This is a very simple q agent for discrete-action, discrete-state
# environments.  It uses epsilon-greedy exploration.
# 
# We've made a decision to store the previous action and observation in 
# their raw form, as structures.  This code could be simplified and you
# could store them just as ints.


# TO USE THIS Agent [order doesn't matter]
# NOTE: I'm assuming the Python codec is installed an is in your Python path
#   -  Start the rl_glue executable socket server on your computer
#   -  Run the SampleMinesEnvironment and SampleExperiment from this or a
#   different codec (Matlab, Python, Java, C, Lisp should all be fine)
#   -  Start this agent like:
#   $> python sample_q_agent.py

dynamicEpsilon = '1'

class q_agent(Agent):

    q_stepsize = 0.1
    q_epsilon = 0.1
    q_gamma = 0.9

    randGenerator=Random()
    lastAction=Action()
    lastObservation=Observation()

    numStates = 0
    numActions = 0
    value_function = None
    
    policyFrozen=False
    exploringFrozen=False
    
    episode = 0

    def agent_init(self,taskSpecString):
        TaskSpec = TaskSpecVRLGLUE3.TaskSpecParser(taskSpecString)
        if TaskSpec.valid:
            assert len(TaskSpec.getIntObservations())==1, "expecting 1-dimensional discrete observations"
            assert len(TaskSpec.getDoubleObservations())==0, "expecting no continuous observations"
            assert not TaskSpec.isSpecial(TaskSpec.getIntObservations()[0][0]), " expecting min observation to be a number not a special value"
            assert not TaskSpec.isSpecial(TaskSpec.getIntObservations()[0][1]), " expecting max observation to be a number not a special value"
            self.numStates=TaskSpec.getIntObservations()[0][1]+1;

            assert len(TaskSpec.getIntActions())==1, "expecting 1-dimensional discrete actions"
            assert len(TaskSpec.getDoubleActions())==0, "expecting no continuous actions"
            assert not TaskSpec.isSpecial(TaskSpec.getIntActions()[0][0]), " expecting min action to be a number not a special value"
            assert not TaskSpec.isSpecial(TaskSpec.getIntActions()[0][1]), " expecting max action to be a number not a special value"
            self.numActions=TaskSpec.getIntActions()[0][1]+1;
            
            self.value_function=[self.numActions*[0.0] for i in range(self.numStates+1)]

            self.episode = 0

        else:
            print "Task Spec could not be parsed: "+taskSpecString;
            
        validstatefile = open('valid_states.dat','r')
        unpickler = pickle.Unpickler(validstatefile)
        self.valid_states = unpickler.load()        

        self.lastAction=Action()
        self.lastObservation=Observation()
        
    def egreedy(self, state):
        maxIndex=0
        a=1
        if not self.exploringFrozen and self.randGenerator.random()<self.q_epsilon:
            return self.randGenerator.randint(0,self.numActions-1)

#       return self.value_function[state].index(max(self.value_function[state]))
        temp = [(i,v) for i,v in enumerate(self.value_function[state])]
        shuffle(temp)
        a = max(temp,key=itemgetter(1))[0]
        return a

    def agent_start(self,observation):
        theState=self.valid_states.index(observation.intArray[0])

        if dynamicEpsilon=='1':
            self.q_epsilon = 0.5-0.0008*self.episode
        else:
            self.q_epsilon = 0.1

        thisIntAction=self.egreedy(theState)
        returnAction=Action()
        returnAction.intArray=[thisIntAction]
        
        self.lastAction=copy.deepcopy(returnAction)
        self.lastObservation=copy.deepcopy(observation)

        self.episode += 1

        return returnAction
    
    def agent_step(self,reward, observation):
        newState=self.valid_states.index(observation.intArray[0])
        lastState=self.valid_states.index(self.lastObservation.intArray[0])
        lastAction=self.lastAction.intArray[0]

        newIntAction=self.egreedy(newState)

        # update q-value
        Q_sa=self.value_function[lastState][lastAction]
        max_Q_sprime_a = max(self.value_function[newState])     
        new_Q_sa=Q_sa + self.q_stepsize  * (reward + self.q_gamma * max_Q_sprime_a - Q_sa)

        if not self.policyFrozen:
            self.value_function[lastState][lastAction]=new_Q_sa

        returnAction=Action()
        returnAction.intArray=[newIntAction]
        
        self.lastAction=copy.deepcopy(returnAction)
        self.lastObservation=copy.deepcopy(observation)

        return returnAction
    
    def agent_end(self,reward):
        lastState=self.valid_states.index(self.lastObservation.intArray[0])
        lastAction=self.lastAction.intArray[0]

        Q_sa=self.value_function[lastState][lastAction]

        new_Q_sa=Q_sa + self.q_stepsize * (reward - Q_sa)

        if not self.policyFrozen:
            self.value_function[lastState][lastAction]=new_Q_sa

    
    def agent_cleanup(self):
        pass

    def save_value_function(self, fileName):
        theFile = open(fileName, "w")
        pickle.dump(self.value_function, theFile)
        theFile.close()

    def load_value_function(self, fileName):
        theFile = open(fileName, "r")
        self.value_function=pickle.load(theFile)
        theFile.close()
    
    def agent_message(self,inMessage):
        
        #   Message Description
        # 'freeze learning'
        # Action: Set flag to stop updating policy
        #
        if inMessage.startswith("freeze learning"):
            self.policyFrozen=True
            return "message understood, policy frozen"

        #   Message Description
        # unfreeze learning
        # Action: Set flag to resume updating policy
        #
        if inMessage.startswith("unfreeze learning"):
            self.policyFrozen=False
            return "message understood, policy unfrozen"

        #Message Description
        # freeze exploring
        # Action: Set flag to stop exploring (greedy actions only)
        #
        if inMessage.startswith("freeze exploring"):
            self.exploringFrozen=True
            return "message understood, exploring frozen"

        #Message Description
        # unfreeze exploring
        # Action: Set flag to resume exploring (e-greedy actions)
        #
        if inMessage.startswith("unfreeze exploring"):
            self.exploringFrozen=False
            return "message understood, exploring frozen"

        #Message Description
        # save_policy FILENAME
        # Action: Save current value function in binary format to 
        # file called FILENAME
        #
        if inMessage.startswith("save_policy"):
            splitString=inMessage.split(" ");
            self.save_value_function(splitString[1]);
            print "Saved.";
            return "message understood, saving policy"

        #Message Description
        # load_policy FILENAME
        # Action: Load value function in binary format from 
        # file called FILENAME
        #
        if inMessage.startswith("load_policy"):
            splitString=inMessage.split(" ")
            self.load_value_function(splitString[1])
            print "Loaded."
            return "message understood, loading policy"

        return "SampleqAgent(Python) does not understand your message."



if __name__=="__main__":
    AgentLoader.loadAgent(q_agent())

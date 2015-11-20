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
import math
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
import scipy.stats
import numpy as np

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

    q_stepsize = 0.2
    q_gamma = 0.8

    q_epsilon = 0.1 # set later
    randGenerator=Random()
    lastAction=Action()
    lastObservation=Observation()

    #TODO: Parameters to set
    SIZE_WORLD = 10
    N_PC = 100 # No. of place cells
    N_AC = 10 # No. of action cells
    sigma_AC = 2

    policyFrozen=False
    exploringFrozen=False
    
    episode = 0

    def agent_init(self,taskSpecString):
        TaskSpec = TaskSpecVRLGLUE3.TaskSpecParser(taskSpecString)
        if TaskSpec.valid:
            self.W = np.asarray([[random.random()*0.5 for j in range(self.N_AC)] for i in range(self.N_PC)])
            self.P = np.asarray([[0.0 for j in range(self.N_AC)] for i in range(self.N_PC)])

            # Place cell Gaussians
            self.mean_PC = []
            start_i = float(self.SIZE_WORLD)/(2*math.sqrt(self.N_PC))
            for i in xrange(int(math.sqrt(self.N_PC))):
                start_j = float(self.SIZE_WORLD)/(2*math.sqrt(self.N_PC))
                for j in xrange(int(math.sqrt(self.N_PC))):
                    self.mean_PC.append((start_i,start_j))
                    start_j += float(self.SIZE_WORLD)/math.sqrt(self.N_PC)
                start_i += float(self.SIZE_WORLD)/math.sqrt(self.N_PC)

            self.sigma_PC = (float(self.SIZE_WORLD)/math.sqrt(self.N_PC) * math.sqrt(2))/6

            self.episode = 0

        else:
            print "Task Spec could not be parsed: "+taskSpecString; 

        self.lastQ = 0.0
        
    def egreedy(self, state, r_PC):
        index=0

        r_1_AC = []
        for i in xrange(self.N_AC):
            r_1_AC.append(np.dot(np.asarray(r_PC),self.W[:,i]))

        if not self.exploringFrozen and self.randGenerator.random()<self.q_epsilon:
            return (random.random()*2*math.pi, r_1_AC)

        num = 0.0
        den = 0.0
        for i in xrange(self.N_AC):
            num += r_1_AC[i] * math.sin(2*math.pi*i/self.N_AC)
            den += r_1_AC[i] * math.cos(2*math.pi*i/self.N_AC)

        phi_AC = math.atan2(num,den)           
        if phi_AC < 0:
            phi_AC += 2*math.pi

        return (phi_AC, r_1_AC)

    # Get r_PC values
    def getProbGaussians(self, x, y):
        prob = []
        for i in xrange(self.N_PC):
            prob.append(scipy.stats.norm(self.mean_PC[i][0],self.sigma_PC).pdf(x) * scipy.stats.norm(self.mean_PC[i][1],self.sigma_PC).pdf(y))
        return prob

    def agent_start(self,observation):
        self.P = np.asarray([[0.0 for j in range(self.N_AC)] for i in range(self.N_PC)])

        theState=observation.doubleArray

        if dynamicEpsilon=='1':
            self.q_epsilon = 0.3-0.005*self.episode
        else:
            self.q_epsilon = 0.3

        r_PC = self.getProbGaussians(theState[0], theState[1]) 
        res = self.egreedy(theState, r_PC)
        phi_AC = res[0]
        r_1_AC = res[1]
        r_2_AC = []
        for i in xrange(self.N_AC):
            r_2_AC.append(math.exp( (-1*(phi_AC - 2*math.pi*i/self.N_AC)**2)/(2*self.sigma_AC**2) ))

        # Update P_ij
        for i in xrange(self.N_AC):
            for j in xrange(self.N_PC):
                self.P[j,i] = self.q_stepsize*self.P[j,i] + r_2_AC[i]*r_PC[j]

        returnAction=Action()
        returnAction.doubleArray=[phi_AC]
        
        # finding closest AC
        closest_AC = r_2_AC.index(max(r_2_AC))
        self.lastQ = r_1_AC[closest_AC]

        self.episode += 1

        return returnAction
    
    def agent_step(self,reward, observation):
        theState=observation.doubleArray

        r_PC = self.getProbGaussians(theState[0], theState[1])    
        res = self.egreedy(theState, r_PC)
        phi_AC = res[0]
        r_1_AC = res[1]
        r_2_AC = []
        for i in xrange(self.N_AC):
            r_2_AC.append(math.exp( (-1*(phi_AC - 2*math.pi*i/self.N_AC)**2)/(2*self.sigma_AC**2) ))

        # Calculate reward prediction error
        delta = reward + self.q_gamma*max(r_1_AC) - self.lastQ
        #print self.q_gamma*max(r_1_AC), self.lastQ

        # Update synaptic weights
        for i in xrange(self.N_PC):
            for j in xrange(self.N_AC):       
                self.W[i,j] = self.q_stepsize * delta * self.P[i,j]

        # Update P_ij
        for i in xrange(self.N_AC):
            for j in xrange(self.N_PC):
                self.P[j,i] = self.q_stepsize*self.P[j,i] + r_2_AC[i]*r_PC[j]

        returnAction=Action()
        returnAction.doubleArray=[phi_AC]
        
        # finding closest AC
        closest_AC = r_2_AC.index(max(r_2_AC))
        self.lastQ = r_1_AC[closest_AC]

        self.episode += 1

        return returnAction
    
    def agent_end(self,reward):

        # Calculate reward prediction error
        delta = reward - self.lastQ

        # Update synaptic weights
        for i in xrange(self.N_PC):
            for j in xrange(self.N_AC):       
                self.W[i,j] = self.q_stepsize * delta * self.P[i,j]
    
    def agent_cleanup(self):
        pass
    
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

#!/bin/bash
compBasePath=../external/15-rl-competition-2009
systemPath=$compBasePath/system

#Source a script that sets all important functions and variables
#source $systemPath/rl-competition-includes.sh

#Utility functions from rl-competition-includes.sh
startEnvShellInBackGround

./3room_environment

#Utility functions from rl-competition-includes.sh
waitForEnvShellToDie

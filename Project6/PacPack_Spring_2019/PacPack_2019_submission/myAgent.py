# myAgentP3.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
# This file was based on the starter code for student bots, and refined 
# by Mesut (Xiaocheng) Yang


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint
import heapq

#########
# Agent #
#########
class MyAgent(CaptureAgent):
  """
  Reflex agent that evaluates its position relative to the food and the enemies and makes a decision. Employs randomness
    when it has not gotten food in a while.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    # Make sure you do not delete the following line.
    # If you would like to use Manhattan distances instead
    # of maze distances in order to save on initialization
    # time, please take a look at:
    # CaptureAgent.registerInitialState in captureAgents.py.
    CaptureAgent.registerInitialState(self, gameState)
    self.start = gameState.getAgentPosition(self.index)
    self.randomElement = 1

  def chooseAction(self, gameState):
    """
    Picks among actions based on its evaluation, will sometimes choose a random value instead to keep from thrashing
    """
    teammateActions = self.receivedBroadcast

    actions = gameState.getLegalActions(self.index)

    filteredActions = actionsWithoutReverse(actionsWithoutStop(actions), gameState, self.index)


    denom = 50
    if random.random() < self.randomElement / denom:
        self.randomElement -= .1 * denom
        return random.choice(filteredActions)

    bestScore = -10000
    bestAction = None

    for a in filteredActions:
      s = self.depthEval(gameState, 5)
      if s > bestScore:
        bestAction = a
        bestScore = s

    if len(gameState.generateSuccessor(self.index, a).getFood().asList()) != len(gameState.getFood().asList()):
        self.randomElement = 0
    else:
        self.randomElement += 1

    return bestAction


  def depthEval(self, gameState, depth):

    if depth <= 0:
        return self.evaluatePosition(gameState) - 1

    actions = gameState.getLegalActions(self.index)
    filteredActions = actionsWithoutReverse(actionsWithoutStop(actions), gameState, self.index)

    positionScore = []
    for a in filteredActions:
        positionScore.append(self.depthEval(gameState.generateSuccessor(self.index, a), depth - 1))

    return max(positionScore) + self.evaluatePosition(gameState) - 1




  def evaluatePosition(self, gameState):
    foodList = gameState.getFood().asList()
    ghostList = [gameState.getAgentPosition(ghost) for ghost in gameState.getGhostTeamIndices()]
    teamList = [gameState.getAgentPosition(teammate) for teammate in gameState.getPacmanTeamIndices() if teammate != self.index]

    fDist = 100000
    fBest = None
    averageFoodDist = 0
    for f in foodList:
        d = self.getMazeDistance(gameState.getAgentPosition(self.index), f)
        averageFoodDist += d
        if d < fDist or (not fBest):
            fBest = f
            fDist = d

    foodScore = 1
    if fDist != 0:
        foodScore = 1 / fDist

    gDist = 1000000
    gBest = None
    for g in ghostList:
        d = self.getMazeDistance(gameState.getAgentPosition(self.index), g)
        if d < fDist or (not gBest):
            gBest = g
            gDist = d

    ghostScore = 0
    if gDist < 10 and gDist != 0:
        ghostScore = 1 / gDist
        if gDist < 3:
            ghostScore = 500
    else:
        gScore = -1

    tDist = 100000
    tBest = None
    for t in teamList:
        d = self.getMazeDistance(gameState.getAgentPosition(self.index), t)
        if d < tDist or (not tBest):
            tBest = t
            tDist = d

    teamScore = 1
    if tDist < 5:
        teamScore = tDist

    return self.getScore(gameState) + (1 / averageFoodDist) +  foodScore - ghostScore

def actionsWithoutStop(legalActions):
  """
  Filters actions by removing the STOP action
  """
  legalActions = list(legalActions)
  if Directions.STOP in legalActions:
    legalActions.remove(Directions.STOP)
  return legalActions

def actionsWithoutReverse(legalActions, gameState, agentIndex):
  """
  Filters actions by removing REVERSE, i.e. the opposite action to the previous one
  """
  legalActions = list(legalActions)
  reverse = Directions.REVERSE[gameState.getAgentState(agentIndex).configuration.direction]
  if len (legalActions) > 1 and reverse in legalActions:
    legalActions.remove(reverse)
  return legalActions


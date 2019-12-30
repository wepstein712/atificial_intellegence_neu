# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util
from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        foodDist = 0
        distanceToEnmies = 0
        scoreMod = 0
        for i, enemy in enumerate(newGhostStates):
            if newScaredTimes[i] == 0:
                enemyPos = enemy.getPosition()
                eDist = util.manhattanDistance(enemyPos, newPos)
                if eDist < 6:
                    if eDist <= 1:
                        scoreMod -= 500
                    else:
                        distanceToEnmies -= 1/util.manhattanDistance(enemyPos, newPos)

        for food in newFood.asList():
           foodDist += (1 /util.manhattanDistance(food, newPos))

        score = successorGameState.getScore() + foodDist + distanceToEnmies + scoreMod
        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        retVal = self.maxValue(gameState, 0, 0)
        return retVal[1]

    def evaluate(self, state, depth, invokerIndex):
        if state.isWin() or state.isLose():
            t = self.evaluationFunction(state)
            return t
        if invokerIndex == state.getNumAgents() - 1:
            if depth + 1 == self.depth:
                score = self.evaluationFunction(state)
                return score
            score = self.maxValue(state, depth + 1, 0)[0]
            return score
        else:
            score =  self.minValue(state, depth, invokerIndex + 1)
            return score

    def maxValue(self, state, depth, actorIndex):
        val = float("-inf")
        bestMove = None
        for s in state.getLegalActions(actorIndex):

            temp = self.evaluate(state.generateSuccessor(actorIndex, s), depth, actorIndex)
            val = max(val, temp)
            if val == temp:
                bestMove = s
        return (val, bestMove)

    def minValue(self, state, depth, actorIndex):
        val = float("inf")
        for s in state.getLegalActions(actorIndex):
            val = min(val, self.evaluate(state.generateSuccessor(actorIndex, s), depth, actorIndex))
        return val

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        retVal = self.maxValue(gameState, 0, 0, float("-inf"), float("inf"))
        return retVal[1]

    def evaluate(self, state, depth, invokerIndex, alpha, beta):
        if state.isWin() or state.isLose():
            t = self.evaluationFunction(state)
            return t
        if invokerIndex == state.getNumAgents() - 1:
            if depth + 1 == self.depth:
                score = self.evaluationFunction(state)
                return score
            score = self.maxValue(state, depth + 1, 0, alpha, beta)[0]
            return score
        else:
            score = self.minValue(state, depth, invokerIndex + 1, alpha, beta)
            return score

    def maxValue(self, state, depth, actorIndex, alpha, beta):
        val = float("-inf")
        bestMove = None
        for s in state.getLegalActions(actorIndex):
            temp = self.evaluate(state.generateSuccessor(actorIndex, s), depth, actorIndex, alpha, beta)
            val = max(val, temp)
            if val == temp:
                bestMove = s
            if val > beta:
                return (val, bestMove)
            alpha = max(alpha, val)
        return (val, bestMove)

    def minValue(self, state, depth, actorIndex, alpha, beta):
        val = float("inf")
        for s in state.getLegalActions(actorIndex):
            val = min(val, self.evaluate(state.generateSuccessor(actorIndex, s), depth, actorIndex, alpha, beta))
            if val < alpha:
                return val
            beta = min(beta, val)
        return val

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        retVal = self.maxValue(gameState, 0, 0)
        return retVal[1]

    def evaluate(self, state, depth, invokerIndex):
        if state.isWin() or state.isLose():
            t = self.evaluationFunction(state)
            return t
        if invokerIndex == state.getNumAgents() - 1:
            if depth + 1 == self.depth:
                score = self.evaluationFunction(state)
                return score
            score = self.maxValue(state, depth + 1, 0)[0]
            return score
        else:
            score = self.minValue(state, depth, invokerIndex + 1)
            return score

    def maxValue(self, state, depth, actorIndex):
        val = float("-inf")
        bestMove = None
        for s in state.getLegalActions(actorIndex):
            temp = self.evaluate(state.generateSuccessor(actorIndex, s), depth, actorIndex)
            val = max(val, temp)
            if val == temp:
                bestMove = s
        return (val, bestMove)

    def minValue(self, state, depth, actorIndex):
        val = 0
        p = len(state.getLegalActions(actorIndex))
        for s in state.getLegalActions(actorIndex):
            val += 1/p * self.evaluate(state.generateSuccessor(actorIndex, s), depth, actorIndex)
        return val


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    fearDuration = [ghostState.scaredTimer for ghostState in ghostStates]


    foodDist = 0
    distEnemy = 0
    scoreMod = 0
    for i, enemy in enumerate(ghostStates):
        if fearDuration[i] == 0:
            ePos = enemy.getPosition()
            eDist = util.manhattanDistance(ePos, pos)
            if eDist < 6:
                if eDist < 1:
                    scoreMod -= 500
                else:
                    distEnemy -= 1 / util.manhattanDistance(ePos, pos)

    for f in food:
        foodDist += 1 / util.manhattanDistance(f, pos)

    score = currentGameState.getScore() + 3 * foodDist + distEnemy + scoreMod
    return score

# Abbreviation
better = betterEvaluationFunction

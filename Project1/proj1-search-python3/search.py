# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import random

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST

    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    #print("Start:", problem.getStartState())
    #print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    #print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    n = Directions.NORTH
    e = Directions.EAST
    f = util.Stack()
    vis = []
    f.push([problem.getStartState()])

    x =  recurvDepth(problem, [problem.getStartState()], vis)

    temp = []
    while x.isEmpty() is False:
        step = x.pop()
        if step is None or len(step) < 2:
            continue
        temp.append(step)
    retVal = [temp[0][1]]
    for x in range(1, len(temp)):
        if temp[x] != temp[x - 1]:
            retVal.append(temp[x][1])
    return retVal


def recurvDepth(problem, node, visitted):
    """ the fringe is a stack of steps along the maze, each iteration you pop off the top of the stack, run down those steps and then expand the children, and add those to the stack. if at any point you get to the end, return that list"""

    visitted.append(node[0])
    if problem.isGoalState(node[0]):
        salvation = util.Stack()
        salvation.push(node)
        return salvation

    for edge in problem.getSuccessors(node[0]):
        if edge[0] not in visitted:
            x = recurvDepth(problem, edge, visitted)
            if x is not None:
                x.push(edge)
                return x

    return None



def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    n = Directions.NORTH
    e = Directions.EAST

    f = util.Queue()

    for e in problem.getSuccessors(problem.getStartState()):
        f.push([e])

    vis = []
    vis.append(problem.getStartState())
    altV = []
    p = util.Queue()
    family = {}
    family[problem.getStartState()] = None
    goal = None

    while f.isEmpty() is False:
        path = f.pop()
        node = path[len(path) - 1]
        if node[0] in vis:
            continue
        vis.append(node[0])
        if problem.isGoalState(node[0]):
            goal = path
            break
        for edge in problem.getSuccessors(node[0]):
            if edge[0] not in vis:
                temp = path.copy()
                temp.append(edge)
                f.push(temp)


    retVal = []
    for x in goal:
        retVal.append(x[1])

    return retVal


def uniformCostSearch(problem):
    """Search the node of least total cost first."""

    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    n = Directions.NORTH
    e = Directions.EAST

    f = util.PriorityQueue()


    vis = {}
    vis[problem.getStartState()] = 0
    family = {}
    family[problem.getStartState()] = 0
    goal = None
    for prelim in problem.getSuccessors(problem.getStartState()):
        f.push(prelim, prelim[2])
        family[prelim[0]] = [problem.getStartState()]


    while f.isEmpty() is False:
        node = f.pop()
        if node[0] in vis.keys():
            continue

        if problem.isGoalState(node[0]):
            goal = node
            break

        parent = family[node[0]]
        vis[node[0]] = vis[parent[0]] + node[2]
        cost = vis[node[0]]
        for edge in problem.getSuccessors(node[0]):
            if edge[0] not in vis.keys():
                if edge[0] not in family.keys():
                    family[edge[0]] = node
                else:
                    exParent = family[edge[0]]
                    exParCost = vis[exParent[0]]
                    otherEdge = 10000
                    for alts in problem.getSuccessors(exParent[0]):
                        if alts[0] == edge[0]:
                            otherEdge = alts
                    if cost + edge[2] <= exParCost + otherEdge[2]:
                        family[edge[0]] = node
                f.update(edge, cost + edge[2])


    x = uniformBacktrace(family, goal, [problem.getStartState()])
    temp = []
    while x.isEmpty() is False:
        step = x.pop()

        if step is None or len(step) < 2:
            continue
        temp.append(step)
    retVal = [temp[0][1]]
    for x in range(1, len(temp)):
        if temp[x] != temp[x - 1]:
            retVal.append(temp[x][1])

    return retVal[::-1]


def uniformBacktrace(bloodlines, end, start):
    curNode = end
    parent = []
    path = util.Queue()
    path.push(end)
    while parent != start:

        parent = bloodlines[curNode[0]]
        path.push(parent)
        curNode = parent
    return path

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    n = Directions.NORTH
    e = Directions.EAST

    f = util.PriorityQueue()


    vis = {}
    vis[problem.getStartState()] = 0
    family = {}
    family[problem.getStartState()] = 0
    goal = None
    for prelim in problem.getSuccessors(problem.getStartState()):
        f.push(prelim, prelim[2] + heuristic(prelim[0], problem))
        family[prelim[0]] = [problem.getStartState()]


    while f.isEmpty() is False:
        node = f.pop()
        if node[0] in vis.keys():
            continue
        parent = family[node[0]]
        vis[node[0]] = vis[parent[0]] + node[2]
        cost = vis[node[0]]
        if problem.isGoalState(node[0]):
            goal = node
            break

        for edge in problem.getSuccessors(node[0]):
            if edge[0] not in vis.keys():
                if edge[0] not in family.keys():
                    family[edge[0]] = node
                else:
                    exParent = family[edge[0]]
                    exParCost = vis[exParent[0]]
                    otherEdge = 10000
                    for alts in problem.getSuccessors(exParent[0]):
                        if alts[0] == edge[0]:
                            otherEdge = alts
                    if cost + edge[2] + heuristic(edge[0], problem) < exParCost + otherEdge[2]:
                        family[edge[0]] = node
                f.update(edge, cost + edge[2] + heuristic(edge[0], problem))



    x = astarBacktrace(family, goal, [problem.getStartState()])
    temp = []
    while x.isEmpty() is False:
        step = x.pop()

        if step is None or len(step) < 2:
            continue
        temp.append(step)
    retVal = [temp[0][1]]
    for x in range(1, len(temp)):
        if temp[x] != temp[x - 1]:
            retVal.append(temp[x][1])
    return retVal[::-1]

def astarBacktrace(bloodlines, end, start):
    curNode = end
    parent = []
    path = util.Queue()
    path.push(end)
    while parent != start:
        parent = bloodlines[curNode[0]]
        path.push(parent)
        curNode = parent
    return path


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

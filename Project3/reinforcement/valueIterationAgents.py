# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        # print("RUNNING ON --- ", iterations)
        # print("GAMMA IS: ", self.discount)
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here

        """
            do iteration # of times:
                for each state:
                    update this states value to be:
                        take the max across all x calculated:
                        for each action you can take from this step:
                            for each outcome of that action:
                               y = p(outcome) * [ reward(outcome) + discount * currentValue of this state]
                            x = sum all Y

        """

        for x in range(self.iterations):
            tempValues = util.Counter()
            # print("-----Starting: ", x)
            # t = input()
            for state in self.mdp.getStates():
                a = []
                for action in self.mdp.getPossibleActions(state):
                    total = self.computeQValueFromValues(state, action)
                    a.append(total)
                if a:
                    tempValues[state] = max(a)
            for key in tempValues.keys():
                self.values[key] = tempValues[key]



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        r = 0
        for outcome in self.mdp.getTransitionStatesAndProbs(state, action):
            # print("T: ", outcome[1])
            # print("R: ", self.mdp.getReward(state, action, outcome[0]))
            # print("Gamma: ", self.discount)
            # print("V(S'): ", self.getValue(outcome[0]))
            # print("R: ", outcome[1] * (self.mdp.getReward(state, action, outcome[0]) + self.discount * self.getValue(outcome[0])))
            r += outcome[1] * (self.mdp.getReward(state, action, outcome[0]) + (self.discount * self.getValue(outcome[0])))
        # print("RETUNING: ", r)
        return r


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        bestAction = None
        bestQ = -10000
        for action in self.mdp.getPossibleActions(state):
            curQ = self.computeQValueFromValues(state, action)
            if curQ > bestQ or bestAction == None:
                bestAction = action
                bestQ = curQ
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        for x in range(self.iterations):
            tempValues = util.Counter()
            states =  self.mdp.getStates()
            state = states[x % len(states)]
            a = []
            for action in self.mdp.getPossibleActions(state):
                total = self.computeQValueFromValues(state, action)
                a.append(total)
            if a:
                tempValues[state] = max(a)
            for key in tempValues.keys():
                self.values[key] = tempValues[key]

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        """
        Compute predecessors of all states.
        Initialize an empty priority queue.
        For each non-terminal state s, do: (note: to make the autograder work for this question, you must iterate over
          states in the order returned by self.mdp.getStates())
            - Find the absolute value of the difference between the current value of s in self.values and the highest
                Q-value across all possible actions from s (this represents what the value should be); call this number
                 diff. Do NOT update self.values[s] in this step.
            - Push s into the priority queue with priority -diff (note that this is negative). We use a negative because
                the priority queue is a min heap, but we want to prioritize updating states that have a higher error.
        For iteration in 0, 1, 2, ..., self.iterations - 1, do:
            - If the priority queue is empty, then terminate.
            - Pop a state s off the priority queue.
            - Update s's value (if it is not a terminal state) in self.values.
            - For each predecessor p of s, do:
            -Find the absolute value of the difference between the current value of p in self.values and the highest
                Q-value across all possible actions from p (this represents what the value should be); call this number
                 diff. Do NOT update self.values[p] in this step.
              -If diff > theta, push p into the priority queue with priority -diff (note that this is negative), as
                long as it does not already exist in the priority queue with equal or lower priority. As before, we use
                 a negative because the priority queue is a min heap, but we want to prioritize updating states that
                 have a higher error.
        :return:
        """
        predecessors = self.computePredecessors()
        pQ = util.PriorityQueue()
        for s in self.mdp.getStates():
            if self.mdp.isTerminal(s):
                continue
            curVal = self.values[s]
            bestAction = self.computeActionFromValues(s)
            qVal = self.computeQValueFromValues(s, bestAction)
            delta = abs(qVal - curVal)
            pQ.push(s, -delta)

        for i in range(self.iterations):
            if pQ.isEmpty():
                break;
            s = pQ.pop()
            if not self.mdp.isTerminal(s):
                a = []
                for action in self.mdp.getPossibleActions(s):
                    total = self.computeQValueFromValues(s, action)
                    a.append(total)
                self.values[s] = max(a)
            for p in predecessors[s]:
                curVal = self.values[p]
                bestAction = self.computeActionFromValues(p)
                qVal = self.computeQValueFromValues(p, bestAction)
                delta = abs(qVal - curVal)
                if delta > self.theta:
                    pQ.update(p, -delta)


    def computePredecessors(self):

        """
        predessecor
            for every state:
                for every action:
                - dictionary of (state, action) => set of states
                - running list of which state,actions we have taken to that all can be updated at each new result
        """
        preds = {}
        states = self.mdp.getStates()
        print(states)
        for state in states:
            # print("STATE: ", state)
            actions = self.mdp.getPossibleActions(state)
            # print("ACTIONS: ", actions)
            for action in actions:
                outcomes = self.mdp.getTransitionStatesAndProbs(state, action)
                # print("OUTCOMES: ", outcomes)
                for out in outcomes:
                    if out[1] > 0:
                        if len(preds.keys()) > 0:
                            if out[0] in preds.keys():
                                preds[out[0]].add(state)
                            else:
                                preds[out[0]] = {state}
                        else:
                            preds[out[0]] = {state}

        for s in states:
            # print("should be calling")
            self.recurcivePredFind(preds, s)

        for k in preds.keys():
            print(k, ": ", preds[k])

        return preds
    def recurcivePredFind(self, ancestry, state):
        # print(state, " --- ", ancestry[state])
        for parent in ancestry[state]:
            # print("parent: ", parent)
            grandparents = ancestry[parent]
            # print("GP: ", grandparents)
            filtered = self.filterList(grandparents, ancestry[state])
            # print("Filt: ", filtered)
            if filtered:
                ancestry[state].update(filtered)
                for f in filtered:
                    filtered.extend(self.recurcivePredFind(ancestry, f))
                return filtered
            else:
                return []

    def filterList(self, items, list):
        unvissited = []
        for i in items:
            if i not in list:
                unvissited.append(i)
        return unvissited


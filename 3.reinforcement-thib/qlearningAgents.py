# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        "*** YOUR CODE HERE ***"
        self.qValues = util.Counter()


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.qValues[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return 0.0
        
        maxVal = float('-inf')
        for action in legalActions:
            val = self.getQValue(state, action)
            maxVal = max(val, maxVal)
        return maxVal
                
    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None
        
        bestAction = None
        maxVal = float('-inf')
        for action in legalActions:
            val = self.getQValue(state, action)
            if val > maxVal:
                maxVal = val
                bestAction = action
        return bestAction

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if not legalActions:
            return None
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(state)

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        self.qValues[(state, action)] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * (reward + self.discount * self.computeValueFromQValues(nextState))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.9,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        featuresVector = self.featExtractor.getFeatures(state, action)
        val = 0
        for feature, value in featuresVector.items():
            val += self.weights[feature] * value
        return val

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        featuresVector = self.featExtractor.getFeatures(state, action)
        diff = (reward + self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state, action)
        
        for feature in featuresVector:
            self.weights[feature] += self.alpha * diff * featuresVector[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass

class SemiGradientTDAgent(ApproximateQAgent):
    def __init__(self, lambda_=0.9, epsilonDecay=0.99, **args):
        super().__init__(**args)
        self.lambda_ = lambda_
        self.eligibilityTraces = util.Counter()
        self.epsilonDecay = epsilonDecay
    
    def update(self, state, action, nextState, reward):
        nextAction = self.getPolicy(nextState)
        if nextAction:
            nextQValue = self.getQValue(nextState, nextAction)
        else:
            nextQValue = 0
        currentQValue = self.getQValue(state, action)
        TDE = reward + (self.discount * nextQValue) - currentQValue

        features = self.featExtractor.getFeatures(state, action)
        for feature in features:
            self.eligibilityTraces[feature] = self.discount * self.lambda_ * self.eligibilityTraces[feature] + features[feature]

        for feature, value in features.items(): 
            self.weights[feature] += self.alpha * TDE * self.eligibilityTraces[feature]
            
    def final(self, state):
        self.eligibilityTraces = util.Counter()
        super().final(state)
        if self.numTraining > 0:
            self.epsilon *= self.epsilonDecay




class TrueOnlineTDAgent(ApproximateQAgent):
    def __init__(self, lambda_=0.9, epsilonDecay=0.99, **args):
        super().__init__(**args)
        self.lambda_ = lambda_
        self.z = util.Counter()
        self.epsilonDecay = epsilonDecay
        self.oldV = 0

    def update(self, state, action, nextState, reward):
        nextAction = self.getPolicy(nextState)
        nextQValue = self.getQValue(nextState, nextAction) if nextAction else 0
        currentQValue = self.getQValue(state, action)
        
        delta = reward + self.discount * nextQValue - currentQValue

        features = self.featExtractor.getFeatures(state, action)

        for feature in features:
            self.z[feature] *= self.discount * self.lambda_
            self.z[feature] += (1 - self.alpha * self.discount * self.lambda_ * self.z[feature]) * features[feature]

            self.weights[feature] += self.alpha * (delta + currentQValue - self.oldV) * self.z[feature]
            self.weights[feature] -= self.alpha * (currentQValue - self.oldV) * features[feature]

        self.oldV = nextQValue

    def final(self, state):
        self.z = util.Counter()
        super().final(state)
        if self.numTraining > 0:
            self.epsilon *= self.epsilonDecay




class TrueOnlineTDAgent2(ApproximateQAgent):
    def __init__(self, epsilon=0.05, alpha=0.5, gamma=1, lambda_=0.9, **args):
        ApproximateQAgent.__init__(self, **args)  # inherits init from ApproximateQAgent
        self.lambda_ = lambda_
        self.epsilon = epsilon  # exploration rate
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.eligibilityTraces = util.Counter()  # Eligibility traces for each feature
        self.oldV = 0  # Store the previous value estimate
        self.lastFeatures = None  # Store the previous state features

    def update(self, state, action, nextState, reward):
        """
        Update weights using the True Online TD(λ) algorithm.
        """
        # Extract features for the current state and action
        features = self.featExtractor.getFeatures(state, action)

        # Get the policy's action for the next state
        nextAction = self.computeActionFromQValues(nextState)
        nextFeatures = self.featExtractor.getFeatures(nextState, nextAction) if nextAction is not None else None

        # Compute the current and next value estimates
        V = sum([self.weights[f] * v for f, v in features.items()])
        V_next = sum([self.weights[f] * v for f, v in nextFeatures.items()]) if nextFeatures is not None else 0

        # TD error
        delta = reward + self.gamma * V_next - V

        # Eligibility trace and weight update
        if self.lastFeatures:
            dot_product = sum([self.eligibilityTraces[f] * self.lastFeatures.get(f, 0) for f in features])
        else:
            dot_product = 0

        for f in features:
            self.eligibilityTraces[f] *= self.gamma * self.lambda_
            self.eligibilityTraces[f] += (1 - self.alpha * self.gamma * self.lambda_ * dot_product) * features[f]

        for f in features:
            self.weights[f] += self.alpha * (delta + V - self.oldV) * self.eligibilityTraces[f]
            if self.lastFeatures is not None:
                self.weights[f] -= self.alpha * (V - self.oldV) * self.lastFeatures.get(f, 0)

        # Update old value estimate
        self.oldV = V_next

        # Update last features
        self.lastFeatures = features.copy() if nextAction is not None else None

    def computeActionFromQValues(self, state):
        """
        Compute the next action to take using the ε-greedy policy.
        """
        # Your implementation for ε-greedy policy
        pass

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getAction(self, state):
        """
        Compute the action to take in the current state, including exploration.
        """
        # Your implementation for action selection, possibly with exploration
        pass

    def getQValue(self, state, action):
        """
        Compute the Q-value for a particular state and action pair.
        """
        features = self.featExtractor.getFeatures(state, action)
        return sum([self.weights[f] * v for f, v in features.items()])

    def final(self, state):
        """
        Perform final updates at the end of an episode.
        """
        ApproximateQAgent.final(self, state)  # Call the final method from the parent class
        # Reset eligibility traces and previous value estimate
        self.eligibilityTraces = util.Counter()
        self.oldV = 0
        self.lastFeatures = None


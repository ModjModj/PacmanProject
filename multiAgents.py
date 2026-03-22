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

#Elijah Simmonds & Micah Suk

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

        "*** YOUR CODE HERE ***"
        evalScore = 0
        if newPos in newFood.asList():
            evalScore += 10
        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            distance = manhattanDistance(newPos, ghostPos)
            if distance < 2 and newScaredTimes[newGhostStates.index(ghostState)] == 0:
                evalScore -= 20
            elif distance < 2 and newScaredTimes[newGhostStates.index(ghostState)] > 0:
                evalScore += 20
            else:
                evalScore += 5 / (distance + 1)
        return evalScore

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
        """
        "*** YOUR CODE HERE ***"
        agentIndex = self.index  # should be 0
        bestScore = -float('inf')
        bestAction = None

        # avoid pacman sitting still unless it's the only move
        actions = [a for a in gameState.getLegalActions(agentIndex)
                   if a != Directions.STOP]
        if not actions:
            actions = gameState.getLegalActions(agentIndex)

        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            # after Pacman's move we advance to the next agent (usually ghost 1)
            score = self.value(successor, agentIndex + 1, self.depth)

            if score > bestScore:
                bestScore = score
                bestAction = action

        # if there are no legal actions (shouldn't happen except in terminal
        # states) just return Stop to be safe
        return bestAction if bestAction is not None else Directions.STOP
    
    def value(self, gameState, agentIndex, depth):
        if agentIndex == gameState.getNumAgents():
            agentIndex = 0
            depth -= 1
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)
        else:
            return self.minValue(gameState, agentIndex, depth)
        
    def maxValue(self, gameState, agentIndex, depth):
        maxScore = -float('inf')
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            score = self.value(successor, agentIndex + 1, depth)
            if score > maxScore:
                maxScore = score
        return maxScore
    
    def minValue(self, gameState, agentIndex, depth):
        minScore = float('inf')
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            score = self.value(successor, agentIndex + 1, depth)
            if score < minScore:
                minScore = score
        return minScore
    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        agentIndex = self.index  # should be 0
        alpha = bestScore = -float('inf')
        beta = float('inf')
        bestAction = None

        # prefer moving over stopping unless trapped
        actions = [a for a in gameState.getLegalActions(agentIndex)
                   if a != Directions.STOP]
        if not actions:
            actions = gameState.getLegalActions(agentIndex)

        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            # after Pacman's move we advance to the next agent (usually ghost 1)
            score = self.value(successor, agentIndex + 1, self.depth, alpha, beta)

            if score > bestScore:
                bestScore = score
                bestAction = action
                alpha = max(alpha, bestScore)

        # if there are no legal actions (shouldn't happen except in terminal
        # states) just return Stop to be safe
        return bestAction if bestAction is not None else Directions.STOP
    
    def value(self, gameState, agentIndex, depth, alpha, beta):
        if agentIndex == gameState.getNumAgents():
            agentIndex = 0
            depth -= 1
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth, alpha, beta)
        else:
            return self.minValue(gameState, agentIndex, depth, alpha, beta)
        
    def maxValue(self, gameState, agentIndex, depth, alpha, beta ):
        actions = gameState.getLegalActions(agentIndex)
        if not actions:
            return self.evaluationFunction(gameState)

        maxScore = -float('inf')
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            score = self.value(successor, agentIndex + 1, depth, alpha, beta)
            if score > maxScore:
                maxScore = score
            if maxScore > beta:
                return maxScore
            alpha = max(alpha, maxScore)
        return maxScore
    
    def minValue(self, gameState, agentIndex, depth, alpha, beta):
        actions = gameState.getLegalActions(agentIndex)
        if not actions:
            return self.evaluationFunction(gameState)

        minScore = float('inf')
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            score = self.value(successor, agentIndex + 1, depth, alpha, beta)
            if score < minScore:
                minScore = score
            if minScore < alpha:
                return minScore
            beta = min(beta, minScore)
        return minScore

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
        "*** YOUR CODE HERE ***"
        agentIndex = self.index  # should be 0
        bestScore = -float('inf')
        bestAction = None
 
        # prefer moving over stopping unless trapped
        actions = [a for a in gameState.getLegalActions(agentIndex)
                   if a != Directions.STOP]
        if not actions:
            actions = gameState.getLegalActions(agentIndex)
 
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            score = self.value(successor, agentIndex + 1, self.depth)
            if score > bestScore:
                bestScore = score
                bestAction = action
 
        return bestAction if bestAction is not None else Directions.STOP
 
    def value(self, gameState, agentIndex, depth):
        if agentIndex == gameState.getNumAgents():
            agentIndex = 0
            depth -= 1
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, depth)
        else:
            return self.expValue(gameState, agentIndex, depth)
 
    def maxValue(self, gameState, agentIndex, depth):
        maxScore = -float('inf')
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            score = self.value(successor, agentIndex + 1, depth)
            if score > maxScore:
                maxScore = score
        return maxScore
 
    def expValue(self, gameState, agentIndex, depth):
        """
        Returns the expected (average) value over all legal ghost actions,
        modelling each ghost as choosing uniformly at random.
        """
        actions = gameState.getLegalActions(agentIndex)
        if not actions:
            return self.evaluationFunction(gameState)
 
        total = 0.0
        prob = 1.0 / len(actions)  # uniform distribution
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            total += prob * self.value(successor, agentIndex + 1, depth)
        return total
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
 
    # --- Base score ---
    score = currentGameState.getScore()
 
    # --- Closest food reward ---
    if foodList:
        minFoodDist = min(manhattanDistance(pos, food) for food in foodList)
        score += 1.0 / (minFoodDist + 1) * 10
    
    # --- Penalty for food remaining ---
    score -= len(foodList) * 4
 
    # --- Ghost interactions ---
    for ghostState in ghostStates:
        ghostPos = ghostState.getPosition()
        dist = manhattanDistance(pos, ghostPos)
        if ghostState.scaredTimer > 0:
            # Scared ghost: reward proximity (chase it!)
            score += 1.0 / (dist + 1) * 20
        else:
            # Active ghost: heavy penalty when close
            if dist < 2:
                score -= 500
            elif dist < 4:
                score -= 50
            else:
                score -= 1.0 / (dist + 1) * 5
 
    # --- Capsule penalty (fewer remaining is better) ---
    score -= len(capsules) * 8
 
    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
# Abbreviation
better = betterEvaluationFunction


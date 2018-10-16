# myTeam.py
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

from game import Actions
from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DefensiveAgent', second = 'OffensiveAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]


######################################
# Method 2 - Heuristic Search Agents #
######################################

class BasicAgent(CaptureAgent):

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        # Get the x-axis value of the middle wall in the game board
        self.midWidth = gameState.data.layout.width/2
        # Get the legal positions that agents could possibly be.
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        # Use a maze distance calculator
        self.distancer.getMazeDistances()
        # Define which mode the agent is in
        self.offenseTactic = False
        # Get enemies' index
        self.enemyIndices = self.getOpponents(gameState)
        # Initialize position probability distribution of each enemy by assigning every legal position the same probability 
        self.enemyPositionitionDistribution = {}
        for enemy in self.enemyIndices:
            self.enemyPositionitionDistribution[enemy] = util.Counter()
            self.enemyPositionitionDistribution[enemy][gameState.getInitialAgentPosition(enemy)] = 1.


    def initializeEnemyPositionitionDistribution(self, enemy):
        # Initialize position probability distribution of a specified enemy by assigning every legal position the same probability
        self.enemyPositionitionDistribution[enemy] = util.Counter()
        for position in self.legalPositions:
            self.enemyPositionitionDistribution[enemy][position] = 1.0
        self.enemyPositionitionDistribution[enemy].normalize()


    def forwardStep(self, enemy, gameState):
        # Calculate position probability distribution for next step and update
        newDistribution = util.Counter()
        for thisPosition in self.legalPositions:
            nextPositionDistribution = util.Counter()
            possiblePositions = [(thisPosition[0]+1, thisPosition[1]), (thisPosition[0]-1, thisPosition[1]), (thisPosition[0], thisPosition[1]+1), (thisPosition[0], thisPosition[1]-1)]
            for position in possiblePositions:
                if position in self.legalPositions:
                    nextPositionDistribution[position] = 1.0
                else:
                    pass
            nextPositionDistribution.normalize()
            for nextPosition, probability in nextPositionDistribution.items():
                newDistribution[nextPosition] += probability * self.enemyPositionitionDistribution[enemy][thisPosition]
        newDistribution.normalize()
        self.enemyPositionitionDistribution[enemy] = newDistribution


    def observe(self, enemy, observation, gameState):
        # Calculate the position probability distribution for specified enemy
        observedDistance = observation[enemy]
        myPosition = gameState.getAgentPosition(self.index)
        newDistribution = util.Counter()
        for position in self.legalPositions:
            trueDistance = util.manhattanDistance(myPosition, position)
            emissionModel = gameState.getDistanceProb(trueDistance, observedDistance)
            # if self.red:
            #     print self.index, myPosition, position, observedDistance, trueDistance, emissionModel
            if self.red:
                bePacman = position[0] < self.midWidth
            else:
                bePacman = position[0] > self.midWidth
            # If distance is less than 5, we can observe it
            if trueDistance <= 5:
                newDistribution[position] = 0.
            elif bePacman != gameState.getAgentState(enemy).isPacman:
                newDistribution[position] = 0.
            else:
                # Calculate real probability by multiplying observation probability and emission probability
                newDistribution[position] = self.enemyPositionitionDistribution[enemy][position] * emissionModel
        if newDistribution.totalCount() == 0:
            self.initializeEnemyPositionitionDistribution(enemy)
        else:
            newDistribution.normalize()
            self.enemyPositionitionDistribution[enemy] = newDistribution


    def chooseAction(self, gameState):
        # Choose an action based on evaluated values of next possible states 
        myPosition = gameState.getAgentPosition(self.index)
        observedDistances = gameState.getAgentDistances()
        newState = gameState.deepCopy()
        for enemy in self.enemyIndices:
            enemyPosition = gameState.getAgentPosition(enemy)
            if enemyPosition:
                newDistribution = util.Counter()
                newDistribution[enemyPosition] = 1.0
                self.enemyPositionitionDistribution[enemy] = newDistribution
            else:
                self.forwardStep(enemy, gameState)
                self.observe(enemy, observedDistances, gameState)
        for enemy in self.enemyIndices:
            probablePosition = self.enemyPositionitionDistribution[enemy].argMax()
            conf = game.Configuration(probablePosition, Directions.STOP)
            newState.data.agentStates[enemy] = game.AgentState(conf, newState.isRed(probablePosition) != newState.isOnRedTeam(enemy))
        action = self.getMaxEvaluatedAction(newState, depth=2)[1]
        return action


    def getMaxEvaluatedAction(self, gameState, depth):
        # Evaluate state by averaging all possible next states' value
        if depth == 0 or gameState.isOver():
            return self.evluateState(gameState), Directions.STOP
        actions = gameState.getLegalActions(self.index)
        actions.remove(Directions.STOP)
        nextGameStates = [gameState.generateSuccessor(self.index, action)
                                 for action in actions] 
        scores = []
        for state in nextGameStates:
            newStates = [state]
            for enemy in self.enemyIndices:
                tmp = [state.generateSuccessor(enemy, action) for action in state.getLegalActions(enemy) for state in newStates]
                newStates = tmp
            nextScores = [self.getMaxEvaluatedAction(state, depth - 1)[0] for state in newStates]
            scores.append(sum(nextScores)/len(nextScores))
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if
                         scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        return bestScore, actions[chosenIndex]


    def getEnemyDistances(self, gameState):
        # Return all enemies' positions
        distances = []
        for enemy in self.enemyIndices:
            enemyPosition = gameState.getAgentPosition(enemy)
            myPosition = gameState.getAgentPosition(self.index)
            if enemyPosition:
                pass
            else:
                enemyPosition = self.enemyPositionitionDistribution[enemy].argMax()
            distances.append((enemy, self.distancer.getDistance(myPosition, enemyPosition)))
        return distances


    def evluateState(self, gameState):
        """
        Evaluate the utility of a game state.
        """
        util.raiseNotDefined()


class OffensiveAgent(BasicAgent):
    # An offensive agent have more probability to eat enemies' food

    def registerInitialState(self, gameState):
        BasicAgent.registerInitialState(self, gameState)
        # Define weights for features
        self.returning = False
        self.weights = util.Counter()
        self.weights['score'] = 2
        self.weights['numFood'] = -100
        self.weights['numCapsule'] = -10000
        self.weights['closetDistanceToFood'] = -3
        self.weights['closetDistanceToMiddle'] = -2
        # self.weights['closetDistanceToGhosts'] = 500
        self.weights['closetDistanceToGhosts'] = 100
        self.weights['closetDistanceToCapsule'] = -5

    def chooseAction(self, gameState):
        enemyScareTime = [gameState.getAgentState(enemy).scaredTimer for enemy in self.enemyIndices]
        score = self.getScore(gameState)
        if score < 7:
            carryLimit = 6
        else:
            carryLimit = 4
        if gameState.getAgentState(self.index).numCarrying < carryLimit and len(self.getFood(gameState).asList()) > 2:
            self.returning = False
        else:
            if min(enemyScareTime) > 5:
                self.returning = False
            else:
                self.returning = True
        return BasicAgent.chooseAction(self, gameState)


    def evluateState(self, gameState):
        features = self.getFeatures(gameState)
        return features * self.weights
        # if self.returning:
        #     return - 2 * features['closetDistanceToMiddle'] + 500 * features['closetDistanceToGhosts']
        # else:
        #     if features['minEnemyScareTime'] <= 6 and features['closetDistanceToGhosts'] < 4:
        #         features['closetDistanceToGhosts'] *= -1
        #     return 2 * features['score'] - 100 * features['numFood'] - \
        #            3 * features['closetDistanceToFood'] - 10000 * features['numCapsule']- \
        #            5 * features['closetDistanceToCapsule'] + 100 * features['closetDistanceToGhosts']

    def getFeatures(self, gameState):
        if self.returning:
            myPosition = gameState.getAgentPosition(self.index)
            closetDistanceToMiddle = min([self.distancer.getDistance(myPosition, (self.midWidth, i))
                                 for i in range(gameState.data.layout.height)
                                 if (self.midWidth, i) in self.legalPositions])
            distanceToEnemies = self.getEnemyDistances(gameState)
            distanceToGhosts = [distance for index, distance in distanceToEnemies if
                                 not gameState.getAgentState(index).isPacman]
            closetDistanceToGhosts = min(distanceToGhosts) if len(distanceToGhosts) else 0
            features = util.Counter()
            features['closetDistanceToGhosts'] = closetDistanceToGhosts
            features['closetDistanceToMiddle'] = closetDistanceToMiddle
            return features
        else:
            myPosition = gameState.getAgentPosition(self.index)
            closetDistanceToMiddle = min([self.distancer.getDistance(myPosition, (self.midWidth, i))
                                     for i in range(gameState.data.layout.height)
                                     if (self.midWidth, i) in self.legalPositions])
            foodToEat = self.getFood(gameState).asList()
            distanceToFood = [self.distancer.getDistance(myPosition, food) for
                                 food in foodToEat]
            closetDistanceToFood = min(distanceToFood) if len(distanceToFood) else 0            
            distanceToEnemies = self.getEnemyDistances(gameState)
            distanceToGhosts = [distance for index, distance in distanceToEnemies if
                                 not gameState.getAgentState(index).isPacman]
            closetDistanceToGhosts = min(distanceToGhosts) if len(distanceToGhosts) else 0
            if closetDistanceToGhosts >= 4:
                closetDistanceToGhosts = 0      
            if self.red:
                capsuleToEat = gameState.getBlueCapsules()
            else:
                capsuleToEat = gameState.getRedCapsules()
            for enemy in self.enemyIndices:
                if not gameState.getAgentState(enemy).isPacman:
                    enemyPosition = gameState.getAgentPosition(enemy)
                    if enemyPosition != None:
                        distanceToGhosts.append(self.distancer.getDistance(myPosition, enemyPosition))
            distanceToCapsule = [self.distancer.getDistance(myPosition, capsule) for capsule in
                                        capsuleToEat]
            closetDistanceToCapsule = min(distanceToCapsule) if len(distanceToCapsule) else 0
            enemyScareTime = [gameState.getAgentState(enemy).scaredTimer for enemy
                                 in self.enemyIndices]
            minEnemyScareTime = min(enemyScareTime)
            if minEnemyScareTime <= 6 and closetDistanceToGhosts < 4:
                closetDistanceToGhosts *= -1
            features = util.Counter()
            features['numFood'] = len(foodToEat)
            features['score'] = self.getScore(gameState)
            features['numCapsule'] = len(distanceToCapsule)
            features['closetDistanceToFood'] = closetDistanceToFood
            features['closetDistanceToGhosts'] = closetDistanceToGhosts
            features['closetDistanceToCapsule'] = closetDistanceToCapsule
            return features



class DefensiveAgent(BasicAgent):
    # A defensive agent has more probability to defend its food
    def registerInitialState(self, gameState):
        BasicAgent.registerInitialState(self, gameState)
        self.offenseTactic = False
        self.weights = util.Counter()
        self.weights['score'] = 2
        self.weights['numFood'] = 100
        self.weights['numPacman'] = -99999
        self.weights['numCapsule'] = 0
        self.weights['closetDistanceToFood'] = -3
        self.weights['closetDistanceToGhosts'] = 1
        self.weights['closetDistanceToPacmans'] = -10
        self.weights['closetDistanceToCapsule'] = -1


    def chooseAction(self, gameState):
        invaders = [enemy for enemy in self.enemyIndices if
                    gameState.getAgentState(enemy).isPacman]
        enemyScareTime = [gameState.getAgentState(enemy).scaredTimer for enemy in
                         self.enemyIndices]
        if len(invaders) == 0 or min(enemyScareTime) > 8:
            self.offenseTactic = True
        else:
            self.offenseTactic = False
        return BasicAgent.chooseAction(self, gameState)


    def evluateState(self, gameState):
        features = self.getFeatures(gameState)
        return features * self.weights
        # if self.offenseTactic == False:
        #     return -999999 * features['numPacman'] - 10 * features['closetDistanceToPacmans'] - features['closetDistanceToCapsule']
        # else:
        #     return 2 * features['score'] - 100 * features['numFood'] - \
        #            3 * features['closetDistanceToFood'] + features['closetDistanceToGhosts']

    def getFeatures(self, gameState):
        if self.offenseTactic == False:
            myPosition = gameState.getAgentPosition(self.index)
            distanceToEnemies = self.getEnemyDistances(gameState)
            distanceToPacmans = [distance for index, distance in distanceToEnemies if
                        gameState.getAgentState(index).isPacman]
            closetDistanceToPacmans = min(distanceToPacmans) if len(distanceToPacmans) else 0
            capsuleToDefend = self.getCapsulesYouAreDefending(gameState)
            distanceToCapsule = [self.getMazeDistance(myPosition, capsule) for capsule in
                        capsuleToDefend]
            closetDistanceToCapsule = min(distanceToCapsule) if len(distanceToCapsule) else 0
            features = util.Counter()
            features['numPacman'] = len(distanceToPacmans)
            features['closetDistanceToPacmans'] = closetDistanceToPacmans
            features['closetDistanceToCapsule'] = closetDistanceToCapsule
            return features
        else:
            myPosition = gameState.getAgentPosition(self.index)
            
            distanceToEnemies = self.getEnemyDistances(gameState)
            distanceToGhosts = [distance for index, distance in distanceToEnemies if
                                 not gameState.getAgentState(index).isPacman]
            closetDistanceToGhosts = min(distanceToGhosts) if len(distanceToGhosts) else 0
            
            foodToEat = self.getFood(gameState).asList()
            distanceToFood = [self.distancer.getDistance(myPosition, food) for food in
                             foodToEat]
            closetDistanceToFood = min(distanceToFood) if len(distanceToFood) else 0            
            features = util.Counter()
            features['numFood'] = len(foodToEat)
            features['score'] = self.getScore(gameState)
            features['numGhost'] = len(distanceToGhosts)
            features['closetDistanceToFood'] = closetDistanceToFood
            features['closetDistanceToGhosts'] = closetDistanceToGhosts
            return features
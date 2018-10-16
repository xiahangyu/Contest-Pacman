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

#########################
# Method 1 - Side Agent #
#########################

class SideAgent(CaptureAgent):

  def registerInitialState(self, gameState):

    CaptureAgent.registerInitialState(self, gameState)

    self.foodTotal = len(self.getFood(gameState).asList())
    self.foodDefendingPrev = self.getFoodYouAreDefending(gameState).asList()
    self.allyIndex = (self.index+2)%4
    if self.index < 2:
      self.side = "Top"
    else:
      self.side = "Bottom"

    self.searchTree = SearchTree()

    self.foodLost = []
    self.patience = 0

  def chooseAction(self, gameState):
    # start = time.time()

    myPos = gameState.getAgentState(self.index).getPosition()
    allyPos = gameState.getAgentState(self.allyIndex).getPosition()
    foodDefendingNow = self.getFoodYouAreDefending(gameState).asList()
    foodLost = [food for food in self.foodDefendingPrev if food not in foodDefendingNow]

    _, foodLost = self.isEnemiesNearby(myPos, allyPos, foodLost)

    if len(foodLost)!=0:
      self.foodLost = foodLost
      self.patience = self.getMazeDistance(myPos, foodLost[0]) - 5

    if self.patience > 0:
      self.patience -= 1
    else:
      self.patience = 0
      self.foodLost = []

    # print self.foodLost, self.patience
    myState = gameState.getAgentState(self.index)
    if myState.isPacman:
      best_action = self.actionAsPacman(gameState)
    else:
      best_action = self.actionAsGhost(gameState)
    self.foodDefendingPrev = foodDefendingNow
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
    return best_action



  """
  ==============
  Help functions
  ==============
  """
  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def topFoods(self, gameState):
    foodList = self.getFood(gameState).asList()
    topFoods = [(x,y) for (x,y) in foodList if y>=gameState.data.layout.height/2]
    return topFoods

  def bottomFoods(self, gameState):
    foodList = self.getFood(gameState).asList()
    bottomFoods = [(x,y) for (x,y) in foodList if y<gameState.data.layout.height/2]
    return bottomFoods

  def sideFoods(self, gameState):
    foodList = []
    if self.side == "Top":
      foodList = self.topFoods(gameState)
      if len(foodList)==0:
        foodList = self.bottomFoods(gameState)
    else:
      foodList = self.bottomFoods(gameState)
      if len(foodList)==0:
        foodList = self.topFoods(gameState)
    return foodList

  def getBorders(self, gameState):
    width = gameState.data.layout.width
    height = gameState.data.layout.height
    if self.red:
      return [(width/2-2, h) for h in range(0, height) if (width/2-2, h) not in gameState.data.layout.walls.asList()]
    else:
      return [(width/2+1, h) for h in range(0, height) if (width/2+1, h) not in gameState.data.layout.walls.asList()]

  def getCapsules(self, gameState):
    if self.red:
      return gameState.getBlueCapsules()
    else:
      return gameState.getRedCapsules()

  def findClosestFood(self, gameState):
    bestAction = None
    minDistance = 1200
    for action in gameState.getLegalActions(self.index):
      successor = self.getSuccessor(gameState, action)
      myPos = successor.getAgentState(self.index).getPosition()

      foodList = self.sideFoods(gameState)
      minDistanceToFood, food  = min([(self.getMazeDistance(myPos, food), food) for food in foodList])
      if minDistanceToFood < minDistance:
        minDistance = minDistanceToFood
        bestAction = action

    return bestAction, minDistance, food


  """
  =========================
  Action decision as Pacman
  =========================
  """
  def actionAsPacman(self, gameState):
    """
    Finds an action to take as a pacman.
    """
    myPos = gameState.getAgentState(self.index).getPosition()
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    protectorPos = [a.getPosition() for a in enemies if (not a.isPacman and a.getPosition() != None and a.scaredTimer==0)]
    foodList = self.getFood(gameState).asList()
    if len(foodList)==0 or len(self.foodLost)!=0 and gameState.getAgentState(self.index).scaredTimer==0:
      return self.searchTree.search(self, gameState, protectorPos, self.findWayBack)

    isProtectorNear, nearProtectors = self.isProtectorsNearby(gameState, myPos, protectorPos)
    if not isProtectorNear:
      return self.actionWithNoDanger(gameState, nearProtectors)
    else:
      return self.actionInDanger(gameState, nearProtectors)


  def isProtectorsNearby(self, gameState, myPos, protectorPos):
    nearProtectors = []
    isProtectorNear = False

    for protectorPos in protectorPos:
      distance = self.getMazeDistance(myPos, protectorPos)
      if distance <= 5:
        nearProtectors.append(protectorPos)
        isProtectorNear = True
    return isProtectorNear, nearProtectors


  def actionWithNoDanger(self, gameState, nearProtectors):
    """
    Finds an action to take as a pacman when no ghosts have been observed.
    """
    best_action, minDistanceToFood, food = self.findClosestFood(gameState)

    timeLeft = gameState.data.timeleft/4+1
    myPos = gameState.getAgentState(self.index).getPosition()
    distanceToBorder = min([self.getMazeDistance(myPos, border) for border in self.getBorders(gameState)])
    distanceFoodToBorder = min([self.getMazeDistance(food, border) for border in self.getBorders(gameState)])

    carrying = gameState.getAgentState(self.index).numCarrying
    foodLeft = len(self.getFood(gameState).asList())
    if (minDistanceToFood + distanceFoodToBorder) >= timeLeft or carrying >= 10 or minDistanceToFood >= distanceToBorder*foodLeft/self.foodTotal and carrying > 0:
      best_action = self.searchTree.search(self, gameState, nearProtectors, self.findWayBack)
    return best_action

  def actionInDanger(self, gameState, nearProtectors):
    """
    Finds an action to take as a pacman when ghosts are nearby.
    """
    timeLeft = gameState.data.timeleft/4+1
    capsulesPos = self.getCapsules(gameState)
    if len(capsulesPos) != 0:
      best_action = self.searchTree.search(self, gameState, nearProtectors, self.findCapsulesOrWayBack)
    else:
      best_action = self.searchTree.search(self, gameState, nearProtectors, self.findWayBack)
    return best_action

  def findWayBack(self, gameState, node):
    disToBorder = min([self.getMazeDistance(node.myPos, border) for border in self.getBorders(gameState)])
    value = -disToBorder
    if node.myPos in node.enemiesPos:
      value = -1000
    return value

  def findCapsulesOrWayBack(self, gameState, node):
    disToBorder = min([self.getMazeDistance(node.myPos, border) for border in self.getBorders(gameState)])
    value = -disToBorder
    if len(node.capsules)!=0:
      disToCapsule = min([self.getMazeDistance(node.myPos, capsule) for capsule in node.capsules])
      if disToBorder > disToCapsule:
        value = -disToCapsule
      else:
        value = -disToBorder
    if node.myPos in node.enemiesPos:
      value = -1000
    return value


  """
  ========================
  Action decision as Ghost
  ========================
  """
  def actionAsGhost(self, gameState):
    """
    Finds an action to take as a ghost.
    """
    myPos = gameState.getAgentState(self.index).getPosition()
    allyPos = gameState.getAgentState(self.allyIndex).getPosition()
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invadersPos = [a.getPosition() for a in enemies if a.isPacman and a.getPosition() != None]
    isInvadersNear, nearInvaders = self.isEnemiesNearby(myPos, allyPos, invadersPos)

    if gameState.getAgentState(self.index).scaredTimer>0:
      return self.actionAttack(gameState)
    elif isInvadersNear:
      return self.actionWithInvaders(gameState, nearInvaders)
    elif len(self.foodLost)!=0:
      action = self.actionWithInvaders(gameState, self.foodLost)
      print self.foodLost, self.patience, action
      return action

    ghostsPos = [a.getPosition() for a in enemies if (not a.isPacman and a.getPosition() != None) and a.scaredTimer==0]
    isGhostsNear, nearGhosts = self.isEnemiesNearby(myPos, allyPos, ghostsPos)
    if isGhostsNear:
      return self.actionWithGhosts(gameState, nearGhosts)

    return self.actionAttack(gameState)

  def isEnemiesNearby(self, myPos, allyPos, enemiesPos):
    nearEnemies = []
    isEnemiesNear = False

    for enemyPos in enemiesPos:
      myDistance = self.getMazeDistance(myPos, enemyPos)
      allyDistance = self.getMazeDistance(allyPos, enemyPos)
      if myDistance <= allyDistance:
        nearEnemies.append(enemyPos)
        isEnemiesNear = True
    return isEnemiesNear, nearEnemies

  def actionWithInvaders(self, gameState, nearInvaders):
    bestAction = None
    minDistance = 1200

    for action in gameState.getLegalActions(self.index):
      successor = self.getSuccessor(gameState, action)
      myState = successor.getAgentState(self.index)
      if myState.isPacman:
        continue
      myPos = myState.getPosition()

      minDistanceToInvaders = min([self.getMazeDistance(myPos, nearInvader) for nearInvader in nearInvaders])

      if minDistanceToInvaders < minDistance:
        minDistance = minDistanceToInvaders
        bestAction = action
      elif minDistanceToInvaders == minDistance and action in ["South","North"]:
        minDistance = minDistanceToInvaders
        bestAction = action

    return bestAction

  def actionWithGhosts(self, gameState, nearGhosts):
    bestAction = None
    minDistance = 1200

    borders = self.getBorders(gameState)
    _, nearestBordersToGhost = min([(self.getMazeDistance(border, ghost), border) for border in borders for ghost in nearGhosts])

    for action in gameState.getLegalActions(self.index):
      successor = self.getSuccessor(gameState, action)
      myPos = successor.getAgentState(self.index).getPosition()
      distanceToBorder = self.getMazeDistance(myPos, nearestBordersToGhost)
      if distanceToBorder < minDistance:
        minDistance = distanceToBorder
        bestAction = action
    return bestAction

  def actionAttack(self, gameState):
    foodList = self.sideFoods(gameState)
    if len(foodList)==0:
      return 'Stop'

    bestAction = None
    minDistance = 1200
    for action in gameState.getLegalActions(self.index):
      successor = self.getSuccessor(gameState, action)
      myPos = successor.getAgentState(self.index).getPosition()

      minDistanceToFood = min([self.getMazeDistance(myPos, food) for food in foodList])
      if minDistanceToFood < minDistance:
        minDistance = minDistanceToFood
        bestAction = action
    return bestAction

  def findWayFightBack(self, gameState, node):
    value = 77
    disToEnemies = min([self.getMazeDistance(node.myPos, enemy) for enemy in node.enemiesPos])
    value = -disToEnemies
    if node.myPos in node.enemiesPos:
      value = -1000
    return value


class SearchTree:

  def search(self, agent, gameState, root_enemiesPos, evaluate_fun):
    capsules = agent.getCapsules(gameState)
    root_pos = gameState.getAgentState(agent.index).getPosition()
    root_node = TreeNode(root_pos, root_enemiesPos, capsules)
    root_node.value = evaluate_fun(gameState, root_node)

    start = time.time()
    q = util.Queue()
    q.push(root_node)
    closeList = {}
    while not q.isEmpty() and time.time() - start <= 0.7:
      curr_node = q.pop()
      curr_myPos = curr_node.myPos
      curr_enemiesPos = curr_node.enemiesPos
      curr_capsules = curr_node.capsules

      if curr_myPos in agent.getBorders(gameState):
        continue

      if curr_myPos in curr_capsules:
        curr_capsules.remove(curr_myPos)
        continue
      
      actions = ['North', 'South', 'East', 'West', 'Stop']
      for action in actions:
      	dx, dy = Actions.directionToVector(action)
        child_myPos = (curr_myPos[0]+dx, curr_myPos[1]+dy)
        child_depth = curr_node.depth+1

        if child_myPos in gameState.data.layout.walls.asList():
        	continue

        if child_myPos not in closeList or child_depth<closeList[child_myPos]:
        	closeList[child_myPos] = child_depth
        else:
        	continue

        child_enemiesPos = []
        for x,y in curr_enemiesPos:
          possibleEnemiesPos = [ (x+dx,y+dy) for (dx, dy) in [(0,1), (0,-1), (1,0), (-1,0), (0,0)] ]
          possibleEnemiesPos = [pos for pos in possibleEnemiesPos if pos not in gameState.data.layout.walls.asList()]
          nearestEnemiesPos = self.allMin([(agent.getMazeDistance(child_myPos, enemiesPos), enemiesPos) for enemiesPos in possibleEnemiesPos])
          for pos in nearestEnemiesPos:
          	if pos not in child_enemiesPos:
          		child_enemiesPos.append(pos)

        child_node = TreeNode(child_myPos, child_enemiesPos, curr_capsules, curr_node)
        child_node.value = evaluate_fun(gameState, child_node)
        curr_node.childs[action] = child_node

        if child_node.myPos not in child_node.enemiesPos:
          q.push(child_node)

    root_node = self.backPropagation(root_node)
    # print agent.index, root_node.bestAction
    # for action in root_node.childs:
    #   child = root_node.childs[action]
    #   if child!=None:
    # 		print "  ", action, child.value
    return root_node.bestAction

  def allMin(self, listOfTupple):
  	key, value = min(listOfTupple)
  	return [value for (k,v) in listOfTupple if k==key]

  def backPropagation(self, node):
  	if node.isLeafNode():
  		return node

  	maxValue = -10000
  	miniDepth = 10000
  	for action in node.childs:
  	  child = node.childs[action]
  	  if child!=None :
  		child = self.backPropagation(child)
  		if child.value > maxValue:
  			maxValue = child.value
  			miniDepth = child.branchDepth
  			bestAction = action
  		elif child.value == maxValue and child.branchDepth < miniDepth:
  			miniDepth = child.branchDepth
  			bestAction = action

  	updated_node = node
  	updated_node.value = maxValue
  	updated_node.branchDepth = miniDepth
  	updated_node.bestAction = bestAction
  	return updated_node

class TreeNode:

	def __init__(self, myPos, enemiesPos, capsules, parent = None):
		if parent == None:
			self.depth = 1
			self.branchDepth = 1
		else:
			self.depth = parent.depth + 1
			self.branchDepth = parent.branchDepth + 1

		self.myPos = myPos
		self.enemiesPos = enemiesPos
		self.capsules = capsules

		self.parent = parent
		self.childs = {"North":None, "South":None, "East":None, "West":None, "Stop":None}
		self.bestAction = None
		self.value = None

	def isLeafNode(self):
		n_childs = sum([child==None for child in self.childs.values()])
		return n_childs==5


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
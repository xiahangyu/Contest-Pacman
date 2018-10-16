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
import sys
sys.path.append('teams/Hangyu-Shawn-Chris/')

from game import Actions
from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint

from treeNode import TreeNode
from searchTree import SearchTree

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'MCTSAgent', second = 'MCTSAgent'):
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

##########
# Agents #
##########

class MCTSAgent(CaptureAgent):

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
      return [(width/2-3, h) for h in range(0, height) if (width/2-3, h) not in gameState.data.layout.walls.asList()]
    else:
      return [(width/2+2, h) for h in range(0, height) if (width/2+2, h) not in gameState.data.layout.walls.asList()]

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
      return self.actionWithInvaders(gameState, self.foodLost)

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

  def isEnemiesNearbyIndex(self, myPos, allyPos, enermiesStates):
    nearEnemies = []
    isEnemiesNear = False
    
    for enermy in enermiesStates:
      myDistance = self.getMazeDistance(myPos, enermy.getPosition())
      allyDistance = self.getMazeDistance(allyPos, enermy.getPosition())
      if myDistance <= allyDistance:
        nearEnemies.append(enermy.index)
        isEnemiesNear = True
    return isEnemiesNear, nearEnemies

  def actionWithInvaders(self, gameState, nearInvaders):
    bestAction = None
    minDistance = 1200
    # Use Monte Carlo to kill the invader if very close
    myPos = gameState.getAgentState(self.index).getPosition()
    enermiesIndex = self.getOpponents(gameState)
    nearestDistance = 1200
    nearestEnermy = 0 
    for enermy in enermiesIndex:
      enermyPos = gameState.getAgentPosition(enermy)
      if enermyPos!= None:
        #print(myPos,enermyPos)
        if self.getMazeDistance(myPos,enermyPos) < nearestDistance:
          nearestDistance = self.getMazeDistance(myPos,enermyPos)
          nearestEnermy = enermy

    if(self.getMazeDistance(myPos,nearInvader)< 5 for nearInvader in nearInvaders):
      root = Node(gameState,self.index,nearestEnermy,None,0,None)
      root.expand()
      first_layer_sons = root.sonNodes
      second_layer_sons = []
      third_layer_sons = []
      # First Layer triversal/search
      for node in first_layer_sons:
        node.triversal()
        if(node.isEnd == False):
          node.expand()
          for son_node in node.sonNodes:
            second_layer_sons.append(son_node)
      # Second Layer triversal/search
      for node in second_layer_sons:
        node.triversal()
        if(node.isEnd == False):
          node.expand()
          for son_node in node.sonNodes:
            third_layer_sons.append(son_node)
      # Third Layer triversal/search
      for node in third_layer_sons:
        node.triversal()
      return root.getBestAction()
      
    else:  
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

  """
  ========================
  Monte Carlo Tree Search Help Function
  ========================
  """
class Node:
    def __init__(self,gameState,agentIndex,enermyIndex,positionDefending,turn,parentNode):
        self.gameState = gameState
        self.agentIndex = agentIndex
        self.enermyIndex = enermyIndex
        self.foodsDefending = positionDefending
        self.turn = turn #0 for my turn, 1 for enermy turn
        self.parentNode = parentNode
        self.sonActions = []
        self.sonNodes = []
        self.winCount = 0
        self.lossCount = 0
        self.drawCount = 0
        self.totalCount = 0
        self.isEnd = False

    def triversal(self):
      # if enermy dead or go back, wincounts ++
      if(self.gameState.getAgentState(self.enermyIndex).isPacman == False):
        self.getWin()
      else:
        self.getDraw()
      return
      # if enery achieved any one of positionDefending loss ++
      #self.getLoss()
      #return 
      # None above draw ++
      #self.getDraw()
      #return 
    def expand(self):
      if self.turn == 0:
        successorGameState = []
        for move in self.gameState.getLegalActions(self.agentIndex):
          new_state = self.gameState.generateSuccessor(self.agentIndex,move)
          successorGameState.append(new_state)
          self.sonActions.append(move)
          self.sonNodes.append(Node(new_state,self.agentIndex,self.enermyIndex,self.foodsDefending,1,self))

      else:
        successorGameState = []
        for move in self.gameState.getLegalActions(self.enermyIndex):
          new_state = self.gameState.generateSuccessor(self.enermyIndex,move)
          successorGameState.append(new_state)
          self.sonNodes.append(Node(new_state,self.agentIndex,self.enermyIndex,self.foodsDefending,0,self))
          
    def getWin(self):
      self.winCount += 1
      self.totalCount += 1
      self.isEnd = True
      if(self.parentNode != None):
        self.parentNode.getWin()
    def getLoss(self):
      self.lossCount += 1
      self.totalCount += 1
      if(self.parentNode != None):
        self.parentNode.getLoss()
    def getDraw(self):
      self.drawCount += 1
      self.totalCount += 1
      if(self.parentNode != None):
         self.parentNode.getDraw()

    def getWinRate(self):
      return self.winCount/float(self.totalCount)
    def getBestAction(self):
      winRates = []
      for i in self.sonNodes:
        winRates.append(i.getWinRate())
      index = winRates.index(max(winRates))
      return self.sonActions[index]

          
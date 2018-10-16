
from treeNode import TreeNode
from game import Actions
from captureAgents import CaptureAgent
import util, time
import game


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
    while not q.isEmpty() and time.time() - start <= 0.95:
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
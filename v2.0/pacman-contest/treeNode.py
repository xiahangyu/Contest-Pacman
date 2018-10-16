
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


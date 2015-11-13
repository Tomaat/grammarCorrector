import nltk

class dfirst(object):
	def __init__(self,tree):
		self.ctype = nltk.tree.Tree
		if not isinstance(tree,self.ctype):
			msg = 'wrong input of type %s for type %s!'%(str(type(tree) ), str(self.ctype)) 
			raise TypeError(msg)
		self.currentTree = tree
		self.current = -1
		self.todo = []
	
	def __iter__(self):
		return self
	
	def __next__(self):
		return self.next()
	
	def next(self):
		if self.current == -1:
			self.current += 1
			return self.currentTree.label()
		elif self.current < len(self.currentTree):
			ans = self.currentTree[self.current]
			self.current += 1
			if not isinstance(ans,self.ctype):
				return ans
			else:
				self.todo += [ans]
				return ans.label()
		elif len(self.todo) > 0:
			self.currentTree = self.todo.pop(0)
			self.current = 0
			return self.next()
		else:
			raise StopIteration()


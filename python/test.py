class Test:
	def test1(self, a, b):
		c = self.test2(a,b)
		print c

	def test2(self, a, b):
		c = a+b
		return c



if __name__ == '__main__':
	t = Test()
	t.test1(1,2)
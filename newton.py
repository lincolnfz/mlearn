# -*- coding: utf-8 -*-
__author__ = 'lincolnfz@gmail.com'
'''
牛顿法求方程根
'''

'''
def fun1(x):
	return float(2*(x**2) - 13*x + 1)

def fun2(x):
	return float(4*x - 13)
'''

def fun1(x):
	return float(x**2-2)

def fun2(x):
	return float(2*x)

def newton( deta ):
	x = 2; x1 = 0
	tedeta = 1
	while tedeta >= deta:
		x1 = x - fun1(x) / fun2(x)
		tedeta = abs(x1 - x)
		x = x1
	return x1

class Person(object):
	"""docstring for Person"""
	def __init__(self, name):
		super(Person, self).__init__()
		self.name = name
		

if __name__ == '__main__':
	print (newton(0.00001))
	#x = 10
	#print fun1(x) / fun2(x)
	'''a = 3
	b = 5
	print float(a)/float(b)'''
	alua = Person('alua')
	alua.age = 2
	print(alua.name)
	print(alua.age)
	del alua.age
	print(Person)
	print(alua)

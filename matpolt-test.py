import matplotlib
import numpy
import scipy
import matplotlib.pyplot as plt
import math
from sklearn import datasets, svm, metrics

#plt.plot([1,2,3])
#plt.ylabel('some numbers')
#plt.show()

def sigmoid(intX):
	#print(intX)
	return 1.0/(1+math.exp(-intX))

#print( 'xx=%lf, x=%f' % (math.pi ,math.exp(1)) )
if __name__ == '__main__':
	numat = numpy.mat([-10,-450,-32.5,0])
	m,n = numpy.shape(numat)
	lab = numpy.ones((n,1))
	#print(lab)
	val = sigmoid(numat*lab)
	xxa= numpy.arange(-1,1.06,0.01)
	print( 'xx=%lf' % (val) )
	print( 'sin=%lf' % numpy.sin(90 * numpy.pi / 180) )
	print( 'xx=%lf' % (len(xxa)) )
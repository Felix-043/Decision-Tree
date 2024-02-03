import numpy
from sklearn import metrics
import matplotlib.pyplot as plt 

actual = numpy.random.binomial(1, 0.9, size = 1000)
predicted = numpy.random.binomial(1, 0.9, size = 1000)

Accuracy = metrics.accuracy_score(actual, predicted)
print(Accuracy)
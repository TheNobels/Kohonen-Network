import numpy as numpy
from numpy.linalg import norm
from sklearn.preprocessing import normalize
import pandas as panda
from matplotlib import pyplot as pyplot
from sklearn.metrics.pairwise import euclidean_distances


#Load training data from training.txt
training = panda.read_csv("training.txt").values

#Load parameters values from params.csv
param = panda.read_csv("params.txt").values

variables = int(param[0,0])
learning_rate = param[0,1]
flag = int(param[0,2])
w0 = param[len(param[:,-2]) - variables : len(param[:,-2]), -2:]


def normalise(x):
    return normalize(x, norm = 'l2')

#Check for flag parameter for normalization
if(flag != 0):
    training_ = normalise(training)
    w0_ = normalise(w0)
else:
    training_ = training
    w0_ = w0

title = "Normalize 2-input Kohonen NN with 3 variables"
figure_size = 7.5
plot_max = numpy.max(training_) * 1.2

pyplot.clf()
f, ax = pyplot.subplots(figsize = (figure_size, figure_size))
ax.scatter(training_[:, 0], training_[:, 1], marker = '.', s = 30, color = "black")

rgb = numpy.array(["brown", "orange", "green", "blue", "purple", "grey"])

#Plotting of initial cluster centriod position
#ax.scatter(w0_[:,0], w0_[:, 1], marker = 's', s = 100, color = numpy.array(rgb[:variables]), label = "Initial")

w = w0_
for k in range(int(param[0,3])):
    for i in range(training_.shape[0]):
        net = []
        for j in range(w.shape[0]):
            net.append(training_[i] @ w[j])
        
        winner = numpy.argmax(net)

        #Plotting of cluster forming 
        #ax.scatter(training_[i, 0], training_[i, 1], marker = "o", s = 20, color = rgb[winner])

        w[winner, :] += learning_rate * (training_[i, :] - w[winner, :])
        if(flag == 1):
            w[winner, :] = normalise([w[winner, :]])

        #Plotting of cluster centriod reassignment after each training
        #pyplot.scatter(w[winner, 0], w[winner, 1], marker = '.', s = 1, color = rgb[winner])

#Plotting of final cluster centriod position.
ax.scatter(w[:, 0], w[:, 1], marker = '*', s = 200, color = numpy.array(rgb[:variables]), label = "Final")
pyplot.xlim(-plot_max, plot_max)
pyplot.ylim(-plot_max, plot_max)
pyplot.xlabel("x1")
pyplot.ylabel("x2")
pyplot.grid()


pyplot.title(title, fontsize = 18)

pyplot.show()
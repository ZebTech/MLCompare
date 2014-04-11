#-*- coding: utf-8 -*-
import matplotlib.pyplot as uniquePyPlot


def plot(x, y, title='', xlabel='', ylabel=''):
    uniquePyPlot.clf()
    figure = uniquePyPlot
    figure.scatter(x, y)
    figure.title(title)
    figure.xlabel(xlabel)
    figure.ylabel(ylabel)
    figure.autoscale(tight=True)
    figure.grid()
    figure.savefig(title + '.svg', format='svg')

#Plots multiple arrays.(Max. 5)
# plotArray[0][i]: X values for plot n°i
# plotArray[1][i]: Y values for plot n°i
def multiPlot(plotArray, title='', xlabel='', ylabel=''):
    uniquePyPlot.clf()
    figure = uniquePyPlot
    figure.title(title)
    figure.xlabel(xlabel)
    figure.ylabel(ylabel)
    markers = 'xo><v'
    colors = 'rbgyo'
    if len(plotArray[0]) > 5:
        print 'More than 5 plots given as parameters (plotArray)'
        return False
    for i in xrange(len(plotArray[0])):
        figure.scatter(
            plotArray[0][i],
            plotArray[1][i],
            marker = markers[i],
            c = colors[i]
            )

    figure.autoscale(tight=True)
    figure.grid()
    figure.savefig(title + '.svg', format='svg')

def plotLines(plotArray, title='', xlabel='', ylabel=''):
    uniquePyPlot.clf()
    figure = uniquePyPlot
    figure.title(title)
    figure.xlabel(xlabel)
    figure.ylabel(ylabel)
    markers = '+,.1234'
    colors = 'rbgyo'
    if len(plotArray[0]) > 5:
        print 'More than 5 plots given as parameters (plotArray)'
        return False
    for i in xrange(len(plotArray[0])):
        figure.plot(
            plotArray[0][i],
            plotArray[1][i],
            marker = markers[i],
            c = colors[i]
            )

    # figure.autoscale(tight=True)
    figure.grid()
    figure.savefig(title + '.svg', format='svg')
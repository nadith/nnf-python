# -*- coding: utf-8 -*-
"""
.. module:: Plot
   :platform: Unix, Windows
   :synopsis: Represent Plot class.

.. moduleauthor:: Nadith Pathirage <chathurdara@gmail.com>
"""

# Global Imports
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

# Local Imports

class Plot(object):
    """Plot draw various types of plots."""

    ##########################################################################
    # Public Interface
    ##########################################################################
    @staticmethod
    def barchart(x1, x2, x1_legand='True', x2_legand='Predicted', ylabel='Damage', title='Bar Chart', auto_label=False):
        """x1, x2 must be vectors"""
        assert(x1.ndim == 1 and x2.ndim == 1)
        assert(x1.size == x2.size)

        N = x1.size
        ind = np.arange(N)  # the x locations for the groups
        width = 0.30        # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(ind, x1, width, color='r')
        rects2 = ax.bar(ind + width, x2, width, color='y')

        # add some text for labels, title and axes ticks
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(ind + width)

        # Convert indices to string representations
        ax.set_xticklabels(tuple(map(str, ind)))
        ax.legend((rects1[0], rects2[0]), (x1_legand, x2_legand))

        def autolabel(rects):
            """
            Attach a text label above each bar displaying its height
            """
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                        '%d' % int(height),
                        ha='center', va='bottom')

        if (auto_label):
            autolabel(rects1)
            autolabel(rects2)

        plt.show()

    @staticmethod
    def regression(X1, X2, xlabel='X1', ylabel='X2', title='Regression'):
        """x1, x2, can be vectors or matrices"""
        assert(X1.size == X2.size)
    
        X1 = X1.flatten()
        X2 = X2.flatten()
    
        # Test


        fig, ax = plt.subplots()

        # Fit linear model
        fit = np.polyfit(X1, X2, deg=1)

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(X1,X2)

        # Draw the line
        ax.plot(X1, fit[0] * X1 + fit[1], color='red')

        # Scatter plot of data   
        ax.scatter(X1, X2)

        # Set info
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_title(title)

        pos = 0.20
        fig.text(pos, 0.95, "Slope:", ha="center", va="bottom", size="medium")
        fig.text(pos + 0.08, 0.95, str(round(slope, 5)), ha="center", va="bottom", size="medium",color="red")

        pos += 0.08 + 0.1
        fig.text(pos, 0.95, "R-Val:", ha="center", va="bottom", size="medium")
        fig.text(pos + 0.08, 0.95, str(round(r_value, 5)), ha="center", va="bottom", size="medium",color="red")

        pos += 0.08 + 0.1
        fig.text(pos, 0.95, "P-Val:", ha="center", va="bottom", size="medium")
        fig.text(pos + 0.08, 0.95, str(round(p_value, 5)), ha="center", va="bottom", size="medium",color="red")

        pos += 0.08 + 0.1
        fig.text(pos, 0.95, "StdErr:", ha="center", va="bottom", size="medium")
        fig.text(pos + 0.08, 0.95, str(round(std_err, 5)), ha="center", va="bottom", size="medium",color="red")

        # Show the plot
        plt.show()

    @staticmethod
    def test():
        # Bar chart example
        x1 = np.arange(70) #np.array([20, 35, 30, 35, 27, 33, 30, 56, 20])
        x2 = np.arange(70) #np.array([25, 32, 34, 20, 25, 33, 30, 56, 20])
        Plot.barchart(x1, x2)
        
        # Regression plot example
        n = 50
        X1 = np.random.randn(n)
        X2 = X1 * np.random.randn(n)
        Plot.regression(X1, X2)




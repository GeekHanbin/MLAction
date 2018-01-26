import matplotlib
import matplotlib.pyplot as plt


# y=x的一条直线
def linedisplay():
    plt.plot(range(5),linestyle = '--',linewidth = 3,color = 'red')
    plt.xlabel('my x label')
    plt.ylabel('my y label')
    plt.show()

linedisplay()
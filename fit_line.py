#Linear solver
def my_linfit(x,y):
    a = (sum(y*x)/sum(x*x))-(sum(x)*sum(y+(sum(x*y)/sum(x*x))))/(sum(x)-sum(x*x))
    b = (sum(y-(sum(x*y)/sum(x*x)))/((sum(x)/sum(x*x))-1))

    return a,b

#Main
import matplotlib.pyplot as mpl
import numpy as np

x = np.random.uniform(-2,5,10)
y = np.random.uniform(0,3,10)
a,b = my_linfit(x,y)
mpl.plot(x,y,'kx')
xp = np.arange(-2,5,0.1)
mpl.plot(xp,a*xp+b, 'r-')
print(f"My_fit:_a={b}_and_b={b}")
mpl.show()
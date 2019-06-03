import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sympy as S
file = 'sample_submission.csv'
data = np.loadtxt(file, delimiter = ',', skiprows = 1, usecols=[0,1])
x = np.loadtxt(file, delimiter= ',' ,skiprows = 1, usecols= [0])
y = np.loadtxt(file, delimiter= ',' ,skiprows = 1, usecols= [1])

X= S.symbols('x')
Y= S.symbols('y')





def polyfit2(x,y,n):

    def inv(A):
        return np.linalg.inv(A)
    def trans(A):
        return A.getT()
    def oneMat(xl,n):
        return np.ones((xl,n),dtype=int)
    def prod(A,B):
        return np.dot(A,B)

    xlen = len(x)
    ylen = len(y)
    one = np.ones((xlen,n+1),dtype=int)
    c1=one[:,[1]]
    xT=np.matrix(x)
    yT=np.matrix(y)
    c2=xT.getT()
    c3=np.power(c2,2)
    A=np.hstack([c1,c2,c3])
    print(A)
    return prod(prod(inv(prod(trans(A),A)),trans(A)),trans(yT))

print(polyfit2(x,y,2))

Y = 1.4785 + 5.4032 * x - 2.8146 * x**2
f = S.lambdify(X,Y,'math')
yen = f(x)
print(yen)
yout= yen.astype(list)

plt.scatter(x,y, color = 'red')
plt.plot(x,yout , color='green')
plt.show()

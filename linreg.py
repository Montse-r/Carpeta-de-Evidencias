import pandas as pd

x = [1.0, 1.6, 3.4, 4.0, 5.2]
y = [1.2, 2.0, 2.4, 3.5, 3.5]

def lin_reg(a,b):
    xf= pd.array(a)
    yf= pd.array(b)

    def suma(a):
        sum = 0
        for i in range(len(a)):
            sum = a[i] + sum
        return sum
    xy = xf * yf
    x2 = xf*xf
    n = len(xf)

    a = ((n * suma(xy)) - (suma(xf) * suma(yf))) / ((n * suma(x2)) - (suma(xf) * suma(xf)))
    b = ((suma(yf) * suma(x2)) - (suma(xy) * suma(xf))) / ((n * suma(x2)) - (suma(xf) * suma(xf)))

    return a,b

print(lin_reg(x,y))

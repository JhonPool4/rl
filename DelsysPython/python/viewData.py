import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

tb = pd.read_csv("data.csv")

def getEMG(n):
    return tb.iloc[:,n-1]

def getACC(n):
    Axx = tb.iloc[:,16 + (n-1)*3 + 0]
    Ayy = tb.iloc[:,16 + (n-1)*3 + 1]
    Azz = tb.iloc[:,16 + (n-1)*3 + 2]
    data = np.array([Axx,Ayy,Azz]) - np.array([Axx[0],Ayy[0],Azz[0]]).reshape((3,1))
    return data.T

# Axx,Ayy,Azz = [*getACC(9)]

plt.plot(getACC(9))

plt.show()



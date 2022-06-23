import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("data.csv")


plt.plot(df['acc_16_x'])
plt.plot(df['acc_16_y'])
plt.plot(df['acc_16_z'])

plt.show()



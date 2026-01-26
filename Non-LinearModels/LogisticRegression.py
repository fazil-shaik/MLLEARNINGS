import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)


X = np.linspace(0,10,10)
print(X)
y_prob = 1/(1+np.exp(-(X-10)))
y = (y_prob>0.5).astype(int)
noise = np.random.normal(0,0.5,size=10)
y_prob_noisy = 1 / (1 + np.exp(-(X - 5))) + noise
y = (y_prob_noisy > 0.5).astype(int)

# plt.scatter(X,y,label='Actual data')
plt.plot(X, y_prob_noisy)
plt.xlabel("Study Hours")
plt.ylabel("Probability of Pass")
plt.show()


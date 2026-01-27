import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n = 300

num_links = np.random.poisson(2,n)
email_length = np.random.randint(50,2000,n)

z = 0.8*num_links + 0.001*email_length - 3
prob = 1 / (1 + np.exp(-z))

y = (prob > 0.5).astype(int)

X = np.column_stack((num_links, email_length))

# plt.scatter(X,y,label='Actual data')

plt.plot(X,prob,label='Predicted data')

plt.xlabel("Total links")
plt.ylabel("Email data")
plt.legend()
plt.show()

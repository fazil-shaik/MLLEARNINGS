import numpy as np

def softmax(x):
    x = x-np.max(x)
    return np.exp(x)/sum(np.exp(x))

res = np.array([-3.2,0.6,-0.9,1.1,256])

print(softmax(res))




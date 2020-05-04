import numpy as np
from time import time

def get_time(fun):
    s = time()
    fun()
    return time() - s

N = 10000
print(np.sum([get_time(lambda: set()) for _ in range(N)]))
print(np.sum([get_time(lambda: {*[]}) for _ in range(N)]))

a = list(np.random.randint(200, size=(100)))

print(np.sum([get_time(lambda: set(a)) for _ in range(N)]))
print(np.sum([get_time(lambda: {*a}) for _ in range(N)]))
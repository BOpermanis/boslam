import numpy as np
from queue import Queue
from threading import Thread
import matplotlib.pyplot as plt
import numpy as np
from time import sleep

def gen(q1):
    while True:
        a = np.random.uniform()
        q1.put(a)
        sleep(0.2)

def show(q1):
    plt.axis([0, 10, 0, 1])
    vals = []
    while True:
        a = q1.get()
        vals.append(a)
        if len(vals) > 100:
            vals.pop(0)
        plt.plot(vals)
        plt.show()
        print(a)

q = Queue(maxsize=100)
threads = [
    Thread(target=gen, args=(q,)),
    Thread(target=show, args=(q,))
]

for t in threads:
    t.start()



for t in threads:
    t.join()


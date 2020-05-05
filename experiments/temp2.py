# from multiprocessing import Process, Manager, Queue, Value
from time import sleep
from threading import Thread, Lock


# class Cl:
#     def __init__(self):
#         self.t = 0
#
#     def t(self, a):
#         return a + 1
#
# cl = Cl()
# print(cl.t)
# print(cl.t(3))
#
# exit()
# def f(d, l):
#     d[1] = '1'
#     d['2'] = 2
#     d[0.25] = None
#     l.reverse()
#
# if __name__ == '__main__':
#     manager = Manager()
#
#     d = manager.dict()
#     l = manager.list(range(10))
#
#     p = Process(target=f, args=(d, l))
#     p.start()
#     p.join()
#
#     print(d)

class Luck:
    def __init__(self):
        self.lock = Lock()
        self.flag_locked = False

    def __enter__(self, *args):
        self.acquire(*args)

    def __exit__(self, *args):
        self.release(*args)

    def acquire(self, *args):
        if not self.flag_locked:
            self.lock.__enter__(*args)
            self.flag_locked = True

    def release(self, *args):
        if self.flag_locked:
            self.lock.__exit__(*args)
            self.flag_locked = False


# lock = Lock()
# lock1 = Lock()

lock = Luck()
lock1 = Luck()

def show(ob):
    with lock and lock1:
        sleep(2)
        print("show: ", ob)

def change(ob):
    sleep(1)
    with lock:
        with lock1:
            with lock:
                ob.append(123)
                print("change: ", ob)

ob = []
# ob = Value([])
threads = [
    Thread(target=show, args=(ob,)),
    Thread(target=change, args=(ob,))
]

for t in threads:
    t.daemon = True
    t.start()

for t in threads:
    t.join()

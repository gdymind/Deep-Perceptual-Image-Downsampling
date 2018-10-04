import time

class Timer():
    def __init__(self):
        self.acc = 0 # total ticks
        self.tic()

    def tic(self): # set start time
        self.t0 = time.time()

    def toc(self): # get total time from the tick moment
        return time.time() - self.t0

    def hold(self): # add toc
        self.acc += self.toc()

    def release(self): # reset current acc and return it
        res = self.acc
        self.acc = 0

        return res

    def reset(self): # reset acc
        self.acc = 0
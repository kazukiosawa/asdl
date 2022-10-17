from logging import raiseExceptions
import math,sys

class IntervalScheduler():
    def __init__(self,interval:int, T_max:int, warmup_ratio:float, inverse=False):
        self.update_times=T_max//interval
        self.T_max = T_max
        self.warmup=int(self.update_times*warmup_ratio)
        self.update_step=self.decide_update_step()

        if inverse:
            self.update_step.append(True)
            self.update_step = list(reversed(self.update_step[1:]))
        
        if self.update_times != self.update_step.count(True):
            print(self.update_times)
            print(self.update_step.count(True))
            print('Error: interval assignment failed', file=sys.stderr)
            sys.exit(1)
        
    def updateTF(self):
        return self.update_step.pop(0)

    def decide_update_step(self):
        NotImplemented

class StepIntervalScheduler(IntervalScheduler):
    def __init__(self,interval,T_max,warmup_ratio,inverse=False):
        super().__init__(interval,T_max,warmup_ratio,inverse=inverse)

    def decide_update_step(self):
        update_step = [False] * self.T_max
        update_step[:self.warmup] = [True] * self.warmup

        if self.update_times-self.warmup!=0:
            interval = math.ceil((self.T_max-self.warmup)/(self.update_times-self.warmup))
        else:
            return update_step
        update_step[self.warmup:] = [x%interval==0 for x in range(self.T_max-self.warmup)]

        for i in range(self.T_max-self.warmup):
            if update_step.count(True) < self.update_times:
                update_step[self.warmup+i]=True
            else:
                break
        
        return update_step

class LinearIntervalScheduler(IntervalScheduler):
    def __init__(self,interval,T_max,warmup_ratio,inverse=False):
        super().__init__(interval,T_max,warmup_ratio,inverse=inverse)

    def decide_update_step(self):
        update_step = [False] * self.T_max
        update_step[:self.warmup] = [True] * self.warmup

        if self.update_times-self.warmup==0:
            return update_step

        a = 2*(self.T_max-self.warmup)/((self.update_times-self.warmup)*(self.update_times-self.warmup+1))

        for i in range(self.update_times-self.warmup):
            if self.warmup + int(a*i*(i+1)/2) < self.T_max:
                update_step[self.warmup + int(a*i*(i+1)/2)]=True

        for i in range(self.T_max-self.warmup):
            if update_step.count(True) < self.update_times:
                update_step[self.warmup+i]=True
            else:
                break
        
        return update_step


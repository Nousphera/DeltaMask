"""Beta MAP update (paper Eq.3), using unweighted binary client observations."""
import math
import numpy as np

class BayesianMask:
    def __init__(self,dimension,participation=1.):
        if not 0<participation<=1 or dimension<1: raise ValueError('Invalid aggregation configuration')
        self.period=max(1,math.floor(1/participation))
        self.alpha=np.ones(dimension,dtype=np.float64)
        self.beta=np.ones(dimension,dtype=np.float64)
        self.last_round=0

    def update(self,masks,round_id):
        if round_id!=self.last_round+1 or not masks: raise ValueError('Expected next round and nonempty clients')
        masks=np.asarray(masks)
        if masks.ndim!=2 or masks.shape[1]!=len(self.alpha) or not np.isin(masks,[0,1]).all(): raise ValueError('Expected binary masks')
        if round_id%self.period==0:
            self.alpha.fill(1);self.beta.fill(1)
        self.alpha+=masks.sum(axis=0)
        self.beta+=len(masks)-masks.sum(axis=0)
        self.last_round=round_id
        return np.clip((self.alpha-1)/(self.alpha+self.beta-2),1e-7,1-1e-7)

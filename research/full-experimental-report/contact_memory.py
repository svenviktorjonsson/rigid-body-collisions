"""Small elastic-contact matrix core; rigid bodies, contact-local history.

This is a verified sticking/bilateral linear branch, not a whole collision law.
Opening, evolving frames, plastic yielding and Coulomb branches must be supplied
before native or empirical use. K,C are constitutive inputs, not learned material
values. eta accumulates projected contact motion; angular channels are ell-scaled.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_factor,cho_solve


def symmetric(matrix,name,positive):
    a=np.asarray(matrix,float)
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or not np.all(np.isfinite(a)):
        raise ValueError(name+' must be finite and symmetric')
    if np.linalg.norm(a-a.T,ord=np.inf)>1e-12*max(1,np.linalg.norm(a,ord=np.inf)):
        raise ValueError(name+' must be finite and symmetric')
    eigen=np.linalg.eigvalsh(a)
    if (positive and eigen[0]<=0) or (not positive and eigen[0]<-1e-12*max(1,np.linalg.norm(a,2))):
        raise ValueError(name+' has an invalid sign')
    return .5*(a+a.T)


@dataclass
class ElasticMemory:
    mobility: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray
    timestep: float

    def __post_init__(self):
        self.mobility=symmetric(self.mobility,'mobility',False)
        self.stiffness=symmetric(self.stiffness,'stiffness',True)
        self.damping=symmetric(self.damping,'damping',False)
        if self.stiffness.shape!=self.mobility.shape or self.damping.shape!=self.mobility.shape:
            raise ValueError('equal matrix dimensions required')
        h=self.timestep
        if not np.isfinite(h) or h<=0:raise ValueError('positive finite timestep required')
        Q=.5*h*h*self.stiffness+h*self.damping
        self.compliance=2*np.linalg.solve(Q,np.eye(Q.shape[0]))
        self.factor=cho_factor(self.mobility+self.compliance,lower=True)
        self.history_force=self.compliance@(h*self.stiffness)

    def step(self,motion,history):
        u,eta=np.asarray(motion,float),np.asarray(history,float)
        shape=(self.mobility.shape[0],)
        if u.shape!=shape or eta.shape!=shape or not np.all(np.isfinite(np.r_[u,eta])):
            raise ValueError('finite motion/history of correct dimension required')
        impulse=cho_solve(self.factor,-2*u-self.history_force@eta)
        outgoing=u+self.mobility@impulse
        midpoint=.5*(u+outgoing)
        updated=eta+self.timestep*midpoint
        loss=self.timestep*(midpoint@self.damping@midpoint)
        return outgoing,updated,impulse,float(loss)

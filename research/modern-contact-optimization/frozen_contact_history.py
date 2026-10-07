"""Passive contact-local spring/slider memory, one to three declared modes.

Rigid-body mobility includes lever moments AND independent angular impulses.
Mode directions must be supplied consistently by the caller; this primitive does
not resolve the general moving t/s frame or jointly solve normal impact/groups.
Opening transfers stored energy into a separately reported internal-mode ledger,
instead of silently erasing it or applying friction across a detached contact.
"""
from dataclasses import dataclass
import itertools
import math
import numpy as np


@dataclass(frozen=True)
class HistoryResult:
    motion: np.ndarray
    history: np.ndarray
    impulse: np.ndarray
    sliding: bool
    stored_energy: float
    plastic_loss: float
    released_mode_energy: float
    energy_residual: float


class ContactHistory:
    """Implicit midpoint elasticity and a convex dynamic-capacity return map.

    Sticking first tests the static capacity. If it yields, a unique constrained
    solve uses dynamic capacities. With at most three modes there are at most 27
    active sets; each uses an already factored one/two/three-dimensional matrix.
    There are no contact-site loops, fitted corrections or deformation meshes.
    """
    def __init__(self,mobility,stiffness,timestep):
        G=np.array(mobility,dtype=float,copy=True);K=np.array(stiffness,dtype=float,copy=True)
        if G.ndim!=2 or G.shape[0]!=G.shape[1] or not 1<=len(G)<=3 or K.shape!=(len(G),):
            raise ValueError('One to three modes, square mobility and diagonal stiffness required')
        if not np.isfinite(G).all() or not np.isfinite(K).all() or np.any(K<=0) or not math.isfinite(timestep) or timestep<=0:
            raise ValueError('Finite mobility, positive stiffness/timestep required')
        scale=max(np.max(np.abs(G)),1e-300)
        if np.max(np.abs(G-G.T))>1e-12*scale or np.linalg.eigvalsh(G)[0]<-1e-12*scale:
            raise ValueError('Symmetric positive-semidefinite body mobility required')
        self.G=.5*(G+G.T);self.K=K;self.h=timestep;self.n=len(K)
        self.A=.5*self.h*self.G+np.diag(2/(self.h*self.K))
        self.factors={}
        for mask in itertools.product([False,True],repeat=self.n):
            free=np.flatnonzero(mask)
            if len(free):self.factors[tuple(free)]=np.linalg.cholesky(self.A[np.ix_(free,free)])

    def _solve(self,free,rhs):
        factor=self.factors[tuple(free)]
        return np.linalg.solve(factor.T,np.linalg.solve(factor,rhs))

    def step(self,motion,history,static_capacity,dynamic_capacity,*,active=True):
        u,eta,cs,cd=[np.asarray(x,dtype=float) for x in [motion,history,static_capacity,dynamic_capacity]]
        if any(x.shape!=(self.n,) or not np.isfinite(x).all() for x in [u,eta,cs,cd]) or np.any(cd<0) or np.any(cs<cd):
            raise ValueError('Finite mode arrays and 0 <= dynamic <= static capacities required')
        if type(active)!=bool:raise ValueError('Explicit contact activity required')
        initial=.5*float(self.K@(eta*eta))
        if not active:
            # Released energy remains a physical state for the caller to retain
            # or relax; it is not counted as measured dissipation or returned
            # kinetic energy without a body/mode coupling law.
            return HistoryResult(u.copy(),np.zeros(self.n),np.zeros(self.n),False,0.,0.,initial,0.)
        b=self.h*u+2*eta
        all_modes=np.arange(self.n);j=self._solve(all_modes,-b)
        sticking=bool(np.all(np.abs(j)<=cs))
        if not sticking:
            found=False
            # 2 means fixed zero capacity; no sign condition is meaningful on
            # a singleton feasible interval. Other statuses are lower/free/upper.
            options=[[-1,0,1] if cap>0 else [2] for cap in cd]
            for state in itertools.product(*options):
                state=np.array(state);free=np.flatnonzero(state==0);fixed=np.flatnonzero(state!=0)
                j=np.zeros(self.n);j[fixed]=np.where(state[fixed]==2,0,state[fixed]*cd[fixed])
                if len(free):j[free]=self._solve(free,-b[free]-self.A[np.ix_(free,fixed)]@j[fixed])
                gradient=self.A@j+b
                tolerance=128*np.finfo(float).eps*max(np.max(np.abs(b)),np.max(np.abs(self.A@j)),1e-300)
                if np.any(np.abs(j[free])>cd[free]):continue
                if np.any(gradient[state==-1]<-tolerance) or np.any(gradient[state==1]>tolerance):continue
                found=True;break
            if not found:raise RuntimeError('No admissible spring/slider active set; no fallback')
        updated=-2*j/(self.h*self.K)-eta
        outgoing=u+self.G@j;midpoint=.5*(u+outgoing)
        plastic=self.h*midpoint-(updated-eta)
        loss=-float(j@plastic)/self.h
        final=.5*float(self.K@(updated*updated))
        work=float(j@midpoint);residual=work+final-initial+loss
        scale=max(abs(work),initial,final,abs(loss),1e-300)
        if not np.isfinite(np.r_[outgoing,updated,j,loss,final,residual]).all() or loss < -1e-11*scale or abs(residual)>2e-10*scale:
            raise RuntimeError('Contact-history energy gate rejected update')
        return HistoryResult(outgoing,updated,j,not sticking,final,loss,0.,residual)

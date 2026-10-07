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
    # Numerical hint only: lower/free/upper/zero-capacity = -1/0/1/2.
    # Caller owns contact identity; the law object retains no per-contact state.
    active_set: tuple | None = None


class ContactHistory:
    """Implicit midpoint elasticity and a convex dynamic-capacity return map.

    Sticking first tests the static capacity. If it yields, a unique constrained
    solve uses dynamic capacities. With at most three modes there are at most 27
    active sets; each uses an already factored one/two/three-dimensional matrix.
    A caller-owned active set can be tried first, but must pass the current KKT
    conditions. It never bypasses static capacity, opening or energy checks.
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
        self._triangles={}
        for mask in itertools.product([False,True],repeat=self.n):
            free=np.flatnonzero(mask)
            if len(free):
                factor=np.linalg.cholesky(self.A[np.ix_(free,free)])
                self.factors[tuple(free)]=factor
                self._triangles[tuple(free)]=tuple(float(factor[i,j]) for i in range(len(free)) for j in range(i+1))
        self._all_modes=np.arange(self.n)
        self._sets={};self._sets_by_zero={}
        for zero in itertools.product([False,True],repeat=self.n):
            entries=[]
            for key in itertools.product(*[[2] if z else [-1,0,1] for z in zero]):
                state=np.array(key);free=np.flatnonzero(state==0);fixed=np.flatnonzero(state!=0)
                entry=(key,state,free,fixed,self.A[np.ix_(free,fixed)])
                self._sets[key]=entry;entries.append(entry)
            self._sets_by_zero[zero]=tuple(entries)
        # Prepared coefficients cannot be changed behind their cached factors.
        for values in [self.G,self.K,self.A,*self.factors.values()]:values.flags.writeable=False

    def _solve(self,free,rhs):
        # Small triangular substitutions, without a generic LAPACK dispatch or
        # forming an inverse. Cached scalar factors preserve coupled mobility.
        L=self._triangles[tuple(free)];n=len(free)
        y0=float(rhs[0])/L[0]
        if n==1:return np.array([y0/L[0]])
        y1=(float(rhs[1])-L[1]*y0)/L[2]
        if n==2:
            x1=y1/L[2];return np.array([(y0-L[1]*x1)/L[0],x1])
        y2=(float(rhs[2])-L[3]*y0-L[4]*y1)/L[5]
        x2=y2/L[5];x1=(y1-L[4]*x2)/L[2]
        return np.array([(y0-L[1]*x1-L[3]*x2)/L[0],x1,x2])

    def _candidate(self,entry,b,capacity):
        _,state,free,fixed,coupling=entry
        j=np.zeros(self.n);j[fixed]=np.where(state[fixed]==2,0,state[fixed]*capacity[fixed])
        if len(free):j[free]=self._solve(free,-b[free]-coupling@j[fixed])
        response=self.A@j;gradient=response+b
        tolerance=128*np.finfo(float).eps*max(np.max(np.abs(b)),np.max(np.abs(response)),1e-300)
        if np.any(np.abs(j[free])>capacity[free]):return None
        if np.any(gradient[state==-1]<-tolerance) or np.any(gradient[state==1]>tolerance):return None
        return j

    def step(self,motion,history,static_capacity,dynamic_capacity,*,active=True,active_set=None):
        u,eta,cs,cd=[np.asarray(x,dtype=float) for x in [motion,history,static_capacity,dynamic_capacity]]
        if any(x.shape!=(self.n,) or not np.isfinite(x).all() for x in [u,eta,cs,cd]) or np.any(cd<0) or np.any(cs<cd):
            raise ValueError('Finite mode arrays and 0 <= dynamic <= static capacities required')
        if type(active)!=bool:raise ValueError('Explicit contact activity required')
        if active_set is not None:
            if not isinstance(active_set,tuple) or len(active_set)!=self.n or any(type(x)!=int or x not in (-1,0,1,2) for x in active_set):
                raise ValueError('Active-set hint must be a tuple of -1/0/1/2 mode statuses')
        initial=.5*float(self.K@(eta*eta))
        if not active:
            # Released energy remains a physical state for the caller to retain
            # or relax; it is not counted as measured dissipation or returned
            # kinetic energy without a body/mode coupling law.
            return HistoryResult(u.copy(),np.zeros(self.n),np.zeros(self.n),False,0.,0.,initial,0.)
        b=self.h*u+2*eta
        j=self._solve(self._all_modes,-b)
        sticking=bool(np.all(np.abs(j)<=cs))
        accepted_set=tuple(0 for _ in range(self.n))
        if not sticking:
            # 2 means fixed zero capacity; no sign condition is meaningful on
            # a singleton feasible interval. Other statuses are lower/free/upper.
            zero=tuple(bool(cap==0) for cap in cd)
            hint=None
            if active_set is not None:
                hint=tuple(2 if zero[i] else (0 if s==2 else s) for i,s in enumerate(active_set))
                j=self._candidate(self._sets[hint],b,cd)
                if j is not None:accepted_set=hint
            else:j=None
            if j is None:
                for entry in self._sets_by_zero[zero]:
                    if entry[0]==hint:continue
                    j=self._candidate(entry,b,cd)
                    if j is not None:accepted_set=entry[0];break
            if j is None:raise RuntimeError('No admissible spring/slider active set; no fallback')
        updated=-2*j/(self.h*self.K)-eta
        outgoing=u+self.G@j;midpoint=.5*(u+outgoing)
        plastic=self.h*midpoint-(updated-eta)
        loss=-float(j@plastic)/self.h
        final=.5*float(self.K@(updated*updated))
        work=float(j@midpoint);residual=work+final-initial+loss
        scale=max(abs(work),initial,final,abs(loss),1e-300)
        if not np.isfinite(np.r_[outgoing,updated,j,loss,final,residual]).all() or loss < -1e-11*scale or abs(residual)>2e-10*scale:
            raise RuntimeError('Contact-history energy gate rejected update')
        return HistoryResult(outgoing,updated,j,not sticking,final,loss,0.,residual,accepted_set)

"""Contact-graph algebra and a proposed passive endpoint comparator.

The target projection is not an exact Coulomb or compliant material law.
"""
import numpy as np
from scipy.optimize import minimize


def assemble_planar(centers, masses, inertias, contacts, ell=1.0):
    """Contacts contain (body1, body2, world_point, normal_2_to_1).

    Every point is a separate edge, even when two edges share a body pair.
    Return scaled inverse body mass H^-1, contact map G, mobility K.
    Positive infinity denotes a prescribed degree of freedom (zero inverse
    mass/inertia); its supplied velocity still enters the relative contact map.
    """
    centers=np.asarray(centers,dtype=float)
    masses=np.asarray(masses,dtype=float);inertias=np.asarray(inertias,dtype=float)
    if not np.isfinite(ell) or ell<=0 or not np.all(np.isfinite(centers)) or np.any(np.isnan(masses)) or np.any(np.isnan(inertias)) or np.any(masses<=0) or np.any(inertias<=0):
        raise ValueError('Positive reference length, masses, and central inertias required')
    inverse=np.diag(np.column_stack((1/masses,1/masses,ell**2/inertias)).ravel())
    G=np.zeros((3*len(contacts),3*len(centers)))
    for edge,(one,two,point,normal) in enumerate(contacts):
        normal=np.asarray(normal,dtype=float)
        normal=normal/np.linalg.norm(normal)
        tangent=np.array([-normal[1],normal[0]])
        Q=np.array([[normal[0],tangent[0],0],[normal[1],tangent[1],0],[0,0,1.]])
        for body,sign in ((one,1),(two,-1)):
            r=np.asarray(point,dtype=float)-centers[body]
            T=np.array([[1,0,-r[1]/ell],[0,1,r[0]/ell],[0,0,1.]])
            G[3*edge:3*edge+3,3*body:3*body+3]+=sign*Q.T@T
    return inverse,G,G@inverse@G.T


def inelastic_normal_solve(inverse,G,velocity):
    """Global frictionless, zero-restitution unilateral projection.

    Handles redundant rows without inverting the contact mobility. Does not
    claim to solve arbitrary heterogeneous exact restitution constraints.
    """
    normal=G[::3];K=normal@inverse@normal.T;u=normal@velocity
    result=minimize(lambda p:.5*p@K@p+u@p,np.zeros(len(u)),
                    jac=lambda p:K@p+u,method='L-BFGS-B',
                    bounds=[(0,None)]*len(u),options={'ftol':0,'gtol':1e-12,'maxiter':5000,'maxcor':20,'maxls':100})
    p=np.maximum(result.x,0.)
    # Objective stagnation is not a KKT certificate, especially in long chains.
    # Polish the candidate's positive active set by solving its stationarity
    # residual. Least-squares corrections retain null-space impulse components
    # for redundant constraints rather than inverting singular mobilities.
    for _ in range(8):
        gradient=K@p+u
        active=(p>1e-10)|(gradient < -1e-10)
        if not np.any(active):break
        correction=np.linalg.lstsq(K[np.ix_(active,active)],-gradient[active],rcond=1e-12)[0]
        trial=p.copy();trial[active]+=correction
        if np.min(trial)<-1e-9:break  # reject below if a verified active set cannot be obtained
        p=np.maximum(trial,0.)
        if np.min(K@p+u)>=-1e-10 and np.max(np.abs(p*(K@p+u)))<=1e-9:break
    post=velocity+inverse@normal.T@p
    w=normal@post
    if np.min(w)<-1e-7 or np.max(np.abs(p*w))>1e-7:
        raise RuntimeError('Normal complementarity residual exceeds tolerance')
    full=np.zeros(G.shape[0]);full[::3]=p
    return post,full,{'normal_velocity':w.tolist(),'complementarity_residual':float(np.max(np.abs(p*w)))}


def passive_target_projection(K_scaled,u_scaled,restitution,mu_static,mu_dynamic,
                              kappa_static,kappa_dynamic,ell=1.0):
    """Single-pair convex passive closest-target comparator in 2D.

    Exact normal restitution; bounded tangential/rolling impulse; target-error
    minimization in the inverse-mobility metric. Dynamic capacities are chosen
    when the unconstrained target exceeds static capacity. Capacity selection
    is a phenomenological heuristic, not slip/stick complementarity.
    """
    scale=np.diag([1.,1.,ell]);inverse_scale=np.diag([1.,1.,1/ell])
    K=inverse_scale@K_scaled@inverse_scale
    u=inverse_scale@u_scaled
    if u[0]>=0:
        raise ValueError('Comparator requires an approaching single normal contact')
    e=np.asarray(restitution,dtype=float)
    if np.any(e<0) or np.any(e>1):raise ValueError('Restitution targets must be in [0,1]')
    if not 0<=mu_dynamic<=mu_static or not 0<=kappa_dynamic<=kappa_static:
        raise ValueError('Require nonnegative dynamic capacity <= static capacity')
    metric=np.linalg.inv(K)
    target=-e*u
    candidate=np.linalg.solve(K,target-u)
    normal_capacity=max(candidate[0],0)
    mu=mu_dynamic if abs(candidate[1])>mu_static*normal_capacity+1e-12 else mu_static
    kappa=kappa_dynamic if abs(candidate[2])>kappa_static*normal_capacity+1e-12 else kappa_static
    bound=np.array([[mu,1,0],[mu,-1,0],[kappa,0,1],[kappa,0,-1.]])
    start=np.array([-(1+e[0])*u[0]/K[0,0],0.,0.])
    def energy(p):return u@p+.5*p@K@p
    def objective(p):
        residual=u+K@p-target
        return .5*residual@metric@residual
    constraints=[{'type':'eq','fun':lambda p:(u+K@p-target)[0],'jac':lambda p:K[0]},
                 {'type':'ineq','fun':lambda p:bound@p,'jac':lambda p:bound},
                 {'type':'ineq','fun':lambda p:-energy(p),'jac':lambda p:-(u+K@p)}]
    result=minimize(objective,start,jac=lambda p:u+K@p-target,
                    method='SLSQP',bounds=[(0,None),(None,None),(None,None)],
                    constraints=constraints,options={'ftol':1e-12,'maxiter':1000})
    p=result.x
    if abs((u+K@p-target)[0])>1e-7 or np.min(bound@p)<-1e-7 or energy(p)>1e-7:
        raise RuntimeError('Passive projection did not meet its stated constraints')
    return inverse_scale@p,{'success':bool(result.success),'message':result.message,
                            'energy_change':float(energy(p)),
                            'chosen_mu_capacity':float(mu),'chosen_kappa_capacity':float(kappa),
                            'target_residual':(u+K@p-target).tolist(),
                            'post_velocity':(scale@(u+K@p)).tolist()}

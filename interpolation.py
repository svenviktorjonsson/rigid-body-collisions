import numpy as np
import matplotlib.pyplot as plt

def new_point(x,p0,p1,p2,p3):
    v=p2-p0
    u=p1-p3
    w=p2-p1
    w2=np.dot(w,w)
    wu=np.dot(w,u)
    v2=np.dot(v,v)
    vu=np.dot(v,u)
    u2=np.dot(u,u)
    A=w2*u2-wu**2
    B=v2*u2-vu**2
    t=np.sqrt(A/B)
    mu=(p1+p2)/2
    px=p1+t*v
    angle=np.arccos(vu/np.sqrt(u2*v2))
    return mu*x+(1-x)*px,angle*180/np.pi



fig, ax = plt.subplots(figsize=(8,4),dpi=200)
fig.patch.set_facecolor('black')
ax.set_facecolor('black')



# Remove all spines, ticks, and labels
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)
ax.axis("off")
ax.set_aspect("equal")

theta=np.arange(3600)*np.pi/1800
r=np.sqrt(2)

cx=r*np.cos(theta)
cy=r*np.sin(theta)

ax.plot(cx,cy,'w--',lw=0.5)
lams=[0.58578645,0.52002325,0.5050898202,0.50382174,0.503789734]
angles=[]
ps=[np.r_[-1,-1],np.r_[-1,1],np.r_[1,1],np.r_[1,-1]]
for n,lam in enumerate(lams):
    new_points=[]
    for shift in range(len(ps)):
        psr=np.roll(ps,1-shift,axis=0)[:4]
        pn,angle=new_point(lam,*psr)
        new_points.append(pn)
    for k,p in enumerate(new_points):
        ps.insert(2*k+1,p)
    angles.append(angle)
    px,py=zip(*ps)
    ax.plot(px+px[:1],py+py[:1], lw=0.5,label=f"iteration={n:.8f}")
    
fig.legend()

fig2,ax2=plt.subplots()
ax2.plot(angles,lams,marker=".")
ax2.set_xlabel("Angle (deg)")
ax2.set_ylabel("$\lambda$")
ax2.grid(True)
plt.show()
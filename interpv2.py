import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect, curve_fit

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
    val = vu/np.sqrt(u2*v2)
    val = np.clip(val, -1.0, 1.0)
    angle=np.arccos(val)
    return mu*x+(1-x)*px, angle  # angle in radians

fig, ax = plt.subplots(figsize=(8,4),dpi=200)
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
ax.set_xlim(-1.5,1.5)
ax.set_ylim(-1.5,1.5)
ax.axis("off")
ax.set_aspect("equal")

r_target = 1
theta=np.linspace(0,2*np.pi,3600)
cx=r_target*np.cos(theta)
cy=r_target*np.sin(theta)
ax.plot(cx,cy,'w--',lw=0.5)

angles=[]
lam_values=[]
for num_points in [3,4,5,7,9,11,13,15]:
    angles_for_ps = np.linspace(0,2*np.pi,num_points,endpoint=False)
    ps = [np.array([np.cos(a), np.sin(a)]) for a in angles_for_ps]

    def radius_diff(lam, ps):
        new_points=[]
        for shift in range(len(ps)):
            psr=np.roll(ps,1-shift,axis=0)[:4]
            if len(psr)==3:
                psr=(*psr,psr[0])
            pn,_=new_point(lam,*psr)
            new_points.append(pn)
        test_ps=ps[:]
        for k,p in enumerate(new_points):
            test_ps.insert(2*k+1,p)
        dist = np.linalg.norm(new_points[0])
        return dist - r_target

    for i in range(6):
        px,py=zip(*ps)
        ax.plot(px+px[:1],py+py[:1], lw=0.5, label=f"iteration={i}")
        lam = bisect(radius_diff, 0.3, 0.8, args=(ps,))
        new_points=[]
        for shift in range(len(ps)):
            psr=np.roll(ps,1-shift,axis=0)[:4]
            if len(psr)==3:
                psr=(*psr,psr[0])
            pn,angle=new_point(lam,*psr)
            new_points.append(pn)
        for k,p in enumerate(new_points):
            ps.insert(2*k+1,p)
        angles.append(angle)
        lam_values.append(lam)

fig2, ax2 = plt.subplots()
y = np.array(lam_values)

# Fix y0 = 0.5 and x0 = pi
x0_fixed = np.pi
y0_fixed = 0.5

def fit_func_fixed_x0_y0(x, a, p):
    return y0_fixed + a*(np.abs(x - x0_fixed)**p)

# Initial guesses for a and p
a_init = 0.03
p_init = 2.0

fit_x = np.linspace(min(angles), max(angles), 200)
initial_y = fit_func_fixed_x0_y0(fit_x, a_init, p_init)

ax2.plot(angles, y, "r.", label="Data")
ax2.plot(fit_x, initial_y, "g--", label=f"Initial guess (y0={y0_fixed:.3f}, x0=$\pi$, a={a_init:.3f}, p={p_init:.3f})")

popt, pcov = curve_fit(fit_func_fixed_x0_y0, angles, y, p0=[a_init, p_init])
a_fit, p_fit = popt

fit_y = fit_func_fixed_x0_y0(fit_x, a_fit, p_fit)
ax2.plot(fit_x, fit_y, "b-", label=f"Fit: y0={y0_fixed:.3f}, x0=$\pi$, a={a_fit:.5f}, p={p_fit:.5f}")

ax2.set_xlabel("Angle (rad)")
ax2.set_ylabel("$\lambda$")
ax2.grid(True)
ax2.legend()

# Show the equation in LaTeX style on the plot
equation_text = r'$\lambda(\theta) = 0.5 + a|\theta - \pi|^{p}$'
ax2.text(0.5, 0.5, equation_text, transform=ax2.transAxes, verticalalignment='top')

plt.show()

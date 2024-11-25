import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Iterable
import abc
from matplotlib.patches import Circle
import numpy as np
import time
import sys

def causal_independent_rows(data):
    flat_data = data.flatten()
    _, unique_indices = np.unique(flat_data, return_index=True)
    return np.unique(unique_indices // data.shape[1])


class Animation(abc.ABC):

    def __init__(self, dt, bg_color="black"):
        self.dt = dt
        self.fig = plt.figure(figsize=(6, 6), dpi=150)
        self.fig.set_facecolor(bg_color)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0,1)
        self.ax.set_ylim(0,1)
        self.ax.set_facecolor(bg_color)
        self.ax.axis("off")

    @abc.abstractmethod
    def update(self, time_index: int) -> Iterable:
        pass

    def run(self) -> None:
        self.animation = FuncAnimation(
            self.fig, self.update, frames = 1000, interval=self.dt * 1000, blit=True
        )
        plt.show()


class Simulation(Animation):

    def __init__(self,dt, e = 1, g = 9.82,**kwargs):
        super().__init__(dt,**kwargs)
        self.e = e
        self.g = g
        self.objects = np.zeros((100,11))
        self.graph_objects = {}
        self.end_index = 0
        self.cols = {"t":0,"x":slice(1,3),"v":slice(3,5),"r":5,"m":6,"xx":7,"xv":8,"vv":9,"rr":10}
        self.collisions_left = True
        self.update_times = []

    def get_all(self,col):
        return self.objects[:self.end_index,self.cols[col]]
    
    def overlaps(self, circle:dict, margin=0) -> np.ndarray[bool]:
        x = circle["position"][None,:]
        r = circle["radius"]
        xs = self.get_all("x")
        rs = self.get_all("r")
        return np.sum((x-xs)**2,axis=1)<=(r+rs+margin)**2
    
    def inside(self,circle:dict) -> tuple[bool,bool]:
        x,y = circle["position"]
        r = circle["radius"]
        return r<x<1-r, r<y<1-r

    def add_object(self,obj):
        if self.end_index>=len(self.objects):
            self.objects = np.pad(self.objects,[(0, len(self.objects)),(0,0)], constant_values=0)
        if obj["type"]=="circle":
            x = obj["position"]
            v = obj["velocity"]
            r = obj["radius"]
            m = obj["density"] * np.pi * r*r
            self.objects[self.end_index,:] = 0,*x,*v,r,m,np.sum(x*x),np.sum(x*v),np.sum(v*v),r*r
            circle = Circle(x,r,facecolor=obj["color"], edgecolor='none',antialiased=True)
            self.graph_objects[self.end_index] = circle
            self.ax.add_patch(circle)
            self.end_index+=1
        if obj["type"]=="line_segment":
            pass


    def recalculate_objects(self,time_index):
        self.collisions_left = True
        time_zero = time.perf_counter()
        dt = self.dt
        t = self.get_all("t")
        x = self.get_all("x")
        v = self.get_all("v")
        r = self.get_all("r")
        m = self.get_all("m")
            # r,m,xx,xv,vv,rr = self.objects[:self.end_index,5:].T
        while self.collisions_left:
            self.collisions_left = False
            i, j = np.triu_indices(len(x), k=1)

            x0 = x - v*t[:,None]

            A = np.sum((x0[i]-x0[j])*(v[i]-v[j]),axis=1)
            B = np.sum((v[i]-v[j])**2, axis=1)
            C = np.sum((x0[i]-x0[j])**2, axis=1)
            D = (r[i]+r[j])**2


            S = A**2-B*(C-D)
            filter1 = S>0
            if np.any(filter1):
                i = i[filter1]
                j = j[filter1]
                TC = (-A[filter1]-np.sqrt(S[filter1]))/B[filter1]
                filter2 = (TC>t[i])&(TC>t[j])&(TC<dt)
                if np.any(filter2):
                    TC = TC[filter2]


                    time_ordered = np.argsort(TC)

                    i = i[filter2][time_ordered]
                    j = j[filter2][time_ordered]
                    TC = TC[time_ordered]

                    causal_indices = causal_independent_rows(np.column_stack([i,j]))
                    i = i[causal_indices]
                    j = j[causal_indices]
                    TC = TC[causal_indices]
                    
                    x[i] += (TC-t[i])[:,None]*v[i]
                    x[j] += (TC-t[j])[:,None]*v[j]
                    t[i] = TC
                    t[j] = TC
                    E = (m[i]+m[j])*(r[i]+r[j])**2
                    F = (1+self.e)*(x[i]-x[j])*np.sum((x[i]-x[j])*(v[i]-v[j]),keepdims = True, axis=1)/E[:,None]
                    v[j] += F*m[i,None]
                    v[i] -= F*m[j,None]
                    print(TC)
                    if len(i)>1:
                        self.collisions_left = True

        # v[:,1] -= self.g*dt
        x[:] += v * (dt-t)[:,None]
        t[:] = 0

        self.update_times.append(time.perf_counter()-time_zero)




    def update(self,time_index):
        self.recalculate_objects(time_index)
        for index,row in enumerate(self.objects[:self.end_index]):
            if index not in self.graph_objects:
                continue
            self.graph_objects[index].set_center(row[1:3])
        # if time_index%100==99:
        #     print(f"Avg update time for {N} circles: {np.mean(self.update_times)}")
        #     sys.exit()
        return self.graph_objects.values()

if __name__ == "__main__":
    N = 50
    sim = Simulation(dt = 0.05, e = 1, g=5, bg_color="black")
    while sim.end_index<N:
        circle = dict(
            name=f"circle_{sim.end_index}",
            type="circle",
            position=np.random.random(2),  # Initial position
            velocity=(0.5-np.random.random(2)),  # Velocity vector
            radius=2/N+np.random.random()/N,  # Circle radius
            density=1,
            color=np.random.random(3),
        )
        if not np.any(sim.overlaps(circle,margin=0.005)) and all(sim.inside(circle)):
            sim.add_object(circle)
    width = 0.01
    for x,y in [(0.5,1-width+1e5),(0.5,width-1e5),(1-width+1e5,0.5),(width-1e5,0.5)]:
        circle = dict(
            name=f"circle_{sim.end_index}",
            type="circle",
            position=np.r_[x,y],  # Initial position
            velocity=np.r_[0,0],  # Velocity vector
            radius=1e5,  # Circle radius
            density=1e5,
            color="white",
        )
        sim.add_object(circle)
    sim.run()

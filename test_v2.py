import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Iterable
import abc
from matplotlib.patches import Circle
import numpy as np


class Animation(abc.ABC):

    def __init__(self, delta_t, bg_color="black"):
        self.delta_t = delta_t
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
            self.fig, self.update, frames = 1000, interval=self.delta_t * 1000, blit=True
        )
        plt.show()


class Simulation(Animation):

    def __init__(self, e, g, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.e = e
        self.g = g
        self.objects = []  # Stores physical objects with properties
        self.graph_objects = {}  # Maps names to matplotlib patches

    def update(self, time_index: int) -> Iterable:
        dt = self.delta_t
        moved_objs = set()
        for j,obj in enumerate(self.objects):
            # Update position based on velocity and delta_t
            p = obj["position"]
            v = obj["velocity"]
            r = obj["radius"]
            m = obj["density"]*np.pi*r**2
            p1 = p+v*dt
            for i in range(2):
                if p1[i]+r>1:
                    obj["position"][i] = 2*(1-r)-p[i]
                    obj["velocity"][i]*=-1

                if p1[i]-r<0:
                    obj["position"][i] = 2*r-p[i]
                    obj["velocity"][i]*=-1
            times = []
            if obj["name"] in moved_objs:
                continue
            for obj2 in self.objects[j+1:]:
                if obj2["name"] in moved_objs:
                    continue
                p2 = obj2["position"]
                v2 = obj2["velocity"]
                r2 = obj2["radius"]
                m2 = obj2["density"]*np.pi*r2**2
                D = np.sqrt(np.dot(p2-p, p2-p))
                V = np.sqrt(np.dot(v, v))
                V_2 = np.sqrt(np.dot(v2, v2))
                if D > (V+V_2)*dt + r + r2:
                    continue
                P2 = np.dot(p2-p, p2-p)
                DPV2 = np.dot(p2-p, v2-v)
                DV2 = np.dot(v-v2, v-v2)
                R2 = (r+r2)**2
                A = -DPV2/DV2
                S = A**2+(R2-P2)/DV2
                if S<=0:
                    continue
                tc = A-np.sqrt(S)
                if tc<0 or tc>dt:
                    continue
                pt = p+v*tc
                p2t = p2+v2*tc
                nnT = np.outer(p2t-pt,p2t-pt)/R2
                M = 1/m+1/m2
                J = -(1+self.e)*np.dot(nnT,v2-v)/M
                nv = v-J/m
                nv2 = v2+J/m2
                obj["velocity"] = nv
                obj2["velocity"] = nv2
                obj["position"] = pt-nv*(dt-tc)
                obj2["position"] = p2t-nv2*(dt-tc)
                moved_objs.add(obj["name"])
                moved_objs.add(obj2["name"])


        for j,obj in enumerate(self.objects):
            obj["velocity"][1] -= self.g * self.delta_t
            obj["position"] += obj["velocity"] * self.delta_t

            self.graph_objects[obj["name"]].set_center(obj["position"])
            # self.graph_objects[obj["name"]].set_color(obj["color"])

        return self.graph_objects.values()

    def add_object(self, obj: dict) -> None:
        """Adds a new object to the simulation"""
        self.objects.append(obj)
        if obj["type"] == "circle":
            circle = Circle(
                obj["position"], obj["radius"], color=obj["color"], animated=True
            )
            self.graph_objects[obj["name"]] = circle
            self.ax.add_patch(circle)  # Add circle to the Axes


if __name__ == "__main__":
    sim = Simulation(e = 0.9, g=3, delta_t = 0.05,bg_color="black")
    for i in range(100):
        circle = dict(
            name=f"circle_{i}",
            type="circle",
            position=np.random.random(2),  # Initial position
            velocity=(0.5-np.random.random(2))/2,  # Velocity vector
            radius=0.01+np.random.random()/80,  # Circle radius
            density=1,
            color=np.random.random(3),
        )
        sim.add_object(circle)
    sim.run()

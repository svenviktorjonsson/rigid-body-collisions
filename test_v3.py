"""Animated smooth-disk simulation. Run with ``python test_v3.py``."""

import abc
import time
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import numpy as np

from physics import advance_disks


class Animation(abc.ABC):
    def __init__(self, dt, bg_color="black"):
        self.dt = dt
        self.fig = plt.figure(figsize=(6, 6), dpi=150)
        self.fig.set_facecolor(bg_color)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_aspect("equal")
        self.ax.set_facecolor(bg_color)
        self.ax.axis("off")

    @abc.abstractmethod
    def update(self, time_index: int) -> Iterable:
        pass

    def run(self) -> None:
        self.animation = FuncAnimation(
            self.fig, self.update,
            init_func=lambda: tuple(self.graph_objects.values()),
            frames=1000, interval=self.dt * 1000, blit=True, repeat=False,
        )
        plt.show()


class Simulation(Animation):
    def __init__(self, dt, e=1, g=9.82, **kwargs):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if not np.isfinite(e) or not 0 <= e <= 1:
            raise ValueError("e must be between 0 and 1")
        if not np.isfinite(g) or g < 0:
            raise ValueError("g must be finite and nonnegative")
        super().__init__(dt, **kwargs)
        self.e = e
        self.g = g
        self.objects = np.zeros((100, 11))
        self.graph_objects = {}
        self.end_index = 0
        self.cols = {
            "t": 0, "x": slice(1, 3), "v": slice(3, 5), "r": 5,
            "m": 6, "xx": 7, "xv": 8, "vv": 9, "rr": 10,
        }
        self.update_times = []

    def get_all(self, col):
        return self.objects[:self.end_index, self.cols[col]]

    def overlaps(self, circle: dict, margin=0):
        x = np.asarray(circle["position"], dtype=float)
        r = circle["radius"]
        return np.sum((x - self.get_all("x")) ** 2, axis=1) < (
            r + self.get_all("r") + margin
        ) ** 2

    def inside(self, circle: dict):
        x, y = circle["position"]
        r = circle["radius"]
        return r <= x <= 1 - r, r <= y <= 1 - r

    def add_object(self, obj):
        if obj["type"] != "circle":
            raise ValueError("Only circle objects are supported")
        x = np.asarray(obj["position"], dtype=float)
        v = np.asarray(obj["velocity"], dtype=float)
        r, density = obj["radius"], obj["density"]
        if x.shape != (2,) or v.shape != (2,) or not np.all(np.isfinite([x, v])):
            raise ValueError("position and velocity must be finite 2D vectors")
        if not np.isfinite(r) or not 0 < r < 0.5:
            raise ValueError("radius must be between 0 and 0.5")
        if not np.isfinite(density) or density <= 0:
            raise ValueError("density must be finite and positive")
        if not all(self.inside(obj)) or np.any(self.overlaps(obj)):
            raise ValueError("Circles must start inside the box without overlaps")
        m = density * np.pi * r * r
        if not np.isfinite(m) or m <= 0:
            raise ValueError("mass must be finite and positive")
        if self.end_index >= len(self.objects):
            self.objects = np.pad(self.objects, [(0, len(self.objects)), (0, 0)])
        self.objects[self.end_index] = (
            0, *x, *v, r, m, np.dot(x, x), np.dot(x, v), np.dot(v, v), r * r
        )
        circle = Circle(x, r, facecolor=obj["color"], edgecolor="none", antialiased=True)
        self.graph_objects[self.end_index] = circle
        self.ax.add_patch(circle)
        self.end_index += 1

    def recalculate_objects(self, time_index):
        start = time.perf_counter()
        x, v = self.get_all("x"), self.get_all("v")
        advance_disks(x, v, self.get_all("r"), self.get_all("m"), self.dt, self.e, self.g)
        self.get_all("xx")[:] = np.sum(x * x, axis=1)
        self.get_all("xv")[:] = np.sum(x * v, axis=1)
        self.get_all("vv")[:] = np.sum(v * v, axis=1)
        self.update_times.append(time.perf_counter() - start)

    def update(self, time_index):
        self.recalculate_objects(time_index)
        for index, circle in self.graph_objects.items():
            circle.set_center(self.objects[index, 1:3])
        return tuple(self.graph_objects.values())


if __name__ == "__main__":
    count = 50
    rng = np.random.default_rng()
    sim = Simulation(dt=0.05, e=1, g=0, bg_color="black")
    for _ in range(100000):
        if sim.end_index == count:
            break
        circle = dict(
            name=f"circle_{sim.end_index}", type="circle",
            position=rng.random(2), velocity=0.5 - rng.random(2),
            radius=2 / count + rng.random() / count,
            density=1, color=rng.random(3),
        )
        if not np.any(sim.overlaps(circle, margin=0.005)) and all(sim.inside(circle)):
            sim.add_object(circle)
    else:
        raise RuntimeError("Could not place all circles; reduce their count or radii")
    sim.run()

import time
import numpy as np
import abc
from matplotlib.patches import Circle


class Object(abc.ABC):

    def __init__(self):
        self.artist = None

    def bind_to(self,data):
        for attr,idx in self.bindings():
            setattr(self.artist,attr,data[idx].view())

    @abc.abstractmethod
    def bindings(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def init(self) -> np.ndarray:
        pass

    def update(self):
        if self.artist is not None:
            self.artist.stale = True

class Circle(Object):
    
    def init(self):
        pass

    def bindings(self):
        pass



class Simulation:

    def __init__(self):
        self.objects = []
        self.data = np.zeros((100,10))

    def add(self, *objects):
        start_index = self.count()
        new_data = np.array([obj.configuration() for obj in objects], dtype=float)
        self.data[start_index:start_index+len(objects)] = new_data
        self.objects.extend(objects)

    def count(self):
        return len(self.objects)

    def next_event(self):
        i,j = np.triu_indices(self.count())
        event_moments = self.calculate_event_times(i,j)
        print(event_moments)

    def interaction_time(self,psi,psj):
        return np.random.random()

    def calculate_event_times(self,is_,js):
        pass
        # psi = np.array([obj[i].configuration) for i in is_])
        # psi = np.array(self.objects[i].configuration)
        # psj = np.array(self.objects[j].configuration)
        # tij = self.interaction_time(psi,psj)




if __name__=="__main__":
    sim = Simulation()

    for i in range(10):
        sim.add_object()

    while sim.has_events():
        time.sleep(1)
        print(sim.next_event())
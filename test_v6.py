import abc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class SharedArtistManager(metaclass=abc.ABCMeta):
    def __init__(self, data_view: np.ndarray, ax: plt.Axes):
        """
        Initialize the manager with a view of shared data and a Matplotlib Axes.

        :param data_view: A view into a shared array (e.g., shape (n_objects, attributes)).
        :param ax: Matplotlib Axes object where the artists are drawn.
        """
        self.data_view = data_view
        self.ax = ax
        self.artists = self._create_artists()

        # Assign the data views to the artists
        self._assign_views_to_artists()

    @abc.abstractmethod
    def _create_artists(self):
        """
        Create and return a list of Matplotlib artists or patches. 
        This must be implemented by subclasses.
        """
        pass

    @abc.abstractmethod
    def _assign_views_to_artists(self):
        """
        Define how the shared data views are assigned to the internal artist data.
        This must be implemented by subclasses.
        """
        pass

    def update(self):
        """
        Trigger a redraw of the canvas to reflect updated data.
        """
        for artist in self.artists:
            artist.stale = True
        self.ax.figure.canvas.draw_idle()


class LineManager(SharedArtistManager):
    def _create_artists(self):
        """
        Create a Line2D object for each row in the shared data view.
        Each Line2D represents a single point (x, y).
        """
        n_objects = self.data_view.shape[0]
        return [self.ax.plot([], [], 'o', markersize=5)[0] for _ in range(n_objects)]

    def _assign_views_to_artists(self):
        """
        Assign each row in the data_view (representing x and y) to a Line2D artist.
        """
        for i, line in enumerate(self.artists):
            line._x = self.data_view[i:i + 1, 0].view()  # x-coordinates
            line._y = self.data_view[i:i + 1, 1].view()  # y-coordinates


# Test script
if __name__ == "__main__":
    # Shared data for 1000 objects (x, y positions)
    n_objects = 1000
    positions = np.random.rand(n_objects, 2) * 10  # Random initial positions

    # Matplotlib setup
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Create a manager for Line2D artists
    manager = LineManager(data_view=positions, ax=ax)

    # Update positions dynamically
    import time

    for _ in range(100):
        positions[:, 0] += np.random.randn(n_objects) * 0.1  # Modify x-coordinates
        positions[:, 1] += np.random.randn(n_objects) * 0.1  # Modify y-coordinates
        manager.update()
        time.sleep(0.1)

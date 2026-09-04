import matplotlib.pyplot as plt
import numpy as np

def distance_squared(p1,p2):
    return np.sum((np.array(p1)-np.array(p2))**2)

class InteractivePolygonDrawer:
    def __init__(self):
        # Initialize the figure and axis with black background
        self.fig, self.ax = plt.subplots()
        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('black')

        # Remove all spines, ticks, and labels
        self.ax.set_xlim(-1,1)
        self.ax.set_ylim(-1,1)
        self.ax.axis("off")


        # Initialize list to store points
        self.polygons = [[]]
        self.polygon_index = 0
        self.point_radius = 0.05

        # Initialize Line2D object for lines
        self.lines = self.new_lines()  # White dotted lines
        self.markers = self.new_markers()
        self.selection = self.new_selection()

        # Connect the click event
        self.press_cid = self.fig.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.move_cid = self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.release_cid = self.fig.canvas.mpl_connect('button_release_event', self.on_button_release)

        self.button_down_point = None
        self.picked_index = None
        self.selected_indices = []
        

        plt.show()
    
    def new_lines(self):
        return self.ax.plot([], [], "w--",linewidth=1)
    
    def new_markers(self):
        return self.ax.plot([], [], "w.",markersize=10)
    
    def new_selection(self):
        return self.ax.plot([], [], "bo",markerfacecolor="none", markersize=12)[0]

    def is_near(self,p1,p2):
        return distance_squared(p1,p2)<self.point_radius**2
    
    def on_button_press(self,event):
        if event.inaxes != self.ax:
            return
        if event.button == 1:
            self.button_down_point = event.xdata, event.ydata
            self.picked_index = next(((i,j) for i,polygon in enumerate(self.polygons) for j,point in enumerate(polygon) if self.is_near(self.button_down_point,point)),None)
            if self.picked_index is not None:
                self.select_picked_point()
        elif event.button == 3:
            self.selected_indices.clear()
            self.update_selection()

    def select_picked_point(self):
        self.selected_indices.append(self.picked_index)
        self.update_selection()

    def on_mouse_move(self,event):
        if self.picked_index is not None:
            x, y = event.xdata, event.ydata
            self.move_picked_point(x,y)

    def move_picked_point(self,x,y):
        i,j = self.picked_index
        self.polygons[i][j] = x,y
        self.update_lines(i)
        self.fig.canvas.draw()

    def on_button_release(self, event):
        # Ensure click is within the axes
        if event.inaxes != self.ax:
            return

        # Left mouse button
        if event.button == 1:
            x, y = event.xdata, event.ydata
            if self.is_near((x,y),self.button_down_point):
                if len(self.polygons[-1])<3 or not self.is_near((x,y),self.polygons[-1][0]):
                    self.polygons[self.polygon_index].append((x, y))
                    self.update_lines()
                    self.fig.canvas.draw()
                else:
                    self.polygons[self.polygon_index].append(self.polygons[self.polygon_index][0])
                    self.update_lines()
                    self.fig.canvas.draw()

                    self.polygons.append([])
                    self.polygon_index+=1
                    self.lines.extend(self.new_lines())
                    self.markers.extend(self.new_markers())
        self.button_down_point = None
        self.picked_index = None

    def update_selection(self):
        if self.selected_indices:
            x, y = np.array([self.polygons[i][j] for i,j in self.selected_indices]).T
            self.selection.set_data(x,y)
        else:
            self.selection.set_data([],[])


    def update_lines(self,index=-1):
        if len(self.polygons[index]) > 0:
            # Unzip the list of points
            x, y = zip(*self.polygons[index])
            self.lines[-1].set_data(x, y)
            self.markers[-1].set_data(x, y)
        else:
            self.lines[-1].set_data([], [])
            self.markers[-1].set_data([], [])

# Run the interactive drawer
if __name__ == "__main__":
    InteractivePolygonDrawer()

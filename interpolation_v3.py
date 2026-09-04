import numpy as np
import matplotlib.pyplot as plt

params = dict(
    y0 = 1/2,
    a0 = np.pi,
    a = 0.03267,
    p = 2.1946
)

def lamb(angle, y0, a, a0, p):
    return y0 + a*np.abs(angle - a0)**p

def new_point(p0,p1,p2,p3):
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
    x=lamb(angle,**params)
    return px+(mu-px)*x

def refine_path(ps,iters=5):
    for i in range(iters):
        new_points=[]
        for shift in range(len(ps)):
            psr=np.roll(ps, 1-shift, axis=0)[:4]
            if len(psr)==3: 
                psr=(*psr,psr[0])
            pn = new_point(*psr)
            new_points.append(pn)
        for k,p in enumerate(new_points):
            ps.insert(2*k+1,p)

def distance_squared(p1,p2):
    return np.sum((np.array(p1)-np.array(p2))**2)

class InteractivePolygonDrawer:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('black')
        self.ax.set_xlim(-1,1)
        self.ax.set_ylim(-1,1)
        self.ax.axis("off")
        self.ax.set_aspect("equal")

        self.polygons = [[]]
        self.polygon_index = 0
        self.point_radius = 0.05

        self.lines = self.new_lines()
        self.markers = self.new_markers()
        self.selection = self.new_selection()

        self.press_cid = self.fig.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.move_cid = self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.release_cid = self.fig.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.key_release_cid = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

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
            self.picked_index = next(((i,j) for i,polygon in enumerate(self.polygons) 
                                      for j,point in enumerate(polygon) 
                                      if self.is_near(self.button_down_point,point)),None)
            if self.picked_index is not None:
                self.select_picked_point()
        elif event.button == 3:
            self.selected_indices.clear()
            self.update_selection()

    def select_picked_point(self):
        self.selected_indices.append(self.picked_index)
        self.update_selection()

    def on_mouse_move(self,event):
        if self.picked_index is not None and event.xdata is not None and event.ydata is not None:
            x, y = event.xdata, event.ydata
            self.move_picked_point(x,y)

    def on_key_press(event):
        pass

    def move_picked_point(self,x,y):
        i,j = self.picked_index
        self.polygons[i][j] = (x,y)
        self.markers[i][j] = (x,y)
        self.update_lines(i)
        self.fig.canvas.draw()

    def on_button_release(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 1:
            x, y = event.xdata, event.ydata
            # Check if it's a click (not a drag)
            if self.is_near((x,y),self.button_down_point):
                # If polygon not complete or not closing
                if len(self.polygons[-1])<3 or not self.is_near((x,y),self.polygons[-1][0]):
                    self.polygons[self.polygon_index].append((x, y))
                    self.update_lines()
                    self.fig.canvas.draw()
                else:
                    # Close the polygon
                    self.polygons[self.polygon_index].append(self.polygons[self.polygon_index][0])
                    self.update_lines()
                    self.fig.canvas.draw()

                    # Now we have a closed polygon, let's refine a portion:
                    ps = self.polygons[self.polygon_index]
                    # Convert to a list of np arrays to work with refine_path
                    ps = [np.array(p) for p in ps]

                    # We consider only from second point to the last-but-one (since last is the first point repeated)
                    # Slice the polygon to skip the first point and the closing point
                    ps_to_refine = ps[:]

                    # Refine only this portion for 3 iterations
                    refine_path(ps_to_refine, iters=3)

                    # Reconstruct the polygon with the first point, refined middle points, and last point
                    ps = [ps[0]] + ps_to_refine + [ps[0]]
                    self.polygons[self.polygon_index] = ps

                    # Update lines to show refined polygon
                    self.update_lines()
                    self.fig.canvas.draw()

                    # Prepare for next polygon
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
            x, y = zip(*[(p[0],p[1]) for p in self.polygons[index]])
            self.lines[-1].set_data(x, y)
            self.markers[-1].set_data(x, y)
        else:
            self.lines[-1].set_data([], [])
            self.markers[-1].set_data([], [])

if __name__ == "__main__":
    InteractivePolygonDrawer()

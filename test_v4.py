
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle,Ellipse,Polygon

if __name__=="__main__":
    line_data = np.random.random((10,12))
    line_data[:,0] = 0
    line_data[:,3:-3]/=5


    fig = plt.figure(figsize=(6, 6), dpi=150)
    fig.set_facecolor("black")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-1,2)
    ax.set_ylim(-1,2)
    ax.set_facecolor("black")
    ax.axis("off")
    dt = 0.01
    patches = []

    def init():
        for t,x,y,rx,ry,a,vx,vy,w,*color in line_data:
            type_ = np.random.random_integers(0,2)
            if type_ == 0:
                obj = Rectangle((x-rx, y-ry), rx*2, ry*2, angle=a*180,color=color,rotation_point = 'center')
            elif type_ == 1:
                obj = Ellipse((x,y),rx*2,ry*2,angle = a*180,color=color)

            patches.append(obj)
            ax.add_patch(obj)

    def update(index):
        angle = line_data[:,5]
        omega = line_data[:,8]
        coord = line_data[:,1:3]
        velocity = line_data[:,6:8]
        angle += 180*omega*dt
        coord += velocity*dt
        for i,(t,x,y,rx,ry,a,vx,vy,w,*color) in enumerate(line_data):
            if isinstance(patches[i],Ellipse):
                patches[i].set_center((x,y))
            elif isinstance(patches[i],Rectangle):
                patches[i].set_bounds(x,y,rx*2,ry*2)
            patches[i].set_angle(a)
        
        return patches


    init()
    animation = FuncAnimation(
            fig, update, frames = 1000, interval=dt * 1000, blit=True
        )
    plt.show()
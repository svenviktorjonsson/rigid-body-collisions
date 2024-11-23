import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Set up the figure and axis

fig, ax = plt.subplots()
fig.set_facecolor("black")
ax.set_facecolor("black")
ax.axis("off")  # Remove axes, labels, and spines

# Set up the dot
radius = 1.0
dot, = ax.plot([], [], 'o', color='white')

# Set limits
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)

# Time step
delta_t = 0.05  # seconds
total_time = 10  # seconds
frames = int(total_time / delta_t)  # Total frames

# Initialize function
def init():
    dot.set_data([], [])
    return dot,

# Update function for animation
def update(time_index):
    t = time_index * delta_t  # Current time
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    dot.set_data([x], [y])
    return dot,

# Create the animation with a specified interval
interval = delta_t * 1000  # Convert seconds to milliseconds
ani = FuncAnimation(fig, update, frames=frames, init_func=init, interval=interval, blit=True)

plt.show()

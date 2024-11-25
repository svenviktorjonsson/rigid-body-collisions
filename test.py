import numpy as np

# Example data
x = np.array([[0.9, 0.8], [0.1, 0.2]])
v = np.array([[0.2, -0.1], [-0.2, 0.3]])
dt = 0.1
r = np.array([[2],[3]])

# High indices: where x + v * dt > 1 - r
hr,hc = np.where(x + v * dt > 1 - r)
if len(hr):
    x[hr,hc] = 2 * (1 - r[hr]) - x[hr,hc]
    v[hr,hc] *= -1

# Low indices: where x + v * dt < r
lr,lc = np.where(x + v * dt < r)
if len(lr):
    x[lr,lc] = 2 * r[lr]- x[lr,lc]
    v[lr,lc] *= -1

# Outputs
print("Updated positions (x):")
print(x)
print("Updated velocities (v):")
print(v)

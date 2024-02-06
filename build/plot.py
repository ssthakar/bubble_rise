import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# Load coordinates and temperatures from files
with open('./gridSave1.000000', 'r') as f_coords:
    coords = np.loadtxt(f_coords)

with open('./fields2.000000', 'r') as f_temps:
    temps = np.loadtxt(f_temps)


f_bubble = np.genfromtxt('./bubble1.csv', delimiter=',')
# Extract x, y, and temperature values
x = coords[:, 0]
y = coords[:, 1]

temperature = temps[:, 3]
# Create a triangulation from the x, y coordinates
triang = tri.Triangulation(x, y)
# Create a second triangulation from the x1, y1 coordinates

# Create a 1x2 grid for subplots
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Plot the first tricontourf plot
contour1 = axs[0].tricontourf(triang, temperature, cmap='jet', levels=40)
axs[0].set_title('x Velocity at t = 0.5 secs')
axs[0].set_xlabel('X-axis')
axs[0].set_ylabel('Y-axis')
axs[0].set_aspect('equal')
fig.colorbar(contour1, ax=axs[0])
axs[0].scatter(f_bubble[:, 0], f_bubble[:, 1], c='black', marker='o', label='Bubble Data')
# Plot the second tricontourf plot
plt.tight_layout()  # Adjust layout for better spacing
plt.show()

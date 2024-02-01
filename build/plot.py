import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# Load coordinates and temperatures from files
with open('total.txt', 'r') as f_coords:
    coords = np.loadtxt(f_coords)

with open('intial.txt', 'r') as f_temps:
    temps = np.loadtxt(f_temps)

# Extract x, y, and temperature values
x = coords[:, 0]
y = coords[:, 1]
# temperature = temps[:,1]
temperature = temps
# Create a triangulation from the x, y coordinates
triang = tri.Triangulation(x, y)

# Plotting the tricontourf plot
plt.figure(figsize=(8, 6))
contour = plt.tricontourf(triang, temperature, cmap='coolwarm',levels=np.linspace(-1.1,1.1,100));
plt.colorbar(contour)  # Add color bar indicating temperature scale
plt.title('x Velcoity at t = 0.5 secs')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.gca().set_aspect('equal')
plt.show()

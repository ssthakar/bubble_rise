
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# Load coordinates and temperatures from files
with open('./gridSave0.500000', 'r') as f_coords:
    coords = np.loadtxt(f_coords)
with open('./total.txt', 'r') as f_coords1:
    coords1 = np.loadtxt(f_coords1)

with open('./intial.txt', 'r') as f_temps1:
    temps1 = np.loadtxt(f_temps1)

with open('./fieldsSave0.500000', 'r') as f_temps:
    temps = np.loadtxt(f_temps)

# Extract x, y, and temperature values
x = coords[:, 0]
y = coords[:, 1]
x1 = coords1[:, 0]
y1 = coords1[:, 1]

temperature = temps[:, 3]
temperature1 = temps1

# Create a triangulation from the x, y coordinates
triang = tri.Triangulation(x, y)

# Create a second triangulation from the x1, y1 coordinates
triang1 = tri.Triangulation(x1, y1)

# Combine the triangulations into one
combined_triang = tri.Triangulation(np.concatenate([triang.x, triang1.x]),
                                     np.concatenate([triang.y, triang1.y]))

# Combine the temperature values into one array
combined_temperature = np.concatenate([temperature, temperature1])

# Plot the tricontourf plot with combined triangulation
plt.figure(figsize=(8, 6))
contour = plt.tricontourf(combined_triang, combined_temperature, cmap='coolwarm', levels=np.linspace(-0.1, 0.1, 100))
plt.colorbar(contour)
plt.title('Combined Tricontourf Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.gca().set_aspect('equal')
plt.show()

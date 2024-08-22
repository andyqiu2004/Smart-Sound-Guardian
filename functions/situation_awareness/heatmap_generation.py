import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
from matplotlib.animation import FuncAnimation


class HeatmapGenerator:
    def __init__(self, resolution=100):
        self.resolution = resolution

    def generate_static_heatmap(
        self, data_points, intensity_values, grid_size=(100, 100)
    ):
        x = np.array([p[0] for p in data_points])
        y = np.array([p[1] for p in data_points])
        intensity = np.array(intensity_values)

        grid_x, grid_y = np.mgrid[
            min(x) : max(x) : complex(grid_size[0]),
            min(y) : max(y) : complex(grid_size[1]),
        ]
        grid_z = griddata((x, y), intensity, (grid_x, grid_y), method="cubic")

        plt.figure(figsize=(8, 6))
        sns.heatmap(grid_z, cmap="coolwarm")
        plt.title("Static Heatmap")
        plt.show()

    def generate_dynamic_heatmap(self, data_series, interval=100):
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.array([p[0] for p in data_series[0]])
        y = np.array([p[1] for p in data_series[0]])
        intensity = np.array([p[2] for p in data_series[0]])

        grid_x, grid_y = np.mgrid[
            min(x) : max(x) : complex(self.resolution),
            min(y) : max(y) : complex(self.resolution),
        ]

        def animate(i):
            ax.clear()
            intensity = np.array([p[2] for p in data_series[i]])
            grid_z = griddata((x, y), intensity, (grid_x, grid_y), method="cubic")
            sns.heatmap(grid_z, cmap="coolwarm", ax=ax)
            ax.set_title(f"Dynamic Heatmap - Frame {i+1}")

        anim = FuncAnimation(fig, animate, frames=len(data_series), interval=interval)
        plt.show()

    def generate_3d_heatmap(self, data_points, intensity_values):
        x = np.array([p[0] for p in data_points])
        y = np.array([p[1] for p in data_points])
        intensity = np.array(intensity_values)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        grid_x, grid_y = np.mgrid[
            min(x) : max(x) : complex(self.resolution),
            min(y) : max(y) : complex(self.resolution),
        ]
        grid_z = griddata((x, y), intensity, (grid_x, grid_y), method="cubic")

        ax.plot_surface(grid_x, grid_y, grid_z, cmap="coolwarm")
        ax.set_title("3D Heatmap")
        plt.show()

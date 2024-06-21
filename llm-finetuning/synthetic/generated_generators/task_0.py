
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Tuple
import random

class SimpleGridDescriptionTask(ABC):
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.color_map = {
            0: "black",
            1: "red",
            2: "green",
            3: "blue",
            4: "yellow",
            5: "purple",
            6: "orange",
        }

    @abstractmethod
    def execute(self) -> tuple[np.ndarray, str]:
        pass

    def visualize(self, grid: np.ndarray, description: str):
        fig, ax = plt.subplots(figsize=(10, 10))

        # Determine the unique values in the grid
        unique_values = np.unique(grid)
        n_colors = len(unique_values)

        # Create a mapping of grid values to color indices
        value_to_index = {val: i for i, val in enumerate(unique_values)}

        # Create a new grid with mapped indices
        indexed_grid = np.vectorize(value_to_index.get)(grid)

        # Create a custom colormap
        colors = [self.color_map[i] for i in unique_values]
        cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors, N=n_colors)

        x_grid_size, y_grid_size = grid.shape
        
        # Display the grid
        cax = ax.imshow(indexed_grid, cmap=cmap, interpolation='nearest')
        ax.set_title("Generated Grid")
        ax.set_xticks(np.arange(-0.5, x_grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, y_grid_size, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", size=0)

        # Create a colorbar
        cbar = fig.colorbar(cax, ax=ax, ticks=range(n_colors))
        cbar.set_ticklabels([self.color_map[i].capitalize() for i in unique_values])

        ax.set_xticks(np.arange(0, x_grid_size, 1))
        ax.set_yticks(np.arange(0, y_grid_size, 1))

        # Add description as text below the grid
        plt.figtext(
            0.5, 0.01, description, wrap=True, horizontalalignment="center", fontsize=12
        )

        # Disable cursor value display
        ax.format_cursor_data = lambda data: ''

        plt.tight_layout()
        plt.show()
        
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Tuple
import random

class SimpleGridDescriptionTask(ABC):
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.color_map = {
            0: "black",
            1: "red",
            2: "green",
            3: "blue",
            4: "yellow",
            5: "purple",
            6: "orange",
        }

    @abstractmethod
    def execute(self) -> tuple[np.ndarray, str]:
        pass

    def visualize(self, grid: np.ndarray, description: str):
        fig, ax = plt.subplots(figsize=(10, 10))

        # Determine the unique values in the grid
        unique_values = np.unique(grid)
        n_colors = len(unique_values)

        # Create a mapping of grid values to color indices
        value_to_index = {val: i for i, val in enumerate(unique_values)}

        # Create a new grid with mapped indices
        indexed_grid = np.vectorize(value_to_index.get)(grid)

        # Create a custom colormap
        colors = [self.color_map[i] for i in unique_values]
        cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors, N=n_colors)

        x_grid_size, y_grid_size = grid.shape
        
        # Display the grid
        cax = ax.imshow(indexed_grid, cmap=cmap, interpolation='nearest')
        ax.set_title("Generated Grid")
        ax.set_xticks(np.arange(-0.5, x_grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, y_grid_size, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", size=0)

        # Create a colorbar
        cbar = fig.colorbar(cax, ax=ax, ticks=range(n_colors))
        cbar.set_ticklabels([self.color_map[i].capitalize() for i in unique_values])

        ax.set_xticks(np.arange(0, x_grid_size, 1))
        ax.set_yticks(np.arange(0, y_grid_size, 1))

        # Add description as text below the grid
        plt.figtext(
            0.5, 0.01, description, wrap=True, horizontalalignment="center", fontsize=12
        )

        # Disable cursor value display
        ax.format_cursor_data = lambda data: ''

        plt.tight_layout()
        plt.show()

class RandomShapes(SimpleGridDescriptionTask):
    def __init__(self, grid_size, num_shapes):
        super().__init__(grid_size)
        self.num_shapes = num_shapes

    def generate_T_shape(self, color, x, y):
        shape = np.zeros((3, 3), dtype=int)
        shape[0, 1] = shape[1, 1] = shape[2, 1] = shape[1, 0] = shape[1, 2] = color
        return shape

    def generate_L_shape(self, color, x, y):
        shape = np.zeros((3, 3), dtype=int)
        shape[0, 0] = shape[1, 0] = shape[2, 0] = shape[2, 1] = shape[2, 2] = color
        return shape

    def place_shape(self, grid, shape, x, y):
        for i in range(3):
            for j in range(3):
                if 0 <= x + i < self.grid_size and 0 <= y + j < self.grid_size:
                    if shape[i, j] != 0:
                        grid[x + i, y + j] = shape[i, j]

    def execute(self) -> tuple[np.ndarray, str]:
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        shapes = []
        
        for _ in range(self.num_shapes):
            shape_type = random.choice(['T', 'L'])
            color = random.randint(1, 6)
            x, y = np.random.randint(0, self.grid_size - 3, size=2)
            if shape_type == 'T':
                shape = self.generate_T_shape(color, x, y)
            else:
                shape = self.generate_L_shape(color, x, y)
            
            self.place_shape(grid, shape, x, y)
            shapes.append((shape_type, self.color_map[color], x, y))
        
        shape_descriptions = [f"a {color} {stype} shape at ({x}, {y})" for stype, color, x, y in shapes]
        description = "A grid with the following shapes: " + ", ".join(shape_descriptions)
        
        return grid, description

# Example usage
grid_size = 10
num_shapes = 5
task = RandomShapes(grid_size, num_shapes)
grid, description = task.execute()

print(description)
print(grid)

# Visualize the grid
task.visualize(grid, description)

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

        unique_values = np.unique(grid)
        n_colors = len(unique_values)

        value_to_index = {val: i for i, val in enumerate(unique_values)}

        indexed_grid = np.vectorize(value_to_index.get)(grid)
        colors = [self.color_map[i] for i in unique_values]
        cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors, N=n_colors)

        x_grid_size, y_grid_size = grid.shape
        cax = ax.imshow(indexed_grid, cmap=cmap, interpolation='nearest')
        ax.set_title("Generated Grid")
        ax.set_xticks(np.arange(-0.5, x_grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, y_grid_size, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", size=0)

        cbar = fig.colorbar(cax, ax=ax, ticks=range(n_colors))
        cbar.set_ticklabels([self.color_map[i].capitalize() for i in unique_values])

        ax.set_xticks(np.arange(0, x_grid_size, 1))
        ax.set_yticks(np.arange(0, y_grid_size, 1))
        plt.figtext(
            0.5, 0.01, description, wrap=True, horizontalalignment="center", fontsize=12
        )
        ax.format_cursor_data = lambda data: ''
        plt.tight_layout()
        plt.show()

class LShapePatternTask(SimpleGridDescriptionTask):
    def __init__(self, grid_size, num_l_shapes):
        super().__init__(grid_size)
        self.num_l_shapes = num_l_shapes

    def execute(self) -> tuple[np.ndarray, str]:
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        l_shapes_info = []

        for _ in range(self.num_l_shapes):
            color = random.randint(1, 6)
            x = random.randint(0, self.grid_size - 2)
            y = random.randint(0, self.grid_size - 2)
            orientation = random.choice(['upright', 'downright', 'upleft', 'downleft'])

            if orientation == 'upright':
                if x + 1 < self.grid_size and y + 2 < self.grid_size:
                    grid[x, y:y+2] = color
                    grid[x+1, y] = color
            elif orientation == 'downright':
                if x + 2 < self.grid_size and y + 1 < self.grid_size:
                    grid[x:x+2, y] = color
                    grid[x+1, y+1] = color
            elif orientation == 'upleft':
                if x + 1 < self.grid_size and y - 1 >= 0:
                    grid[x, y-1:y+1] = color
                    grid[x+1, y] = color
            elif orientation == 'downleft':
                if x + 2 < self.grid_size and y - 1 >= 0:
                    grid[x:x+2, y] = color
                    grid[x+1, y-1] = color

            l_shapes_info.append((color, x, y, orientation))

        description = f"A grid with {self.num_l_shapes} random 'L'-shapes of different colors and orientations."
        return grid, description

# Example usage
grid_size = 20
num_l_shapes = 5
task = LShapePatternTask(grid_size, num_l_shapes)
grid, description = task.execute()

print(description)
print(grid)

# Visualize the grid
task.visualize(grid, description)
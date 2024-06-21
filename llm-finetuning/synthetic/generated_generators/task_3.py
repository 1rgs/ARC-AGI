
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

class CheckerboardWithBlocks(SimpleGridDescriptionTask):
    def __init__(self, grid_size, num_colors):
        super().__init__(grid_size)
        self.num_colors = min(num_colors, 6)

    def execute(self) -> tuple[np.ndarray, str]:
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Choose two colors for the checkerboard pattern
        color1, color2 = random.sample(range(1, 7), 2)
        
        # Create the checkerboard pattern
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i + j) % 2 == 0:
                    grid[i, j] = color1
                else:
                    grid[i, j] = color2

        # Choose a third color for the random blocks
        block_color = random.choice([c for c in range(1, 7) if c not in (color1, color2)])

        # Place a few random blocks on the checkerboard pattern
        num_blocks = random.randint(3, 6)
        for _ in range(num_blocks):
            block_size = random.randint(1, 4)
            x = random.randint(0, self.grid_size - block_size)
            y = random.randint(0, self.grid_size - block_size)
            grid[x:x+block_size, y:y+block_size] = block_color

        description = (
            f"A checkerboard pattern with {self.color_map[color1]} and {self.color_map[color2]} "
            f"with random {self.color_map[block_color]} blocks superimposed."
        )

        return grid, description

# Example usage
grid_size = 20
num_colors = 5
task = CheckerboardWithBlocks(grid_size, num_colors)
grid, description = task.execute()

print(description)
print(grid)

# Visualize the grid
task.visualize(grid, description)
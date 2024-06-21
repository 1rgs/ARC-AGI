
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
        
class RandomForest(SimpleGridDescriptionTask):
    def __init__(self, grid_size, num_trees):
        super().__init__(grid_size)
        self.num_trees = num_trees

    def create_tree(self, grid: np.ndarray, color: int, shape: str, position: Tuple[int, int]) -> bool:
        x, y = position
        if shape == "small_rectangle":
            width, height = 2, 2
        elif shape == "large_rectangle":
            width, height = 3, 3
        elif shape == "L_shape":
            width, height = 3, 2

        # Ensure the tree can be placed within the grid
        if x + width > self.grid_size or y + height > self.grid_size:
            return False

        # Check overlap
        if np.any(grid[x:x + width, y:y + height] != 0):
            return False

        # Place the tree
        if shape == "small_rectangle":
            grid[x:x + width, y:y + height] = color
        elif shape == "large_rectangle":
            grid[x:x + width, y:y + height] = color
        elif shape == "L_shape":
            grid[x:x + 2, y:y + 2] = color
            grid[x + 2, y] = color
            grid[x + 2, y + 1] = color

        return True

    def execute(self) -> tuple[np.ndarray, str]:
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        shapes = ["small_rectangle", "large_rectangle", "L_shape"]
        colors = np.random.choice(range(1, 7), size=self.num_trees, replace=True)

        trees_placed = 0
        attempts = 0
        max_attempts = self.num_trees * 10

        while trees_placed < self.num_trees and attempts < max_attempts:
            color = colors[trees_placed]
            shape = random.choice(shapes)
            position = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))

            if self.create_tree(grid, color, shape, position):
                trees_placed += 1
            attempts += 1

        description = f"A grid representing a forest with {trees_placed} trees of various shapes and colors."
        return grid, description

# Example usage
grid_size = 20  # Increased size for better visibility
num_trees = 10
task = RandomForest(grid_size, num_trees)
grid, description = task.execute()

print(description)
print(grid)

# Visualize the grid
task.visualize(grid, description)
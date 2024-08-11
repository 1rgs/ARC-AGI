
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
        
class ZigzagPaths(SimpleGridDescriptionTask):
    def __init__(self, grid_size, num_zigzags):
        super().__init__(grid_size)
        self.num_zigzags = num_zigzags

    def generate_zigzag(self, color1, color2, start_pos, direction):
        x, y = start_pos
        dx, dy = direction
        zigzag_pattern = []

        for i in range(random.randint(5, 10)):  # Randomly choose the length of zigzag
            segment_length = random.randint(2, 5)
            for _ in range(segment_length):
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    zigzag_pattern.append((x, y, color1 if (i % 2 == 0) else color2))
                x += dx
                y += dy
            # Change direction: zigzag pattern
            dx, dy = dy, dx

        return zigzag_pattern

    def execute(self) -> tuple[np.ndarray, str]:
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # Cardinal directions

        for _ in range(self.num_zigzags):
            color1, color2 = random.sample(range(1, 7), 2)
            start_pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            direction = random.choice(directions)
            zigzag = self.generate_zigzag(color1, color2, start_pos, direction)

            for x, y, color in zigzag:
                grid[x, y] = color

        description = (
            f"A grid with {self.num_zigzags} zigzag paths, each alternating between two colors."
        )

        return grid, description

# Example usage
grid_size = 20
num_zigzags = 5
task = ZigzagPaths(grid_size, num_zigzags)
grid, description = task.execute()

print(description)
print(grid)

# Visualize the grid
task.visualize(grid, description)
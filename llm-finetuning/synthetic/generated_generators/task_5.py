
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
        
class TetrisShapePlacement(SimpleGridDescriptionTask):
    def __init__(self, grid_size):
        super().__init__(grid_size)
        self.tetromino_shapes = {
            'I': np.array([[1, 1, 1, 1]]),
            'O': np.array([[1, 1], [1, 1]]),
            'T': np.array([[0, 1, 0], [1, 1, 1]]),
            'S': np.array([[0, 1, 1], [1, 1, 0]]),
            'Z': np.array([[1, 1, 0], [0, 1, 1]]),
            'J': np.array([[1, 0, 0], [1, 1, 1]]),
            'L': np.array([[0, 0, 1], [1, 1, 1]])
        }

    def place_shape(self, grid, shape, x, y, color):
        shape_height, shape_width = shape.shape
        for i in range(shape_height):
            for j in range(shape_width):
                if shape[i, j] == 1:
                    grid[x + i, y + j] = color

    def can_place_shape(self, grid, shape, x, y):
        shape_height, shape_width = shape.shape
        if x + shape_height > self.grid_size or y + shape_width > self.grid_size:
            return False
        for i in range(shape_height):
            for j in range(shape_width):
                if shape[i, j] == 1 and grid[x + i, y + j] != 0:
                    return False
        return True

    def execute(self) -> tuple[np.ndarray, str]:
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        descriptions = []
        placed_shapes = []

        for _ in range(5):  # Place 5 shapes
            shape_name = random.choice(list(self.tetromino_shapes.keys()))
            shape = self.tetromino_shapes[shape_name]
            color = random.randint(1, 6)

            # Try to place the shape in a random position
            placed = False
            for _ in range(100):  # Try 100 times to place the shape
                x = random.randint(0, self.grid_size - 1)
                y = random.randint(0, self.grid_size - 1)
                
                if self.can_place_shape(grid, shape, x, y):
                    self.place_shape(grid, shape, x, y, color)
                    placed_shapes.append((shape_name, color, (x, y)))
                    descriptions.append(f"{self.color_map[color]} {shape_name} at ({x}, {y})")
                    placed = True
                    break

            if not placed:
                continue

        description = "A grid with the following Tetris shapes placed: " + ", ".join(descriptions)
        return grid, description

# Example usage
grid_size = 20
task = TetrisShapePlacement(grid_size)
grid, description = task.execute()
print(description)
task.visualize(grid, description)
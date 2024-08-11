from ..simple_grid_definition_task import SimpleGridDescriptionTask

import random
import numpy as np
from typing import Tuple

class CornerSquareTask(SimpleGridDescriptionTask):
    def execute(self) -> Tuple[np.ndarray, str]:
        # Generate a random grid size between 3 and 32
        grid_size = random.randint(3, 32)
        grid = np.zeros((grid_size, grid_size), dtype=int)

        # Choose a random corner
        corners = [(0, 0), (0, grid_size - 1), (grid_size - 1, 0), (grid_size - 1, grid_size - 1)]
        corner = random.choice(corners)

        # Choose a random color and size for the square
        color = random.randint(1, 9)
        square_size = random.randint(1, min(grid_size // 2, 5))

        # Place the square in the chosen corner
        for i in range(square_size):
            for j in range(square_size):
                row = corner[0] + i
                col = corner[1] + j
                grid[row, col] = color

        # Generate the description
        corner_names = {
            (0, 0): "top-left",
            (0, grid_size - 1): "top-right",
            (grid_size - 1, 0): "bottom-left",
            (grid_size - 1, grid_size - 1): "bottom-right",
        }
        corner_name = corner_names[corner]
        color_name = self.color_map[color]
        description = f"The grid contains a {color_name} square of size {square_size}x{square_size} positioned in the {corner_name} corner."

        return grid, description

if __name__ == "__main__":
    task = CornerSquareTask()
    grid, description = task.execute()
    print(description)
    task.visualize(grid, description)
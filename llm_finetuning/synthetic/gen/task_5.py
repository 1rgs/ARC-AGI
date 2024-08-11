from ..simple_grid_definition_task import SimpleGridDescriptionTask

import numpy as np
from typing import Tuple
from random import randint, choice

class CenteredSquareTask(SimpleGridDescriptionTask):
    def execute(self) -> Tuple[np.ndarray, str]:
        # Generate a random grid size between 3 and 32
        grid_size = randint(3, 32)
        grid = np.zeros((grid_size, grid_size), dtype=int)

        # Generate a random color and size for the square
        square_color = randint(1, 9)
        square_size = randint(1, grid_size // 2)

        # Calculate the center coordinates of the grid
        center_x, center_y = grid_size // 2, grid_size // 2

        # Place the square in the center of the grid
        for x in range(center_x - square_size // 2, center_x + square_size // 2):
            for y in range(center_y - square_size // 2, center_y + square_size // 2):
                grid[x, y] = square_color

        # Generate the description
        color_name = self.color_map[square_color]
        description = f"The grid contains a single {color_name} square of size {square_size}x{square_size} placed exactly in the center."

        return grid, description

if __name__ == "__main__":
    task = CenteredSquareTask()
    grid, description = task.execute()
    print(description)
    task.visualize(grid, description)
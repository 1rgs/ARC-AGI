from ..simple_grid_definition_task import SimpleGridDescriptionTask

import numpy as np
from random import randint, choice

class SingleSquareTask(SimpleGridDescriptionTask):
    def execute(self) -> tuple[np.ndarray, str]:
        # Generate a random grid size between 3 and 32
        grid_size = randint(3, 32)
        grid = np.zeros((grid_size, grid_size), dtype=int)

        # Place a single square of random color at a random position
        square_color = randint(1, 9)  # Exclude 0 (black)
        square_size = randint(1, min(grid_size // 2, 5))  # Square size between 1 and min(grid_size//2, 5)
        start_row = randint(0, grid_size - square_size)
        start_col = randint(0, grid_size - square_size)

        for row in range(start_row, start_row + square_size):
            for col in range(start_col, start_col + square_size):
                grid[row, col] = square_color

        # Generate the description
        color_name = self.color_map[square_color]
        description = f"The grid contains a single {color_name} square of size {square_size}x{square_size} located at coordinates ({start_row}, {start_col})."

        return grid, description

if __name__ == "__main__":
    task = SingleSquareTask()
    grid, description = task.execute()
    print(description)
    task.visualize(grid, description)
from ..simple_grid_definition_task import SimpleGridDescriptionTask

import numpy as np
from typing import Tuple
from random import randint, sample

class TwoSquaresTask(SimpleGridDescriptionTask):
    def execute(self) -> Tuple[np.ndarray, str]:
        # Generate a random grid size between 3 and 32
        grid_size = randint(3, 32)
        grid = np.zeros((grid_size, grid_size), dtype=int)

        # Choose two random colors
        colors = sample(range(1, 10), 2)

        # Place the first square
        square1_size = randint(1, grid_size // 2)
        square1_x = randint(0, grid_size - square1_size)
        square1_y = randint(0, grid_size - square1_size)
        grid[square1_x:square1_x + square1_size, square1_y:square1_y + square1_size] = colors[0]

        # Place the second square
        square2_size = randint(1, grid_size // 2)
        square2_x = randint(0, grid_size - square2_size)
        square2_y = randint(0, grid_size - square2_size)

        # Check for overlap and adjust positions if necessary
        while np.any(grid[square2_x:square2_x + square2_size, square2_y:square2_y + square2_size] != 0):
            square2_x = randint(0, grid_size - square2_size)
            square2_y = randint(0, grid_size - square2_size)

        grid[square2_x:square2_x + square2_size, square2_y:square2_y + square2_size] = colors[1]

        # Generate the description
        description = f"The grid contains two squares of different colors: a {self.color_map[colors[0]]} square of size {square1_size}x{square1_size} at coordinates ({square1_x}, {square1_y}), and a {self.color_map[colors[1]]} square of size {square2_size}x{square2_size} at coordinates ({square2_x}, {square2_y})."

        return grid, description

if __name__ == "__main__":
    task = TwoSquaresTask()
    grid, description = task.execute()
    print(description)
    task.visualize(grid, description)
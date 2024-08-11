from ..simple_grid_definition_task import SimpleGridDescriptionTask

import numpy as np
from typing import Tuple
from random import randint, choice

class SingleRectangleTask(SimpleGridDescriptionTask):
    def execute(self) -> Tuple[np.ndarray, str]:
        # Generate a random grid size between 3 and 32
        grid_size = randint(3, 32)
        grid = np.zeros((grid_size, grid_size), dtype=int)

        # Generate random rectangle dimensions
        rect_width = randint(1, grid_size // 2)
        rect_height = randint(1, grid_size // 2)

        # Generate random rectangle position
        start_x = randint(0, grid_size - rect_width)
        start_y = randint(0, grid_size - rect_height)

        # Generate random rectangle color
        rect_color = randint(1, 9)

        # Draw the rectangle on the grid
        grid[start_y:start_y + rect_height, start_x:start_x + rect_width] = rect_color

        # Generate the description
        color_name = self.color_map[rect_color]
        description = f"The grid contains a single {color_name} rectangle of size {rect_width}x{rect_height} positioned at ({start_x}, {start_y})."

        return grid, description

if __name__ == "__main__":
    task = SingleRectangleTask()
    grid, description = task.execute()
    print(description)
    task.visualize(grid, description)
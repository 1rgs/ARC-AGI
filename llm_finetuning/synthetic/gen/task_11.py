from ..simple_grid_definition_task import SimpleGridDescriptionTask

import numpy as np
from abc import ABC
import random

class CrossShapeTask(SimpleGridDescriptionTask, ABC):
    def execute(self) -> tuple[np.ndarray, str]:
        # Generate a random grid size between 3 and 32
        grid_size = random.randint(3, 32)
        grid = np.zeros((grid_size, grid_size), dtype=int)

        # Generate a random color for the cross shape
        cross_color = random.randint(1, 9)

        # Calculate the center coordinates
        center_x, center_y = grid_size // 2, grid_size // 2

        # Draw the horizontal line
        grid[center_y, :] = cross_color

        # Draw the vertical line
        grid[:, center_x] = cross_color

        # Generate the description
        description = f"The grid contains a cross shape of {self.color_map[cross_color]} color centered at coordinates ({center_x}, {center_y})."

        return grid, description

if __name__ == "__main__":
    task = CrossShapeTask()
    grid, description = task.execute()
    print(description)
    task.visualize(grid, description)
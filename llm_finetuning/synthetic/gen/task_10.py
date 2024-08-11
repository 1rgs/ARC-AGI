from ..simple_grid_definition_task import SimpleGridDescriptionTask

import numpy as np
from random import randint, choice

class DiagonalLineTask(SimpleGridDescriptionTask):
    def execute(self) -> tuple[np.ndarray, str]:
        # Generate a random grid size between 3 and 32
        grid_size = randint(3, 32)
        grid = np.zeros((grid_size, grid_size), dtype=int)

        # Choose a random color for the diagonal line
        line_color = randint(1, 9)

        # Draw the diagonal line
        for i in range(grid_size):
            grid[i, i] = line_color
            grid[i, grid_size - i - 1] = line_color

        # Generate the description
        description = f"The grid contains a diagonal line of {self.color_map[line_color]} color spanning from the top-left corner (0, 0) to the bottom-right corner ({grid_size - 1}, {grid_size - 1}), and another diagonal line of the same color spanning from the top-right corner (0, {grid_size - 1}) to the bottom-left corner ({grid_size - 1}, 0). The two lines intersect at the center of the grid, forming an 'X' shape."

        return grid, description

if __name__ == "__main__":
    task = DiagonalLineTask()
    grid, description = task.execute()
    print(description)
    task.visualize(grid, description)
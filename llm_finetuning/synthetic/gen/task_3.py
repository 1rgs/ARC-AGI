from ..simple_grid_definition_task import SimpleGridDescriptionTask

import numpy as np
import random
from typing import Tuple

class SinglePyramidTask(SimpleGridDescriptionTask):
    def execute(self) -> Tuple[np.ndarray, str]:
        # Generate a random grid size between 3 and 32
        grid_size = random.randint(3, 32)
        grid = np.zeros((grid_size, grid_size), dtype=int)

        # Generate a random color for the pyramid
        color = random.randint(1, 9)

        # Generate a random position for the pyramid
        start_row = random.randint(0, grid_size - 1)
        start_col = random.randint(0, grid_size - 1)

        # Generate the pyramid
        for row in range(start_row, grid_size):
            num_cols = row - start_row + 1
            start_col_for_row = start_col - (num_cols // 2)
            end_col_for_row = start_col_for_row + num_cols
            grid[row, start_col_for_row:end_col_for_row] = color

        # Generate the description
        color_name = self.color_map[color].capitalize()
        description = f"The grid contains a single {color_name} pyramid. The pyramid is pointing upwards and is centered at coordinates ({start_row}, {start_col}). The base of the pyramid spans from column {start_col - (start_row + 1) // 2} to column {start_col + start_row // 2}, and the height of the pyramid is {start_row + 1} cells."

        return grid, description

if __name__ == "__main__":
    task = SinglePyramidTask()
    grid, description = task.execute()
    print(description)
    task.visualize(grid, description)
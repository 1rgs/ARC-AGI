from ..simple_grid_definition_task import SimpleGridDescriptionTask

import numpy as np
from typing import Tuple
from abc import ABC

class CheckerboardCornerTask(SimpleGridDescriptionTask, ABC):
    def execute(self) -> Tuple[np.ndarray, str]:
        # Generate a random grid size between 3 and 32
        grid_size = np.random.randint(3, 33)
        grid = np.zeros((grid_size, grid_size), dtype=int)

        # Randomly choose a corner to fill with the checkerboard pattern
        corner = np.random.randint(0, 4)

        # Define the checkerboard pattern
        checkerboard = np.array([[0, 1], [1, 0]])

        # Fill the chosen corner with the checkerboard pattern
        if corner == 0:  # Top-left corner
            grid[:2, :2] = checkerboard
        elif corner == 1:  # Top-right corner
            grid[:2, -2:] = checkerboard
        elif corner == 2:  # Bottom-left corner
            grid[-2:, :2] = checkerboard
        else:  # Bottom-right corner
            grid[-2:, -2:] = checkerboard

        # Generate the description
        corner_names = ["top-left", "top-right", "bottom-left", "bottom-right"]
        description = f"The grid contains a 2x2 checkerboard pattern in the {corner_names[corner]} corner. "
        description += "The checkerboard pattern consists of alternating black and red squares. "
        description += "The rest of the grid is filled with black cells."

        return grid, description

if __name__ == "__main__":
    task = CheckerboardCornerTask()
    grid, description = task.execute()
    print(description)
    task.visualize(grid, description)
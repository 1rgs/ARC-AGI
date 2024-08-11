from ..simple_grid_definition_task import SimpleGridDescriptionTask

import numpy as np
from typing import Tuple
from abc import ABC

class FourCornersTask(SimpleGridDescriptionTask, ABC):
    def execute(self) -> Tuple[np.ndarray, str]:
        # Generate a random grid size between 3 and 32
        grid_size = np.random.randint(3, 33)
        grid = np.zeros((grid_size, grid_size), dtype=int)

        # Choose four random colors for the corners
        corner_colors = np.random.choice(list(self.color_map.keys()), 4, replace=False)

        # Place the colored squares in the four corners
        grid[0, 0] = corner_colors[0]
        grid[0, -1] = corner_colors[1]
        grid[-1, 0] = corner_colors[2]
        grid[-1, -1] = corner_colors[3]

        # Generate the description
        description = f"The grid has four small squares of different colors placed in the four corners. "
        description += f"The top-left corner has a {self.color_map[corner_colors[0]]} square, "
        description += f"the top-right corner has a {self.color_map[corner_colors[1]]} square, "
        description += f"the bottom-left corner has a {self.color_map[corner_colors[2]]} square, "
        description += f"and the bottom-right corner has a {self.color_map[corner_colors[3]]} square."

        return grid, description

if __name__ == "__main__":
    task = FourCornersTask()
    grid, description = task.execute()
    print(description)
    task.visualize(grid, description)
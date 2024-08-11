from ..simple_grid_definition_task import SimpleGridDescriptionTask

import numpy as np
from random import randint, choice

class ColoredBorderTask(SimpleGridDescriptionTask):
    def execute(self) -> tuple[np.ndarray, str]:
        # Generate a random grid size between 3 and 32
        grid_size = randint(3, 32)
        grid = np.zeros((grid_size, grid_size), dtype=int)

        # Choose a random color for the border
        border_color = randint(1, 9)

        # Create the border
        for i in range(grid_size):
            grid[0, i] = border_color  # Top border
            grid[grid_size - 1, i] = border_color  # Bottom border
            grid[i, 0] = border_color  # Left border
            grid[i, grid_size - 1] = border_color  # Right border

        # Generate the description
        description = f"The grid has a single-cell wide border colored {self.color_map[border_color].capitalize()}. "
        description += "The inner area of the grid is empty and colored black."

        return grid, description

if __name__ == "__main__":
    task = ColoredBorderTask()
    grid, description = task.execute()
    print(description)
    task.visualize(grid, description)
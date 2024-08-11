from ..simple_grid_definition_task import SimpleGridDescriptionTask

import numpy as np
from abc import ABC
import random

class LShapeTask(SimpleGridDescriptionTask, ABC):
    def execute(self) -> tuple[np.ndarray, str]:
        # Generate a random grid size between 3 and 32
        grid_size = random.randint(3, 32)
        grid = np.zeros((grid_size, grid_size), dtype=int)

        # Generate a random color for the L-shape
        color = random.randint(1, 9)

        # Randomly choose a corner to place the L-shape
        corners = [(0, 0), (0, grid_size - 1), (grid_size - 1, 0), (grid_size - 1, grid_size - 1)]
        corner = random.choice(corners)

        # Determine the orientation of the L-shape
        orientations = ['top-left', 'top-right', 'bottom-left', 'bottom-right']
        orientation = random.choice(orientations)

        # Place the L-shape in the chosen corner and orientation
        if orientation == 'top-left':
            grid[corner[0], corner[1]:corner[1] + 3] = color
            grid[corner[0]:corner[0] + 2, corner[1]] = color
        elif orientation == 'top-right':
            grid[corner[0], corner[1] - 2:corner[1] + 1] = color
            grid[corner[0]:corner[0] + 2, corner[1]] = color
        elif orientation == 'bottom-left':
            grid[corner[0], corner[1]:corner[1] + 3] = color
            grid[corner[0] - 1:corner[0] + 1, corner[1]] = color
        else:  # bottom-right
            grid[corner[0], corner[1] - 2:corner[1] + 1] = color
            grid[corner[0] - 1:corner[0] + 1, corner[1]] = color

        # Generate the description
        color_name = self.color_map[color].capitalize()
        corner_description = f"({corner[0]}, {corner[1]})"
        orientation_description = orientation.replace('-', ' ')

        description = f"The grid contains an L-shaped pattern of {color_name} color positioned in the {orientation_description} corner at {corner_description}."

        return grid, description

if __name__ == "__main__":
    task = LShapeTask()
    grid, description = task.execute()
    print(description)
    task.visualize(grid, description)
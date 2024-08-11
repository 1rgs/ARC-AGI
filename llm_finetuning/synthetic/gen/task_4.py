from ..simple_grid_definition_task import SimpleGridDescriptionTask

import numpy as np
import random
from typing import Tuple

class SingleRightTriangleTask(SimpleGridDescriptionTask):
    def execute(self) -> Tuple[np.ndarray, str]:
        # Generate a random grid size between 3 and 32
        grid_size = random.randint(3, 32)
        grid = np.zeros((grid_size, grid_size), dtype=int)

        # Generate a random color for the triangle
        triangle_color = random.randint(1, 9)

        # Generate random coordinates for the triangle's base
        base_x = random.randint(0, grid_size - 1)
        base_y = random.randint(0, grid_size - 1)

        # Determine the triangle's orientation (top-left, top-right, bottom-left, bottom-right)
        orientation = random.randint(0, 3)

        # Draw the triangle based on the orientation
        if orientation == 0:  # Top-left
            for i in range(base_x + 1):
                for j in range(base_y + 1 - i):
                    grid[j, i] = triangle_color
        elif orientation == 1:  # Top-right
            for i in range(grid_size - base_x):
                for j in range(base_y + 1 - i):
                    grid[j, base_x + i] = triangle_color
        elif orientation == 2:  # Bottom-left
            for i in range(base_x + 1):
                for j in range(base_y, grid_size):
                    grid[j, i] = triangle_color
        else:  # Bottom-right
            for i in range(grid_size - base_x):
                for j in range(base_y, grid_size):
                    grid[j, base_x + i] = triangle_color

        # Generate the description
        color_name = self.color_map[triangle_color].capitalize()
        if orientation == 0:
            description = f"A {color_name} right-angled triangle is positioned in the top-left corner of the grid, with its base spanning from (0, 0) to ({base_x}, 0) and its height extending to ({base_x}, {base_y})."
        elif orientation == 1:
            description = f"A {color_name} right-angled triangle is positioned in the top-right corner of the grid, with its base spanning from ({grid_size - base_x - 1}, 0) to ({grid_size - 1}, 0) and its height extending to ({grid_size - base_x - 1}, {base_y})."
        elif orientation == 2:
            description = f"A {color_name} right-angled triangle is positioned in the bottom-left corner of the grid, with its base spanning from (0, {grid_size - base_y - 1}) to ({base_x}, {grid_size - base_y - 1}) and its height extending to ({base_x}, {grid_size - 1})."
        else:
            description = f"A {color_name} right-angled triangle is positioned in the bottom-right corner of the grid, with its base spanning from ({grid_size - base_x - 1}, {grid_size - base_y - 1}) to ({grid_size - 1}, {grid_size - base_y - 1}) and its height extending to ({grid_size - 1}, {grid_size - 1})."

        return grid, description

if __name__ == "__main__":
    task = SingleRightTriangleTask()
    grid, description = task.execute()
    print(description)
    task.visualize(grid, description)
from ..simple_grid_definition_task import SimpleGridDescriptionTask

import numpy as np
from abc import ABC
import random

class TShapeTask(SimpleGridDescriptionTask):
    def execute(self) -> tuple[np.ndarray, str]:
        # Generate a random grid size between 3 and 32
        grid_size = random.randint(3, 32)
        grid = np.zeros((grid_size, grid_size), dtype=int)

        # Generate a random color for the T-shape
        color = random.randint(1, 9)

        # Determine the position and orientation of the T-shape
        edge = random.randint(0, 3)  # 0: top, 1: right, 2: bottom, 3: left
        if edge == 0:  # Top edge
            start_x = 0
            start_y = random.randint(0, grid_size - 3)
            length_x = grid_size
            length_y = 3
            stem_x = random.randint(0, grid_size - 2)
            stem_y = start_y + 1
        elif edge == 1:  # Right edge
            start_x = random.randint(0, grid_size - 3)
            start_y = 0
            length_x = 3
            length_y = grid_size
            stem_x = start_x
            stem_y = random.randint(0, grid_size - 2)
        elif edge == 2:  # Bottom edge
            start_x = 0
            start_y = random.randint(0, grid_size - 3)
            length_x = grid_size
            length_y = 3
            stem_x = random.randint(0, grid_size - 2)
            stem_y = start_y
        else:  # Left edge
            start_x = random.randint(0, grid_size - 3)
            start_y = 0
            length_x = 3
            length_y = grid_size
            stem_x = start_x + 1
            stem_y = random.randint(0, grid_size - 2)

        # Draw the T-shape on the grid
        grid[start_y:start_y + length_y, start_x:start_x + length_x] = color
        grid[stem_y, stem_x] = color

        # Generate the description
        if edge == 0:
            description = f"A {self.color_map[color].capitalize()} T-shape is positioned along the top edge of the grid. The horizontal bar spans the entire width of the grid, while the vertical stem extends downwards from the center of the horizontal bar."
        elif edge == 1:
            description = f"A {self.color_map[color].capitalize()} T-shape is positioned along the right edge of the grid. The vertical bar spans the entire height of the grid, while the horizontal stem extends leftwards from the center of the vertical bar."
        elif edge == 2:
            description = f"A {self.color_map[color].capitalize()} T-shape is positioned along the bottom edge of the grid. The horizontal bar spans the entire width of the grid, while the vertical stem extends upwards from the center of the horizontal bar."
        else:
            description = f"A {self.color_map[color].capitalize()} T-shape is positioned along the left edge of the grid. The vertical bar spans the entire height of the grid, while the horizontal stem extends rightwards from the center of the vertical bar."

        return grid, description

if __name__ == "__main__":
    task = TShapeTask()
    grid, description = task.execute()
    print(description)
    task.visualize(grid, description)
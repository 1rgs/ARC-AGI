from ..simple_grid_definition_task import SimpleGridDescriptionTask

import numpy as np
from abc import ABC
from typing import Tuple

class ParallelLinesTask(SimpleGridDescriptionTask, ABC):
    def execute(self) -> Tuple[np.ndarray, str]:
        # Generate a random grid size between 3 and 32
        grid_size = np.random.randint(3, 33)
        grid = np.zeros((grid_size, grid_size), dtype=int)

        # Choose two random colors for the lines
        line_colors = np.random.choice(list(self.color_map.keys()), 2, replace=False)

        # Draw the first line
        line_width = np.random.randint(1, grid_size // 2)
        line_start = np.random.randint(0, grid_size - line_width)
        line_end = line_start + line_width
        grid[line_start:line_end, :] = line_colors[0]

        # Draw the second parallel line
        line_spacing = np.random.randint(1, grid_size - line_width)
        line_start = line_end + line_spacing
        line_end = line_start + line_width
        grid[line_start:line_end, :] = line_colors[1]

        # Generate the description
        color1 = self.color_map[line_colors[0]].capitalize()
        color2 = self.color_map[line_colors[1]].capitalize()
        description = f"The grid contains two parallel lines of different colors. The first line is {color1} and spans from row {line_start} to row {line_end - 1} across the entire grid. The second line is {color2} and spans from row {line_start + line_spacing} to row {line_end + line_spacing - 1}, also across the entire grid. The lines are {line_spacing} cells apart."

        return grid, description

if __name__ == "__main__":
    task = ParallelLinesTask()
    grid, description = task.execute()
    print(description)
    task.visualize(grid, description)
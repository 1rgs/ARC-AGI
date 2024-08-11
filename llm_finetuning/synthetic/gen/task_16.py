from ..simple_grid_definition_task import SimpleGridDescriptionTask

import numpy as np
from typing import Tuple
from abc import ABC

class CentralPlusTask(SimpleGridDescriptionTask, ABC):
    def execute(self) -> Tuple[np.ndarray, str]:
        # Generate a random grid size between 3 and 32
        grid_size = np.random.randint(3, 33)
        
        # Initialize the grid with a random color
        grid = np.full((grid_size, grid_size), np.random.randint(1, 10))
        
        # Choose a random color for the plus sign
        plus_color = np.random.randint(1, 10)
        while plus_color == grid[grid_size // 2, grid_size // 2]:
            plus_color = np.random.randint(1, 10)
        
        # Draw the plus sign
        grid[:, grid_size // 2] = plus_color
        grid[grid_size // 2, :] = plus_color
        
        # Generate the description
        description = f"The grid contains a plus sign of {self.color_map[plus_color]} color spanning across the center row and column. " \
                      f"The center cell is filled with {self.color_map[grid[grid_size // 2, grid_size // 2]]} color."
        
        return grid, description

if __name__ == "__main__":
    task = CentralPlusTask()
    grid, description = task.execute()
    print(description)
    task.visualize(grid, description)
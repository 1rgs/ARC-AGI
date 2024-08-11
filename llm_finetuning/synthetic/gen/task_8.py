from ..simple_grid_definition_task import SimpleGridDescriptionTask

import numpy as np
from abc import ABC
import random

class SquareAndTriangleTask(SimpleGridDescriptionTask):
    def execute(self) -> tuple[np.ndarray, str]:
        # Generate a random grid size between 3 and 32
        grid_size = random.randint(3, 32)
        grid = np.zeros((grid_size, grid_size), dtype=int)

        # Generate random colors for the square and triangle
        square_color = random.randint(1, 9)
        triangle_color = random.randint(1, 9)
        while triangle_color == square_color:
            triangle_color = random.randint(1, 9)

        # Generate random positions for the square and triangle
        square_x = random.randint(0, grid_size - 3)
        square_y = random.randint(0, grid_size - 3)
        triangle_x = random.randint(0, grid_size - 3)
        triangle_y = random.randint(0, grid_size - 3)

        # Check if the square and triangle overlap
        overlap = False
        if (
            (square_x <= triangle_x <= square_x + 2 and square_y <= triangle_y <= square_y + 2)
            or (triangle_x <= square_x <= triangle_x + 2 and triangle_y <= square_y <= triangle_y + 2)
            or (square_x <= triangle_x <= square_x + 2 and triangle_y <= square_y + 2 <= triangle_y + 2)
            or (triangle_x <= square_x <= triangle_x + 2 and square_y <= triangle_y + 2 <= square_y + 2)
        ):
            overlap = True

        # Draw the square
        grid[square_y : square_y + 3, square_x : square_x + 3] = square_color

        # Draw the triangle
        grid[triangle_y, triangle_x : triangle_x + 3] = triangle_color
        grid[triangle_y + 1, triangle_x : triangle_x + 2] = triangle_color
        grid[triangle_y + 2, triangle_x] = triangle_color

        # Generate the description
        square_color_name = self.color_map[square_color]
        triangle_color_name = self.color_map[triangle_color]
        description = f"The grid contains a {square_color_name} square spanning from ({square_x}, {square_y}) to ({square_x + 2}, {square_y + 2}) and a {triangle_color_name} right triangle spanning from ({triangle_x}, {triangle_y}) to ({triangle_x + 2}, {triangle_y + 2})."
        if overlap:
            description += " The square and triangle overlap."

        return grid, description

if __name__ == "__main__":
    task = SquareAndTriangleTask()
    grid, description = task.execute()
    print(description)
    task.visualize(grid, description)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from abc import ABC, abstractmethod


class SimpleGridDescriptionTask(ABC):
    def __init__(self):

        self.color_map = {
            0: "black",
            1: "red",
            2: "green",
            3: "blue",
            4: "yellow",
            5: "purple",
            6: "orange",
            7: "pink",
            8: "cyan",
            9: "brown",
        }

    def execute(self) -> tuple[np.ndarray, str]:
        """
        Generate a grid with a specific pattern and provide a detailed human-like description of it.

        This method creates a 2D numpy array representing a colored grid and generates
        a natural language description of the pattern observed in the grid. The grid
        and its description are designed to be used as synthetic data for training
        language models to recognize and describe visual patterns.

        Returns:
            tuple[np.ndarray, str]: A tuple containing two elements:
                1. np.ndarray: A 2D numpy array representing the generated grid.
                - Shape: (n, n) where 3 <= n <= 32
                - Values: Integers from 0 to 9, each representing a distinct color
                2. str: A human-like description of the pattern observed in the grid.

        Description Format:
            The description should be in plain English, similar to how a layman would
            describe the pattern. It should include:
            - Identification of shapes, patterns, or structures in the grid
            - Specific colors of the identified elements
            - Precise positions of elements (e.g., "at coordinates (2, 3)", "spanning from (1, 1) to (4, 5)")
            - Size of elements where applicable (e.g., "a 3x4 rectangle")
            - Relationships between elements, such as overlapping or adjacency
            - Any discernible patterns in positioning or color distribution
            - If no clear pattern exists, explicitly state this
            - For amorphous shapes, provide an ASCII representation of the shape

        Examples of Descriptions:
            1. "The grid contains three rectangles: a red 3x2 rectangle at (0, 0), a blue 2x4 rectangle
               at (2, 3), and a green 1x3 rectangle at (4, 1). The blue and green rectangles overlap
               slightly. There doesn't seem to be any particular pattern in their positioning."

            2. "I see a series of diagonal stripes alternating between yellow and purple. Each stripe
               is 2 cells wide. The pattern starts at the top-left corner and continues to the
               bottom-right, creating a consistent diagonal pattern across the entire grid."

            3. "The grid is divided into four equal quadrants. The top-left quadrant contains a red
               circle, the top-right a blue square, the bottom-left a green triangle, and the
               bottom-right a yellow star. Each shape is centered within its quadrant, creating
               a symmetrical arrangement."

            4. "An amorphous cyan shape dominates the center of the grid. It has an irregular outline
               that can be represented as:
                  oooxx
                  ooxxx
                  oxxxx
                  ooxxx
                  oooxo
               Small orange dots are scattered randomly around this shape, avoiding overlap."

            5. "The grid features a concentric pattern of squares. Starting from the outermost, the
               squares are colored red, blue, green, and yellow. Each square is 2 cells thick.
               The pattern is perfectly centered in the grid, creating a symmetrical design."

        Note:
            - The method should handle various patterns including but not limited to:
              squares, rectangles, triangles, circles, stripes, concentric shapes, and scattered elements.
            - Descriptions should be precise about positions, sizes, and relationships between elements.
            - If there's a discernible pattern in positioning or color distribution, describe it.
            - If there's no clear pattern, explicitly state this observation.
            - For complex or amorphous shapes, provide an ASCII representation to clarify the structure.
            - The description should be detailed enough for a human to visualize the pattern
              without seeing the actual grid.
            - Avoid using technical jargon; descriptions should be accessible to non-experts.

        This method is part of a larger system designed to generate synthetic tasks
        for training language models to recognize and describe patterns in grids.
        """
        # Implementation goes here
        pass

    def visualize(self, grid: np.ndarray, description: str):
        fig, ax = plt.subplots(figsize=(10, 10))

        # Determine the unique values in the grid
        unique_values = np.unique(grid)
        n_colors = len(unique_values)

        # Create a mapping of grid values to color indices
        value_to_index = {val: i for i, val in enumerate(unique_values)}

        # Create a new grid with mapped indices
        indexed_grid = np.vectorize(value_to_index.get)(grid)

        # Create a custom colormap
        colors = [self.color_map[i] for i in unique_values]
        cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors, N=n_colors)

        x_grid_size, y_grid_size = grid.shape

        # Display the grid
        cax = ax.imshow(indexed_grid, cmap=cmap, interpolation="nearest")
        ax.set_title("Generated Grid")
        ax.set_xticks(np.arange(-0.5, x_grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, y_grid_size, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", size=0)

        # Create a colorbar
        cbar = fig.colorbar(cax, ax=ax, ticks=range(n_colors))
        cbar.set_ticklabels([self.color_map[i].capitalize() for i in unique_values])

        ax.set_xticks(np.arange(0, x_grid_size, 1))
        ax.set_yticks(np.arange(0, y_grid_size, 1))

        # Add description as text below the grid
        plt.figtext(
            0.5, 0.01, description, wrap=True, horizontalalignment="center", fontsize=12
        )

        # Disable cursor value display
        ax.format_cursor_data = lambda data: ""

        plt.tight_layout()
        plt.show()

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import matplotlib.colors as mcolors


# seed = 1
# np.random.seed(seed)

# grids can be 2-32 on any side

MAX_GRID_SIZE = 21
MIN_GRID_SIZE = 2


def generate_shape(grid_size, n, return_shape_type=False, allowed_shapes=None):
    shape_type = np.random.choice(
        [
            a
            for a in [
                "square",
                "rectangle",
                "connected",
                "hollow_square",
                "plus",
                "L-shape",
                "T-shape",
            ]
            if allowed_shapes is None or a in allowed_shapes
        ]
    )
    shape_grid = np.zeros((grid_size, grid_size), dtype=int)

    if shape_type == "square":
        size = np.random.randint(n, grid_size // 2 + 1)
        start_x = np.random.randint(0, max(1, grid_size - size))
        start_y = np.random.randint(0, max(1, grid_size - size))
        shape_grid[start_x : start_x + size, start_y : start_y + size] = 1

    elif shape_type == "rectangle":
        height = np.random.randint(n, max(n + 1, grid_size // 2 + 1))
        width = np.random.randint(n, max(n + 1, grid_size // 2 + 1))
        start_x = np.random.randint(0, max(1, grid_size - height))
        start_y = np.random.randint(0, max(1, grid_size - width))
        shape_grid[start_x : start_x + height, start_y : start_y + width] = 1

    elif shape_type == "connected":
        component_size = max(n, np.random.randint(n, max(n + 1, grid_size // 2 + 1)))
        start_x = np.random.randint(0, max(1, grid_size - component_size))
        start_y = np.random.randint(0, max(1, grid_size - component_size))
        shape_grid[
            start_x : start_x + component_size, start_y : start_y + component_size
        ] = 1
        num_components = np.random.randint(2, 5)
        for _ in range(num_components):
            offset_x = np.random.randint(-component_size // 2, component_size // 2)
            offset_y = np.random.randint(-component_size // 2, component_size // 2)
            sub_size = np.random.randint(n, max(n + 1, component_size))
            sub_start_x = np.clip(start_x + offset_x, 0, grid_size - sub_size)
            sub_start_y = np.clip(start_y + offset_y, 0, grid_size - sub_size)
            shape_grid[
                sub_start_x : sub_start_x + sub_size,
                sub_start_y : sub_start_y + sub_size,
            ] = 1

    elif shape_type == "hollow_square":
        size = np.random.randint(n, max(n + 1, grid_size // 2 + 1))
        start_x = np.random.randint(0, max(1, grid_size - size))
        start_y = np.random.randint(0, max(1, grid_size - size))
        shape_grid[start_x : start_x + size, start_y] = 1
        shape_grid[start_x, start_y : start_y + size] = 1
        shape_grid[start_x : start_x + size, start_y + size - 1] = 1
        shape_grid[start_x + size - 1, start_y : start_y + size] = 1

    elif shape_type == "plus":
        size = np.random.randint(n, max(n + 1, grid_size // 4 + 1))
        center_x = np.random.randint(size, max(size + 1, grid_size - size))
        center_y = np.random.randint(size, max(size + 1, grid_size - size))
        shape_grid[center_x - size : center_x + size + 1, center_y] = 1
        shape_grid[center_x, center_y - size : center_y + size + 1] = 1

    elif shape_type == "L-shape":
        height = np.random.randint(n, max(n + 1, grid_size // 2 + 1))
        width = np.random.randint(n, max(n + 1, grid_size // 2 + 1))
        start_x = np.random.randint(0, max(1, grid_size - height))
        start_y = np.random.randint(0, max(1, grid_size - width))
        shape_grid[start_x : start_x + height, start_y] = 1
        shape_grid[start_x + height - 1, start_y : start_y + width] = 1

    elif shape_type == "T-shape":
        height = np.random.randint(n, max(n + 1, grid_size // 2 + 1))
        width = np.random.randint(n, max(n + 1, grid_size // 2 + 1))
        start_x = np.random.randint(0, max(1, grid_size - height))
        start_y = np.random.randint(0, max(1, grid_size - width))
        shape_grid[start_x, start_y : start_y + width] = 1
        shape_grid[start_x : start_x + height, start_y + width // 2] = 1

    if return_shape_type:
        return shape_grid, shape_type
    return shape_grid


from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


from abc import ABC, abstractmethod
import numpy as np


class GridTask(ABC):
    """
    Abstract base class for grid-based tasks.

    This class defines the interface for creating grid-based tasks, including
    methods for generating sample grids, executing transformations, and
    providing descriptions of the grids and the transformation patterns.
    """

    color_map = {
        0: "black",
        1: "red",
        2: "green",
        3: "blue",
        4: "yellow",
        5: "purple",
        6: "orange",
    }

    def __init__(self):
        """
        Initialize the GridTask.

        Subclasses should define task-specific attributes here, such as
        grid size, colors used, and any other parameters needed for the task.
        """
        pass

    @abstractmethod
    def sample(self) -> np.ndarray:
        """
        Generate a sample input grid for the task.

        Returns:
            np.ndarray: A 2D numpy array representing the input grid.

        This method should create and return a grid that serves as the
        starting point for the task. The specific content and structure
        of the grid will depend on the particular task being implemented.
        """
        pass

    @abstractmethod
    def execute(self, grid: np.ndarray) -> (np.ndarray, str):
        """
        Execute the task on the given input grid.

        Args:
            grid (np.ndarray): The input grid to transform.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: The transformed output grid.
                - str: A string combining the description and pattern explanation.

        This method should apply the task's transformation to the input grid
        and return the result. It should also generate a description of the
        grids and an explanation of the transformation pattern, combining
        them into a single string.
        """
        pass

    @abstractmethod
    def generate_description(
        self, input_grid: np.ndarray, output_grid: np.ndarray
    ) -> str:
        """
        Generate a description of the input and output grids.

        Args:
            input_grid (np.ndarray): The original input grid.
            output_grid (np.ndarray): The transformed output grid.

        Returns:
            str: A detailed description of both grids and their differences.

        This method should provide:
        1. A description of the input grid, including its size, colors used,
           and any notable features or patterns.
        2. A description of the output grid, highlighting how it differs
           from the input grid.
        3. An explicit comparison between the input and output grids,
           detailing the changes that occurred during the transformation.

        The description should be clear, concise, and focus on observable
        features of the grids, avoiding explanations of the transformation
        process itself.
        """
        pass

    @abstractmethod
    def generate_pattern_description(self) -> str:
        """
        Generate a general description of the transformation pattern.

        Returns:
            str: A general explanation of how the input grid is transformed
                 into the output grid.

        This method should:
        1. Provide a high-level, general description of how the input grid
           is transformed into the output grid.
        2. Focus solely on the differences between the input and output grids,
           not on the process of creating the initial grid.
        3. Avoid referencing specific colors, sizes, or other instance-specific
           details. The description should apply to all possible instances of
           the task.
        4. Describe the nature of the changes (e.g., color replacements, shifts,
           rotations) without detailing the exact steps of the transformation.
        5. Highlight any consistent patterns or rules in the transformation
           that apply across all instances of the task.

        The description should be detailed enough to give a clear understanding
        of the transformation, but general enough to apply to all instances of
        the task.

        Start with "The pattern that transforms the input grid to the output grid involves..."
        """
        pass


class ColorReplacementTask(GridTask):
    def __init__(self):
        super().__init__()
        self.grid_size = np.random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE + 1)
        self.colors = np.random.choice(
            list(range(1, 7)), size=np.random.randint(2, 7), replace=False
        )

    def sample(self):
        return np.random.choice(self.colors, size=(self.grid_size, self.grid_size))

    def execute(self, grid: np.ndarray) -> (np.ndarray, str):
        self.color1, self.color2 = np.random.choice(self.colors, size=2, replace=False)
        new_grid = np.copy(grid)
        new_grid[grid == self.color1] = self.color2
        description = self.generate_description(grid, new_grid)
        pattern = self.generate_pattern_description()
        return new_grid, f"{description}\n\n{pattern}"

    def generate_description(
        self, input_grid: np.ndarray, output_grid: np.ndarray
    ) -> str:
        def describe_grid(grid, grid_name):
            unique, counts = np.unique(grid, return_counts=True)
            color_counts = dict(zip(unique, counts))
            total_cells = self.grid_size * self.grid_size

            color_descriptions = []
            for color in sorted(color_counts.keys()):
                percentage = (color_counts[color] / total_cells) * 100
                color_descriptions.append(
                    f"{self.color_map[color]} ({percentage:.1f}%)"
                )

            color_list = (
                ", ".join(color_descriptions[:-1]) + f" and {color_descriptions[-1]}"
                if len(color_descriptions) > 1
                else color_descriptions[0]
            )

            return (
                f"The {grid_name} is a {self.grid_size}x{self.grid_size} square filled with colored cells. "
                f"The colors present are {color_list}. "
                f"The cells are randomly distributed, creating a colorful mosaic-like pattern."
            )

        input_description = describe_grid(input_grid, "input grid")
        output_description = describe_grid(output_grid, "output grid")

        difference = (
            f"The main difference between the input and output grids is that "
            f"all {self.color_map[self.color1]} cells in the input grid have been replaced with {self.color_map[self.color2]} in the output grid. "
            f"This change affects {np.sum(input_grid == self.color1)} cells, which is "
            f"{(np.sum(input_grid == self.color1) / (self.grid_size * self.grid_size) * 100):.1f}% of the total grid. "
            f"The positions and colors of all other cells remain unchanged."
        )

        return f"{input_description}\n\n{output_description}\n\n{difference}"

    def generate_pattern_description(self) -> str:
        return f"""
The pattern that transforms the input grid to the output grid involves a specific color replacement:

1. Color Identification:
   The transformation identifies two specific colors in the grid: {self.color_map[self.color1]} (Color A) and {self.color_map[self.color2]} (Color B).

2. Color Replacement:
   Every cell in the grid that is Color A ({self.color_map[self.color1]}) changes to Color B ({self.color_map[self.color2]}).
   Cells that are not Color A remain unchanged.

3. Grid Structure:
   The overall structure of the grid (size and position of cells) remains the same.
   Only the colors of specific cells change.

4. Consistency:
   This color replacement is applied consistently across the entire grid.
   Every occurrence of Color A, regardless of its position or surrounding colors, is changed to Color B.

5. Other Colors:
   All other colors in the grid that are neither Color A nor Color B remain untouched.

This transformation results in a grid that looks similar to the input, but with {self.color_map[self.color1]} completely replaced by {self.color_map[self.color2]}, altering the overall color distribution and appearance of the grid.
"""


class ShiftGridTask(GridTask):
    def __init__(self):
        super().__init__()
        self.grid_size = np.random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE + 1)
        self.colors = np.random.choice(
            list(range(1, 7)), size=np.random.randint(2, 7), replace=False
        )
        self.fill_color = np.random.choice(list(range(1, 7)))
        self.directions = ["up", "down", "left", "right"]

    def sample(self):
        return np.random.choice(self.colors, size=(self.grid_size, self.grid_size))

    def execute(self, grid: np.ndarray) -> (np.ndarray, str):
        self.n = np.random.randint(1, self.grid_size // 2)
        self.direction = np.random.choice(self.directions)
        new_grid = np.full(grid.shape, self.fill_color)

        if self.direction == "right":
            new_grid[:, self.n :] = grid[:, : -self.n]
        elif self.direction == "left":
            new_grid[:, : -self.n] = grid[:, self.n :]
        elif self.direction == "down":
            new_grid[self.n :, :] = grid[: -self.n, :]
        elif self.direction == "up":
            new_grid[: -self.n, :] = grid[self.n :, :]

        description = self.generate_description(grid, new_grid)
        pattern = self.generate_pattern_description()
        return new_grid, f"{description}\n\n{pattern}"

    def generate_description(
        self, input_grid: np.ndarray, output_grid: np.ndarray
    ) -> str:
        def describe_grid(grid, grid_name):
            unique, counts = np.unique(grid, return_counts=True)
            color_counts = dict(zip(unique, counts))
            total_cells = self.grid_size * self.grid_size

            color_descriptions = []
            for color in sorted(color_counts.keys()):
                percentage = (color_counts[color] / total_cells) * 100
                color_descriptions.append(
                    f"{self.color_map[color]} ({percentage:.1f}%)"
                )

            color_list = (
                ", ".join(color_descriptions[:-1]) + f" and {color_descriptions[-1]}"
                if len(color_descriptions) > 1
                else color_descriptions[0]
            )

            return (
                f"The {grid_name} is a {self.grid_size}x{self.grid_size} square filled with colored cells. "
                f"The colors present are {color_list}. "
                f"The cells are randomly distributed, creating a colorful mosaic-like pattern."
            )

        input_description = describe_grid(input_grid, "input grid")
        output_description = describe_grid(output_grid, "output grid")

        shift_description = (
            f"The grid has been shifted {self.n} cells to the {self.direction}. "
        )
        fill_description = (
            f"The {self.color_map[self.fill_color]} color has been used to fill the {self.n} {'rows' if self.direction in ['up', 'down'] else 'columns'} "
            f"that were left empty by the shift on the {'bottom' if self.direction == 'up' else 'top' if self.direction == 'down' else 'right' if self.direction == 'left' else 'left'} side of the grid."
        )

        difference = f"""
The main differences between the input and output grids are:
1. {shift_description}
2. {fill_description}
3. {self.n * self.grid_size} cells ({(self.n * self.grid_size / (self.grid_size * self.grid_size) * 100):.1f}% of the grid) have been replaced with the fill color.
4. The overall color distribution has changed due to the shift and fill operations.
"""

        return f"{input_description}\n\n{output_description}\n\n{difference}"

    def generate_pattern_description(self) -> str:
        return f"""
The pattern that transforms the input grid to the output grid involves a shift operation and color filling. Here's a general description of this pattern:

1. Shift Direction and Magnitude:
   The entire grid content is shifted {self.n} cells to the {self.direction}.

2. Content Preservation:
   The shifted portion of the grid maintains its original color pattern and arrangement.
   However, part of the original grid content is moved out of view due to the shift.

3. Empty Space Filling:
   The space left empty by the shift ({"rows" if self.direction in ['up', 'down'] else "columns"} on the {"bottom" if self.direction == 'up' else "top" if self.direction == 'down' else "right" if self.direction == 'left' else "left"} side)
   is filled with a single color ({self.color_map[self.fill_color]}).

4. Grid Size Consistency:
   The overall dimensions of the grid remain unchanged ({self.grid_size}x{self.grid_size}).

5. Color Distribution Change:
   The shift and fill operations alter the overall color distribution in the grid.
   {self.n * self.grid_size} cells ({(self.n * self.grid_size / (self.grid_size * self.grid_size) * 100):.1f}% of the grid) are replaced with the fill color.

This transformation results in a grid that partially resembles the input, but with a notable shift in pattern and a new block of uniform color introduced to one side.
"""


class DrawSquaresTask(GridTask):
    def __init__(self, grid_size, colors, num_squares):
        super().__init__()
        self.grid_size = grid_size
        self.colors = colors
        self.num_squares = num_squares

    def sample(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for _ in range(self.num_squares):
            size = np.random.randint(1, max(2, self.grid_size // 2))
            color = np.random.choice(self.colors)
            start_x = np.random.randint(0, self.grid_size - size)
            start_y = np.random.randint(0, self.grid_size - size)
            grid[start_x : start_x + size, start_y : start_y + size] = color
        return grid

    def execute(self, grid: np.ndarray) -> (np.ndarray, str):
        self.color1, self.color2 = np.random.choice(self.colors, size=2, replace=False)
        new_grid = np.copy(grid)
        new_grid[grid == self.color1] = self.color2
        description = self.generate_description(grid, new_grid)
        pattern = self.generate_pattern_description()
        return new_grid, f"{description}\n\n{pattern}"

    def generate_description(
        self, input_grid: np.ndarray, output_grid: np.ndarray
    ) -> str:
        def describe_grid(grid, grid_name):
            unique, counts = np.unique(grid, return_counts=True)
            color_counts = dict(zip(unique, counts))
            total_cells = self.grid_size * self.grid_size

            color_descriptions = []
            for color in sorted(color_counts.keys()):
                if color != 0:  # Exclude the background color
                    percentage = (color_counts[color] / total_cells) * 100
                    color_descriptions.append(
                        f"{self.color_map[color]} ({color_counts[color]} cells, {percentage:.1f}%)"
                    )

            color_list = (
                ", ".join(color_descriptions[:-1]) + f" and {color_descriptions[-1]}"
                if len(color_descriptions) > 1
                else color_descriptions[0]
            )

            squares = []
            for color in self.colors:
                if color in color_counts:
                    squares.append(
                        f"a {self.color_map[color]} square with {color_counts[color]} cells"
                    )

            square_description = (
                ", ".join(squares[:-1]) + f" and {squares[-1]}"
                if len(squares) > 1
                else squares[0]
            )

            return f"""Looking at the {grid_name}, we see a {self.grid_size}x{self.grid_size} square grid. 
The grid has a black background, which forms the majority of the space.
There are {self.num_squares} colored square{'s' if self.num_squares > 1 else ''} on this background: {square_description}.
The colors present in this grid are {color_list}.
The colored cells occupy {np.sum(grid > 0)} cells ({(np.sum(grid > 0) / total_cells) * 100:.1f}% of the grid), 
while the remaining {np.sum(grid == 0)} cells ({(np.sum(grid == 0) / total_cells) * 100:.1f}% of the grid) are black background.
The squares appear to be randomly positioned and may have different sizes."""

        input_description = describe_grid(input_grid, "input grid")
        output_description = describe_grid(output_grid, "output grid")

        changed_cells = np.sum(input_grid == self.color1)
        percentage_changed = (changed_cells / (self.grid_size * self.grid_size)) * 100

        difference = f"""
When comparing the input and output grids, we notice several key differences:

1. The most striking change is that all cells that were {self.color_map[self.color1]} in the input grid have been changed to {self.color_map[self.color2]} in the output grid.
2. This color replacement affects {changed_cells} cells, which is {percentage_changed:.1f}% of the total grid.
3. The positions and sizes of all squares remain exactly the same; only the color of one type of square has changed.
4. The overall structure and layout of the grid remains unchanged, with the same number and arrangement of squares.
5. The black background cells are unaffected by this transformation.
6. If there were any {self.color_map[self.color2]} squares in the input grid, they now appear identical to the transformed {self.color_map[self.color1]} squares in the output grid.
7. This change might affect how we perceive the relationships between squares:
   - Squares that were distinct in the input grid might now appear merged if they're adjacent.
   - New color contrasts might have formed between the changed squares and other colored squares.

This transformation results in a grid where one color has been completely replaced by another, potentially altering the visual grouping and separation of squares within the grid.
"""

        return f"{input_description}\n\n{output_description}\n\n{difference}"

    def generate_pattern_description(self) -> str:
        return f"""
After carefully observing the changes between the input and output grids, we can describe the pattern of transformation as follows:

1. Grid Structure:
   - We start with a {self.grid_size}x{self.grid_size} grid.
   - The grid contains {self.num_squares} colored square{'s' if self.num_squares > 1 else ''} on a black background.
   - These squares have various sizes and are randomly positioned.

2. Color Identification:
   - The transformation identifies two specific colors: {self.color_map[self.color1]} (source color) and {self.color_map[self.color2]} (target color).

3. Color Replacement:
   - Every cell in the grid that is the source color ({self.color_map[self.color1]}) is changed to the target color ({self.color_map[self.color2]}).
   - This replacement is applied consistently across the entire grid.

4. Shape Preservation:
   - The sizes, positions, and shapes of all squares remain unchanged.
   - Only the color of the affected squares changes; their structure is maintained.

5. Selective Transformation:
   - Only squares of the source color are affected.
   - Squares of other colors and the black background remain unchanged.

6. Complete Replacement:
   - After the transformation, there are no cells left with the source color in the grid.
   - All instances of the source color have been replaced by the target color.

7. Potential Visual Effects:
   - This transformation can lead to visual merging of previously distinct squares if they become the same color.
   - It may also create new color contrasts between the changed squares and unchanged squares of other colors.

8. Consistency:
   - The same color replacement rule is applied uniformly across the entire grid, regardless of the position or size of the squares.

This pattern effectively performs a targeted color substitution within the grid, maintaining the overall structure while potentially altering the visual relationships between squares. It demonstrates how a simple color change can significantly impact the perception of shapes and their arrangements within a grid.
"""


class ChangeStrokeColorTask(GridTask):
    def __init__(self):
        super().__init__()
        self.grid_size = np.random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE + 1)
        self.colors = np.random.choice(
            list(range(1, 7)), size=np.random.randint(2, 7), replace=False
        )
        self.num_squares = np.random.randint(1, 4)

    def sample(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for _ in range(self.num_squares):
            size = np.random.randint(3, 5)
            color = np.random.choice(self.colors)
            start_x = np.random.randint(0, min(1, self.grid_size - size))
            start_y = np.random.randint(0, min(1, self.grid_size - size))
            grid[start_x : start_x + size, start_y : start_y + size] = color
        return grid

    def execute(self, grid: np.ndarray) -> (np.ndarray, str):
        self.stroke_color = np.random.choice(self.colors)
        new_grid = np.copy(grid)

        # Find all exterior cells using flood fill
        exterior = np.zeros_like(grid, dtype=bool)

        def flood_fill(x, y):
            stack = [(x, y)]
            while stack:
                cx, cy = stack.pop()
                if cx < 0 or cx >= self.grid_size or cy < 0 or cy >= self.grid_size:
                    continue
                if exterior[cx, cy] or grid[cx, cy] > 0:
                    continue
                exterior[cx, cy] = True
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    stack.append((cx + dx, cy + dy))

        for i in range(self.grid_size):
            if grid[i, 0] == 0:
                flood_fill(i, 0)
            if grid[i, self.grid_size - 1] == 0:
                flood_fill(i, self.grid_size - 1)
        for j in range(self.grid_size):
            if grid[0, j] == 0:
                flood_fill(0, j)
            if grid[self.grid_size - 1, j] == 0:
                flood_fill(self.grid_size - 1, j)

        # Change stroke color of shapes by checking exterior cells
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if grid[i, j] > 0:
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + dx, j + dy
                        if (
                            0 <= ni < self.grid_size
                            and 0 <= nj < self.grid_size
                            and exterior[ni, nj]
                        ):
                            new_grid[i, j] = self.stroke_color
                            break

        description = self.generate_description(grid, new_grid)
        pattern = self.generate_pattern_description()
        return new_grid, f"{description}\n\n{pattern}"

    def generate_description(
        self, input_grid: np.ndarray, output_grid: np.ndarray
    ) -> str:
        def describe_grid(grid, grid_name):
            unique, counts = np.unique(grid, return_counts=True)
            color_counts = dict(zip(unique, counts))
            total_cells = self.grid_size * self.grid_size

            color_descriptions = []
            for color in sorted(color_counts.keys()):
                if color != 0:  # Exclude the background color
                    percentage = (color_counts[color] / total_cells) * 100
                    color_descriptions.append(
                        f"{self.color_map[color]} ({percentage:.1f}%)"
                    )

            color_list = (
                ", ".join(color_descriptions[:-1]) + f" and {color_descriptions[-1]}"
                if len(color_descriptions) > 1
                else color_descriptions[0]
            )

            return (
                f"The {grid_name} is a {self.grid_size}x{self.grid_size} square with a black background, containing {self.num_squares} colored shape{'s' if self.num_squares > 1 else ''}. "
                f"The colors present are {color_list}."
            )

        input_description = describe_grid(input_grid, "input grid")
        output_description = describe_grid(output_grid, "output grid")

        changed_cells = np.sum(output_grid == self.stroke_color) - np.sum(
            input_grid == self.stroke_color
        )
        percentage_changed = (changed_cells / (self.grid_size * self.grid_size)) * 100

        difference = f"""
The main differences between the input and output grids are:
1. The stroke (outer edge) of all shapes in the grid has been changed to {self.color_map[self.stroke_color]}.
2. This stroke color change affects {changed_cells} cells, which is {percentage_changed:.1f}% of the total grid.
3. The positions, sizes, and interior colors of the shapes remain the same; only their outer edges have changed color.
4. The black background of the grid remains unchanged.
"""

        return f"{input_description}\n\n{output_description}\n\n{difference}"

    def generate_pattern_description(self) -> str:
        return """
The pattern that transforms the input grid to the output grid involves changing the stroke color of shapes. Specifically:

1. The transformation identifies all shapes present in the input grid.

2. It then changes the color of the outer edge (stroke) of each shape to a new, consistent color.

3. This color change only affects the cells that form the perimeter of each shape, leaving the interior cells and the background unchanged.

4. The stroke color change is applied uniformly to all shapes in the grid, regardless of their original colors.

5. The sizes, positions, and overall structure of the shapes remain constant; only their outer edge color is modified.

6. The black background of the grid, where no shapes were drawn, remains unaffected by this transformation.

This stroke color change results in a grid where all shapes are now outlined in the same color, potentially creating a more unified visual appearance while maintaining the distinct interior colors of each shape. The transformation preserves the overall layout and structure of the shapes within the grid.
"""


class ChangeFillColorTask(GridTask):
    def __init__(self):
        super().__init__()
        self.grid_size = np.random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE + 1)
        self.colors = np.random.choice(
            list(range(1, 7)), size=np.random.randint(2, 7), replace=False
        )
        self.num_squares = np.random.randint(1, 4)

    def sample(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for _ in range(self.num_squares):
            size = np.random.randint(3, 5)
            color = np.random.choice(self.colors)
            start_x = np.random.randint(0, self.grid_size - size)
            start_y = np.random.randint(0, self.grid_size - size)
            grid[start_x : start_x + size, start_y : start_y + size] = color
        return grid

    def execute(self, grid: np.ndarray) -> (np.ndarray, str):
        self.fill_color = np.random.choice(self.colors)
        new_grid = np.copy(grid)
        for i in range(1, self.grid_size - 1):
            for j in range(1, self.grid_size - 1):
                if (
                    grid[i, j] != 0
                    and grid[i - 1, j] != 0
                    and grid[i + 1, j] != 0
                    and grid[i, j - 1] != 0
                    and grid[i, j + 1] != 0
                ):
                    new_grid[i, j] = self.fill_color

        description = self.generate_description(grid, new_grid)
        pattern = self.generate_pattern_description()
        return new_grid, f"{description}\n\n{pattern}"

    def generate_description(
        self, input_grid: np.ndarray, output_grid: np.ndarray
    ) -> str:
        def describe_grid(grid, grid_name):
            unique, counts = np.unique(grid, return_counts=True)
            color_counts = dict(zip(unique, counts))
            total_cells = self.grid_size * self.grid_size

            color_descriptions = []
            for color in sorted(color_counts.keys()):
                if color != 0:  # Exclude the background color
                    percentage = (color_counts[color] / total_cells) * 100
                    color_descriptions.append(
                        f"{self.color_map[color]} ({percentage:.1f}%)"
                    )

            color_list = (
                ", ".join(color_descriptions[:-1]) + f" and {color_descriptions[-1]}"
                if len(color_descriptions) > 1
                else color_descriptions[0]
            )

            return (
                f"The {grid_name} is a {self.grid_size}x{self.grid_size} square with a black background, containing {self.num_squares} colored shape{'s' if self.num_squares > 1 else ''}. "
                f"The colors present are {color_list}."
            )

        input_description = describe_grid(input_grid, "input grid")
        output_description = describe_grid(output_grid, "output grid")

        changed_cells = np.sum((input_grid != output_grid) & (input_grid != 0))
        percentage_changed = (changed_cells / (self.grid_size * self.grid_size)) * 100

        difference = f"""
The main differences between the input and output grids are:
1. The fill color of all shapes in the grid has been changed to {self.color_map[self.fill_color]}.
2. This fill color change affects {changed_cells} cells, which is {percentage_changed:.1f}% of the total grid.
3. The stroke (outer edge) color of the shapes remains unchanged.
4. The positions and sizes of the shapes remain the same; only their interior color has changed.
5. The black background of the grid remains unchanged.
"""

        return f"{input_description}\n\n{output_description}\n\n{difference}"

    def generate_pattern_description(self) -> str:
        return """
The pattern that transforms the input grid to the output grid involves changing the fill color of shapes while preserving their stroke color. Specifically:

1. The transformation identifies all shapes present in the input grid.

2. It then changes the color of the interior (fill) of each shape to a new, consistent color.

3. This color change only affects the cells that form the interior of each shape, leaving the outer edge (stroke) and the background unchanged.

4. The fill color change is applied uniformly to all shapes in the grid, regardless of their original colors.

5. The sizes, positions, and overall structure of the shapes remain constant; only their interior color is modified.

6. The black background of the grid, where no shapes were drawn, remains unaffected by this transformation.

This fill color change results in a grid where all shapes now have the same interior color, while maintaining their original outlines. This creates a unified appearance for the shape interiors while preserving the distinct stroke colors and overall layout of the shapes within the grid.
"""


# class CopyShapeWithPaddingTask(GridTask):
#     def __init__(self, single_color=True):
#         super().__init__()
#         self.grid_size = np.random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE + 1)
#         self.shape_size = np.random.randint(2, max(3, self.grid_size // 2 + 1))

#         self.single_color = single_color
#         self.color = np.random.randint(1, 7) if single_color else None

#     def sample(self):
#         grid = generate_shape(self.grid_size, self.shape_size)
#         if self.single_color:
#             grid[grid == 1] = self.color
#         else:
#             colors = np.random.choice(
#                 range(1, 7), size=np.count_nonzero(grid), replace=True
#             )
#             grid[grid == 1] = colors
#         return grid

#     def execute(self, grid: np.ndarray) -> (np.ndarray, str):
#         shape_coords = np.argwhere(grid > 0)
#         if shape_coords.size == 0:
#             return grid, "No shape found to copy"

#         min_x, min_y = shape_coords.min(axis=0)
#         max_x, max_y = shape_coords.max(axis=0)
#         shape_component = grid[min_x : max_x + 1, min_y : max_y + 1]
#         shape_height, shape_width = shape_component.shape

#         new_grid = np.copy(grid)

#         # Define all possible positions including diagonals
#         all_positions = {
#             "Above": (min_x - shape_height - 1, min_y),
#             "Right": (min_x, min_y + shape_width + 1),
#             "Below": (max_x + 2, min_y),
#             "Left": (min_x, min_y - shape_width - 1),
#             "Top-left": (min_x - shape_height - 1, min_y - shape_width - 1),
#             "Top-right": (min_x - shape_height - 1, min_y + shape_width + 1),
#             "Bottom-left": (max_x + 2, min_y - shape_width - 1),
#             "Bottom-right": (max_x + 2, min_y + shape_width + 1),
#         }

#         # Determine the selection strategy
#         strategy = np.random.choice(
#             ["all", "top_left_bottom_right", "diagonal", "random"],
#             p=[0.25, 0.25, 0.25, 0.25],
#         )

#         if strategy == "all":
#             positions = all_positions
#             instruction = "Copy the original shape, and then copy it above, right, below, and left with padding 1px"
#         elif strategy == "top_left_bottom_right":
#             positions = {
#                 key: all_positions[key] for key in ["Above", "Right", "Below", "Left"]
#             }
#             instruction = np.random.choice(
#                 [
#                     "Copy the original shape, and then copy it above, right, below, and left with padding 1px",
#                     "Copy the original shape to the top, left, bottom, and right with padding 1px",
#                 ]
#             )
#         elif strategy == "diagonal":
#             positions = {
#                 key: all_positions[key]
#                 for key in ["Top-left", "Top-right", "Bottom-left", "Bottom-right"]
#             }
#             instruction = "Copy the original shape, and then copy it to the top-left, top-right, bottom-left, and bottom-right with padding 1px"
#         else:
#             num_directions = np.random.randint(1, len(all_positions) + 1)
#             selected_keys = np.random.choice(
#                 list(all_positions.keys()), num_directions, replace=False
#             )
#             positions = {key: all_positions[key] for key in selected_keys}
#             instruction = f"Copy the original shape to the following positions with padding 1px: {', '.join(positions.keys())}"

#         for direction, pos in positions.items():
#             pos_x, pos_y = pos
#             if pos_x < 0 or pos_y < 0:
#                 # Compute the intersection of the shape with the grid
#                 intersect_x_start = max(0, pos_x)
#                 intersect_y_start = max(0, pos_y)
#                 intersect_x_end = min(self.grid_size, pos_x + shape_height)
#                 intersect_y_end = min(self.grid_size, pos_y + shape_width)

#                 if (
#                     intersect_x_start < intersect_x_end
#                     and intersect_y_start < intersect_y_end
#                 ):
#                     new_grid[
#                         intersect_x_start:intersect_x_end,
#                         intersect_y_start:intersect_y_end,
#                     ] = shape_component[
#                         intersect_x_start - pos_x : intersect_x_end - pos_x,
#                         intersect_y_start - pos_y : intersect_y_end - pos_y,
#                     ]
#             else:
#                 end_x = min(pos_x + shape_height, self.grid_size)
#                 end_y = min(pos_y + shape_width, self.grid_size)
#                 new_grid[pos_x:end_x, pos_y:end_y] = shape_component[
#                     : end_x - pos_x, : end_y - pos_y
#                 ]

#         return new_grid, instruction


class PatternIntersectionUnionTask(GridTask):
    def __init__(self, n, mode, orientation, max_colors):
        super().__init__()
        self.n = n
        self.mode = mode
        self.orientation = orientation
        self.max_colors = max_colors

        if self.orientation == "vertical":
            self.grid_size = (self.n, 2 * self.n + 1)
        else:  # horizontal
            self.grid_size = (2 * self.n + 1, self.n)

    def sample(self):
        grid = np.zeros(self.grid_size, dtype=int)

        if self.orientation == "vertical":
            left_side = np.random.randint(1, self.max_colors + 1, (self.n, self.n))
            right_side = np.random.randint(1, self.max_colors + 1, (self.n, self.n))

            grid[:, : self.n] = left_side
            grid[:, self.n + 1 :] = right_side

            random_color = np.random.randint(1, 7)
            grid[:, self.n] = random_color
        else:  # horizontal
            top_side = np.random.randint(1, self.max_colors + 1, (self.n, self.n))
            bottom_side = np.random.randint(1, self.max_colors + 1, (self.n, self.n))

            grid[: self.n, :] = top_side
            grid[self.n + 1 :, :] = bottom_side

            random_color = np.random.randint(1, 7)
            grid[self.n, :] = random_color

        return grid

    def execute(self, grid: np.ndarray) -> (np.ndarray, str):
        if self.orientation == "vertical":
            left_side = grid[:, : self.n]
            right_side = grid[:, self.n + 1 :]
            divider_color = grid[:, self.n]
        else:  # horizontal
            top_side = grid[: self.n, :]
            bottom_side = grid[self.n + 1 :, :]
            divider_color = grid[self.n, :]

        intersection = (
            np.logical_and(left_side, right_side).astype(int)
            if self.orientation == "vertical"
            else np.logical_and(top_side, bottom_side).astype(int)
        )
        union = (
            np.logical_or(left_side, right_side).astype(int)
            if self.orientation == "vertical"
            else np.logical_or(top_side, bottom_side).astype(int)
        )

        if self.mode == "intersection":
            result_grid = intersection
        elif self.mode == "union":
            result_grid = union
        elif self.mode == "combined":
            result_grid = np.zeros(self.grid_size, dtype=int)
            if self.orientation == "vertical":
                result_grid[:, : self.n] = union
                result_grid[:, self.n + 1 :] = intersection
                result_grid[:, self.n] = divider_color
            else:
                result_grid[: self.n, :] = union
                result_grid[self.n + 1 :, :] = intersection
                result_grid[self.n, :] = divider_color
        else:
            raise ValueError(
                "Invalid mode. Choose from 'intersection', 'union', or 'combined'."
            )

        description = self.generate_description(grid, result_grid)
        pattern = self.generate_pattern_description()
        return result_grid, f"{description}\n\n{pattern}"

    def generate_description(
        self, input_grid: np.ndarray, output_grid: np.ndarray
    ) -> str:
        def describe_grid(grid, grid_name):
            unique, counts = np.unique(grid, return_counts=True)
            color_counts = dict(zip(unique, counts))
            total_cells = self.grid_size[0] * self.grid_size[1]

            color_descriptions = []
            for color in sorted(color_counts.keys()):
                if color != 0:  # Exclude the background color
                    percentage = (color_counts[color] / total_cells) * 100
                    color_descriptions.append(
                        f"{self.color_map[color]} ({percentage:.1f}%)"
                    )

            color_list = (
                ", ".join(color_descriptions[:-1]) + f" and {color_descriptions[-1]}"
                if len(color_descriptions) > 1
                else color_descriptions[0]
            )

            return (
                f"The {grid_name} is a {self.grid_size[0]}x{self.grid_size[1]} grid. "
                f"The colors present are {color_list}."
            )

        input_description = describe_grid(input_grid, "input grid")
        output_description = describe_grid(output_grid, "output grid")

        difference = f"""
The main differences between the input and output grids are:
1. The input grid is divided into two main sections by a {self.orientation} divider.
2. The output grid has been transformed based on the '{self.mode}' operation:
   {"- It shows the intersection of the two sides." if self.mode == "intersection" else
    "- It shows the union of the two sides." if self.mode == "union" else
    f"- The {'left' if self.orientation == 'vertical' else 'top'} side shows the union, while the {'right' if self.orientation == 'vertical' else 'bottom'} side shows the intersection."}
3. The color distribution has changed as a result of this operation.
4. {"The divider from the input grid is preserved in the output." if self.mode == "combined" else "The divider from the input grid is not present in the output."}
"""

        return f"{input_description}\n\n{output_description}\n\n{difference}"

    def generate_pattern_description(self) -> str:
        return f"""
The pattern that transforms the input grid to the output grid involves a {self.mode} operation on a {self.orientation}ly divided grid. Specifically:

1. Input Grid Structure:
   - The input is a {self.grid_size[0]}x{self.grid_size[1]} grid.
   - It is divided into two main sections by a {self.orientation} divider.
   - Each section contains colors ranging from 1 to {self.max_colors}.

2. Transformation Operation:
   - The operation is performed on the two main sections, treating them as binary masks.
   - Colors greater than 0 are treated as 1 (present), and 0 as 0 (absent).

3. Mode of Operation ({self.mode}):
   {"- Intersection: The output shows only the cells that have non-zero values in both sections." if self.mode == "intersection" else
    "- Union: The output shows cells that have non-zero values in either section." if self.mode == "union" else
    f"- Combined: The {'left' if self.orientation == 'vertical' else 'top'} side of the output shows the union, while the {'right' if self.orientation == 'vertical' else 'bottom'} side shows the intersection."}

4. Output Grid:
   {"- The result is an {self.n}x{self.n} grid." if self.mode != "combined" else
    f"- The result maintains the {self.grid_size[0]}x{self.grid_size[1]} size of the input grid, including the divider."}

5. Color Preservation:
   - In the output, cells that meet the operation criteria retain their original color from the input grid.
   - Cells that don't meet the criteria become black (0).

This transformation effectively combines or compares the patterns from the two sections of the input grid, resulting in a new pattern that reflects the specified {self.mode} operation.
"""


class RotateShapeTask(GridTask):
    def __init__(self, grid_size, num_shapes, colors, mode, shape_size):
        super().__init__()
        self.grid_size = grid_size
        self.num_shapes = num_shapes
        self.colors = colors
        self.mode = mode
        self.shape_size = shape_size

    def sample(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for color in self.colors:
            shape = generate_shape(self.grid_size, self.shape_size)
            shape[shape == 1] = color
            grid = np.maximum(grid, shape)
        return grid

    def execute(self, grid: np.ndarray) -> (np.ndarray, str):
        if self.mode == "clockwise":
            new_grid = np.rot90(grid, -1)  # Rotate 90 degrees clockwise
        elif self.mode == "counter-clockwise":
            new_grid = np.rot90(grid, 1)  # Rotate 90 degrees counter-clockwise
        elif self.mode == "180":
            new_grid = np.rot90(grid, 2)  # Rotate 180 degrees
        else:
            raise ValueError(
                "Invalid mode. Choose from 'clockwise', 'counter-clockwise', or '180'."
            )

        description = self.generate_description(grid, new_grid)
        pattern = self.generate_pattern_description()
        return new_grid, f"{description}\n\n{pattern}"

    def generate_description(
        self, input_grid: np.ndarray, output_grid: np.ndarray
    ) -> str:
        def describe_grid(grid, grid_name):
            unique, counts = np.unique(grid, return_counts=True)
            color_counts = dict(zip(unique, counts))
            total_cells = self.grid_size * self.grid_size

            color_descriptions = []
            for color in sorted(color_counts.keys()):
                if color != 0:  # Exclude the background color
                    percentage = (color_counts[color] / total_cells) * 100
                    color_descriptions.append(
                        f"{self.color_map[color]} ({percentage:.1f}%)"
                    )

            color_list = (
                ", ".join(color_descriptions[:-1]) + f" and {color_descriptions[-1]}"
                if len(color_descriptions) > 1
                else color_descriptions[0]
            )

            return (
                f"The {grid_name} is a {self.grid_size}x{self.grid_size} square with a black background, containing {self.num_shapes} shape{'s' if self.num_shapes > 1 else ''}. "
                f"The colors present are {color_list}."
            )

        input_description = describe_grid(input_grid, "input grid")
        output_description = describe_grid(output_grid, "output grid")

        rotation_description = {
            "clockwise": "90 degrees clockwise",
            "counter-clockwise": "90 degrees counter-clockwise",
            "180": "180 degrees",
        }[self.mode]

        difference = f"""
The main differences between the input and output grids are:
1. The entire grid has been rotated {rotation_description}.
2. The shapes maintain their relative positions to each other, but their absolute positions in the grid have changed due to the rotation.
3. The colors and sizes of the shapes remain unchanged.
4. The black background is also rotated, maintaining its relative position to the shapes.
"""

        return f"{input_description}\n\n{output_description}\n\n{difference}"

    def generate_pattern_description(self) -> str:
        return f"""
The pattern that transforms the input grid to the output grid involves a rotation of the entire grid. Specifically:

1. Grid Composition:
   - The grid is {self.grid_size}x{self.grid_size} in size.
   - It contains {self.num_shapes} shape{'s' if self.num_shapes > 1 else ''} of various colors.
   - The shapes are generated with a maximum size of {self.shape_size}x{self.shape_size}.
   - The colors used for the shapes are: {', '.join([self.color_map[color] for color in self.colors])}.

2. Rotation Operation:
   - The entire grid, including all shapes and the background, is rotated {self.mode.replace('-', ' ')}.
   {"- This results in a 90-degree rotation to the right." if self.mode == "clockwise" else
    "- This results in a 90-degree rotation to the left." if self.mode == "counter-clockwise" else
    "- This results in the grid being flipped upside down."}

3. Shape Preservation:
   - The shapes maintain their size, color, and relative positions to each other.
   - Only the absolute positions of the shapes within the grid change due to the rotation.

4. Color and Background:
   - The color of each shape remains the same after rotation.
   - The black background also rotates, maintaining its relative position to the shapes.

5. Grid Boundaries:
   - The rotation is performed within the confines of the original grid size.
   - No part of any shape is cut off or moved outside the grid boundaries during rotation.

This transformation results in a grid that contains the same elements as the input, but with their positions altered by the specified rotation. The overall structure and composition of the grid remain intact, just viewed from a different angle.
"""


class MoveShapeTask(GridTask):
    def __init__(
        self, grid_size, num_shapes, colors, shape_size, move_color, n, direction, mode
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_shapes = num_shapes
        self.colors = colors
        self.shape_size = shape_size
        self.move_color = move_color
        self.n = n
        self.direction = direction
        self.mode = mode

    def sample(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for color in self.colors:
            shape = generate_shape(self.grid_size, self.shape_size)
            shape[shape == 1] = color
            grid = np.maximum(grid, shape)
        return grid

    def execute(self, grid: np.ndarray) -> (np.ndarray, str):
        new_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        if self.mode == "move-and-copy":
            for other_color in self.colors:
                if other_color != self.move_color:
                    new_grid[grid == other_color] = other_color

        def move_shape(new_grid, old_grid, color, n, direction):
            positions = np.argwhere(old_grid == color)
            if direction == "right":
                for i, j in positions:
                    if j + n < self.grid_size:
                        new_grid[i, j + n] = color
            elif direction == "left":
                for i, j in positions:
                    if j - n >= 0:
                        new_grid[i, j - n] = color
            elif direction == "bottom":
                for i, j in positions:
                    if i + n < self.grid_size:
                        new_grid[i + n, j] = color
            elif direction == "top":
                for i, j in positions:
                    if i - n >= 0:
                        new_grid[i - n, j] = color
            elif direction == "top-right":
                for i, j in positions:
                    if i - n >= 0 and j + n < self.grid_size:
                        new_grid[i - n, j + n] = color
            elif direction == "top-left":
                for i, j in positions:
                    if i - n >= 0 and j - n >= 0:
                        new_grid[i - n, j - n] = color
            elif direction == "bottom-right":
                for i, j in positions:
                    if i + n < self.grid_size and j + n < self.grid_size:
                        new_grid[i + n, j + n] = color
            elif direction == "bottom-left":
                for i, j in positions:
                    if i + n < self.grid_size and j - n >= 0:
                        new_grid[i + n, j - n] = color

        move_shape(new_grid, grid, self.move_color, self.n, self.direction)

        description = self.generate_description(grid, new_grid)
        pattern = self.generate_pattern_description()
        return new_grid, f"{description}\n\n{pattern}"

    def generate_description(
        self, input_grid: np.ndarray, output_grid: np.ndarray
    ) -> str:
        def describe_grid(grid, grid_name):
            unique, counts = np.unique(grid, return_counts=True)
            color_counts = dict(zip(unique, counts))
            total_cells = self.grid_size * self.grid_size

            color_descriptions = []
            for color in sorted(color_counts.keys()):
                if color != 0:  # Exclude the background color
                    percentage = (color_counts[color] / total_cells) * 100
                    color_descriptions.append(
                        f"{self.color_map[color]} ({percentage:.1f}%)"
                    )

            color_list = (
                ", ".join(color_descriptions[:-1]) + f" and {color_descriptions[-1]}"
                if len(color_descriptions) > 1
                else color_descriptions[0]
            )

            return (
                f"The {grid_name} is a {self.grid_size}x{self.grid_size} square with a black background, containing {self.num_shapes} shape{'s' if self.num_shapes > 1 else ''}. "
                f"The colors present are {color_list}."
            )

        input_description = describe_grid(input_grid, "input grid")
        output_description = describe_grid(output_grid, "output grid")

        difference = f"""
The main differences between the input and output grids are:
1. All shapes of color {self.color_map[self.move_color]} have been moved {self.n} pixel{'s' if self.n > 1 else ''} to the {self.direction}.
2. {"Other shapes remain in their original positions." if self.mode == "move-and-copy" else "Other shapes have been removed."}
3. The overall distribution of colors has changed due to the movement of the {self.color_map[self.move_color]} shape{'s' if self.num_shapes > 1 else ''}.
4. {"Some parts of the moved shapes may have been cut off if they went beyond the grid boundaries." if self.n > 1 else ""}
"""

        return f"{input_description}\n\n{output_description}\n\n{difference}"

    def generate_pattern_description(self) -> str:
        return f"""
The pattern that transforms the input grid to the output grid involves moving shapes of a specific color. Specifically:

1. Grid Composition:
   - The grid is {self.grid_size}x{self.grid_size} in size.
   - It contains {self.num_shapes} shape{'s' if self.num_shapes > 1 else ''} of various colors.
   - The shapes are generated with a maximum size of {self.shape_size}x{self.shape_size}.
   - The colors used for the shapes are: {', '.join([self.color_map[color] for color in self.colors])}.

2. Movement Operation:
   - Shapes of color {self.color_map[self.move_color]} are moved {self.n} pixel{'s' if self.n > 1 else ''} in the {self.direction} direction.
   - If a part of a shape would move beyond the grid boundaries, that part is not displayed in the output.

3. Mode of Operation ({self.mode}):
   {"- Other shapes remain in their original positions." if self.mode == "move-and-copy" else
    "- Other shapes are removed from the grid."}

4. Color Preservation:
   - The moved shapes retain their original color ({self.color_map[self.move_color]}).
   {"- Other shapes, if present, also retain their original colors." if self.mode == "move-and-copy" else ""}

5. Grid Boundaries:
   - The movement is performed within the confines of the original grid size.
   - Parts of shapes that would move outside the grid boundaries are cut off.

This transformation results in a grid where shapes of a specific color have changed position, while the treatment of other shapes depends on the chosen mode of operation. The overall structure of the grid remains the same size, but the distribution of colors within it changes due to the movement.
"""


class MirrorShapeTask(GridTask):
    def __init__(self, grid_size, num_shapes, colors, axis, shape_size):
        super().__init__()
        self.grid_size = grid_size
        self.num_shapes = num_shapes
        self.colors = colors
        self.axis = axis
        self.shape_size = shape_size

    def sample(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for color in self.colors:
            shape = generate_shape(self.grid_size, self.shape_size)
            shape[shape == 1] = color
            grid = np.maximum(grid, shape)
        return grid

    def execute(self, grid: np.ndarray) -> (np.ndarray, str):
        if self.axis == "horizontal":
            new_grid = np.fliplr(grid)
        else:
            new_grid = np.flipud(grid)

        description = self.generate_description(grid, new_grid)
        pattern = self.generate_pattern_description()
        return new_grid, f"{description}\n\n{pattern}"

    def generate_description(
        self, input_grid: np.ndarray, output_grid: np.ndarray
    ) -> str:
        def describe_grid(grid, grid_name):
            unique, counts = np.unique(grid, return_counts=True)
            color_counts = dict(zip(unique, counts))
            total_cells = self.grid_size * self.grid_size

            color_descriptions = []
            for color in sorted(color_counts.keys()):
                if color != 0:  # Exclude the background color
                    percentage = (color_counts[color] / total_cells) * 100
                    color_descriptions.append(
                        f"{self.color_map[color]} ({color_counts[color]} cells, {percentage:.1f}%)"
                    )

            color_list = (
                ", ".join(color_descriptions[:-1]) + f" and {color_descriptions[-1]}"
                if len(color_descriptions) > 1
                else color_descriptions[0]
            )

            shapes = []
            for color in self.colors:
                shape_cells = np.sum(grid == color)
                if shape_cells > 0:
                    shapes.append(
                        f"a {self.color_map[color]} shape with {shape_cells} cells"
                    )

            shape_description = (
                ", ".join(shapes[:-1]) + f" and {shapes[-1]}"
                if len(shapes) > 1
                else shapes[0]
            )

            return f"""The {grid_name} is a {self.grid_size}x{self.grid_size} square with a black background. 
It contains {self.num_shapes} shape{'s' if self.num_shapes > 1 else ''}: {shape_description}.
The colors present are {color_list}.
The shapes occupy {np.sum(grid > 0)} cells ({(np.sum(grid > 0) / total_cells) * 100:.1f}% of the grid), 
while the remaining {np.sum(grid == 0)} cells ({(np.sum(grid == 0) / total_cells) * 100:.1f}% of the grid) are black background."""

        input_description = describe_grid(input_grid, "input grid")
        output_description = describe_grid(output_grid, "output grid")

        difference = f"""
The main difference between the input and output grids is:
1. The entire grid has been mirrored {self.axis}ly.
2. The shapes maintain their colors and relative positions to each other, but their absolute positions in the grid have changed due to the mirroring.
3. The number of cells occupied by each color remains the same, but their positions have been reversed along the {self.axis} axis.
4. The black background is also mirrored, maintaining its relative position to the shapes.
"""

        return f"{input_description}\n\n{output_description}\n\n{difference}"

    def generate_pattern_description(self) -> str:
        return f"""
The pattern that transforms the input grid to the output grid involves mirroring the entire grid. Specifically:

1. Grid Composition:
   - The grid is {self.grid_size}x{self.grid_size} in size.
   - It contains {self.num_shapes} shape{'s' if self.num_shapes > 1 else ''} of various colors.
   - The shapes are generated with a maximum size of {self.shape_size}x{self.shape_size}.
   - The colors used for the shapes are: {', '.join([self.color_map[color] for color in self.colors])}.

2. Mirroring Operation:
   - The entire grid, including all shapes and the background, is mirrored {self.axis}ly.
   - This results in the grid being flipped along its {self.axis} axis.

3. Shape Preservation:
   - The shapes maintain their size, color, and relative positions to each other.
   - Only the absolute positions of the shapes within the grid change due to the mirroring.

4. Color and Background:
   - The color of each shape remains the same after mirroring.
   - The black background is also mirrored, maintaining its relative position to the shapes.

5. Grid Boundaries:
   - The mirroring is performed within the confines of the original grid size.
   - No part of any shape is cut off or moved outside the grid boundaries during mirroring.

This transformation results in a grid that contains the same elements as the input, but with their positions reversed along the {self.axis} axis. The overall structure and composition of the grid remain intact, just viewed as if reflected in a {self.axis} mirror.
"""


class CompleteDiagonalTask(GridTask):
    def __init__(self, grid_size, color, diagonal_color, num_diagonals):
        super().__init__()
        self.grid_size = grid_size
        self.color = color
        self.diagonal_color = diagonal_color
        self.num_diagonals = num_diagonals

    def sample(self):
        grid = np.full((self.grid_size, self.grid_size), self.color)
        for _ in range(self.num_diagonals):
            length = np.random.randint(3, min(self.grid_size, self.grid_size) + 1)
            start_x = np.random.randint(0, self.grid_size - length + 1)
            start_y = np.random.randint(0, self.grid_size - length + 1)
            for i in range(length):
                grid[start_x + i, start_y + i] = self.diagonal_color
        return grid

    def execute(self, grid: np.ndarray) -> (np.ndarray, str):
        new_grid = np.copy(grid)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if grid[i, j] == self.diagonal_color:
                    # Completing diagonal from (i, j) to the bottom-right corner
                    x, y = i, j
                    while x < self.grid_size and y < self.grid_size:
                        new_grid[x, y] = self.diagonal_color
                        x += 1
                        y += 1

                    # Completing diagonal from (i, j) to the top-left corner
                    x, y = i, j
                    while x >= 0 and y >= 0:
                        new_grid[x, y] = self.diagonal_color
                        x -= 1
                        y -= 1

        description = self.generate_description(grid, new_grid)
        pattern = self.generate_pattern_description()
        return new_grid, f"{description}\n\n{pattern}"

    def generate_description(
        self, input_grid: np.ndarray, output_grid: np.ndarray
    ) -> str:
        def describe_grid(grid, grid_name):
            unique, counts = np.unique(grid, return_counts=True)
            color_counts = dict(zip(unique, counts))
            total_cells = self.grid_size * self.grid_size

            color_descriptions = []
            for color in sorted(color_counts.keys()):
                percentage = (color_counts[color] / total_cells) * 100
                color_descriptions.append(
                    f"{self.color_map[color]} ({color_counts[color]} cells, {percentage:.1f}%)"
                )

            color_list = (
                ", ".join(color_descriptions[:-1]) + f" and {color_descriptions[-1]}"
                if len(color_descriptions) > 1
                else color_descriptions[0]
            )

            diagonal_cells = np.sum(grid == self.diagonal_color)
            diagonal_percentage = (diagonal_cells / total_cells) * 100

            return f"""The {grid_name} is a {self.grid_size}x{self.grid_size} square. 
The background color is {self.color_map[self.color]}, occupying {color_counts[self.color]} cells ({(color_counts[self.color] / total_cells) * 100:.1f}% of the grid).
It contains {self.num_diagonals} diagonal line{'s' if self.num_diagonals > 1 else ''} of color {self.color_map[self.diagonal_color]}, 
occupying {diagonal_cells} cells ({diagonal_percentage:.1f}% of the grid).
The colors present are {color_list}."""

        input_description = describe_grid(input_grid, "input grid")
        output_description = describe_grid(output_grid, "output grid")

        difference = f"""
The main differences between the input and output grids are:
1. The diagonal lines of color {self.color_map[self.diagonal_color]} have been extended to cover entire diagonals of the grid.
2. The number of {self.color_map[self.diagonal_color]} cells has increased from {np.sum(input_grid == self.diagonal_color)} to {np.sum(output_grid == self.diagonal_color)}.
3. The number of {self.color_map[self.color]} (background) cells has decreased from {np.sum(input_grid == self.color)} to {np.sum(output_grid == self.color)}.
4. The overall pattern of the grid has changed from partial diagonal lines to complete diagonal lines traversing the entire grid.
"""

        return f"{input_description}\n\n{output_description}\n\n{difference}"

    def generate_pattern_description(self) -> str:
        return f"""
The pattern that transforms the input grid to the output grid involves completing diagonal lines. Specifically:

1. Grid Composition:
   - The grid is {self.grid_size}x{self.grid_size} in size.
   - The background color is {self.color_map[self.color]}.
   - It contains {self.num_diagonals} partial diagonal line{'s' if self.num_diagonals > 1 else ''} of color {self.color_map[self.diagonal_color]}.

2. Diagonal Completion:
   - Each partial diagonal line of color {self.color_map[self.diagonal_color]} is extended in both directions.
   - The extension continues until it reaches the edges of the grid.
   - This creates complete diagonal lines that traverse the entire grid.

3. Color Preservation:
   - The colors of the diagonal lines ({self.color_map[self.diagonal_color]}) and the background ({self.color_map[self.color]}) remain unchanged.
   - Only the extent of the diagonal lines changes.

4. Background Reduction:
   - As the diagonal lines are extended, they replace cells that were previously the background color.
   - This results in a reduction of the background color's presence in the grid.

5. Pattern Formation:
   - The completed diagonals form a new pattern across the entire grid.
   - This pattern consists of full diagonal lines intersecting at various points.

This transformation results in a grid where partial diagonal lines have been extended to create a more pronounced diagonal pattern, significantly altering the overall appearance of the grid while maintaining the original color scheme.
"""


class SortColumnsRowsByHeightTask(GridTask):
    def __init__(self, grid_size, orientation, sort_order, colors):
        super().__init__()
        self.grid_size = grid_size
        self.orientation = orientation
        self.sort_order = sort_order
        self.colors = colors

    def sample(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        heights = np.random.permutation(np.arange(1, self.grid_size + 1))

        if self.orientation == "rows":
            for i, height in enumerate(heights):
                grid[i, :height] = self.colors[i]
        else:  # columns
            for i, height in enumerate(heights):
                grid[:height, i] = self.colors[i]

        return grid

    def execute(self, grid: np.ndarray) -> (np.ndarray, str):
        if self.orientation == "rows":
            heights = [np.count_nonzero(row) for row in grid]
            sorted_indices = np.argsort(heights)
            if self.sort_order == "descending":
                sorted_indices = sorted_indices[::-1]
            sorted_grid = grid[sorted_indices]

            # Left-align each row
            new_grid = np.zeros_like(grid)
            for i, row in enumerate(sorted_grid):
                num_elements = np.count_nonzero(row)
                new_grid[i, :num_elements] = row[row != 0]
        else:  # columns
            heights = [np.count_nonzero(grid[:, i]) for i in range(grid.shape[1])]
            sorted_indices = np.argsort(heights)
            if self.sort_order == "descending":
                sorted_indices = sorted_indices[::-1]
            sorted_grid = grid[:, sorted_indices]

            # Top-align each column
            new_grid = np.zeros_like(grid)
            for i, col in enumerate(sorted_grid.T):
                num_elements = np.count_nonzero(col)
                new_grid[:num_elements, i] = col[col != 0]

        description = self.generate_description(grid, new_grid)
        pattern = self.generate_pattern_description()
        return new_grid, f"{description}\n\n{pattern}"

    def generate_description(
        self, input_grid: np.ndarray, output_grid: np.ndarray
    ) -> str:
        def describe_grid(grid, grid_name):
            unique, counts = np.unique(grid, return_counts=True)
            color_counts = dict(zip(unique, counts))
            total_cells = self.grid_size * self.grid_size

            color_descriptions = []
            for color in sorted(color_counts.keys()):
                if color != 0:  # Exclude the background color
                    percentage = (color_counts[color] / total_cells) * 100
                    color_descriptions.append(
                        f"{self.color_map[color]} ({color_counts[color]} cells, {percentage:.1f}%)"
                    )

            color_list = (
                ", ".join(color_descriptions[:-1]) + f" and {color_descriptions[-1]}"
                if len(color_descriptions) > 1
                else color_descriptions[0]
            )

            if self.orientation == "rows":
                heights = [np.count_nonzero(row) for row in grid]
                height_description = "row heights (from top to bottom)"
            else:
                heights = [np.count_nonzero(grid[:, i]) for i in range(grid.shape[1])]
                height_description = "column heights (from left to right)"

            return f"""The {grid_name} is a {self.grid_size}x{self.grid_size} square with a black background. 
It contains {self.grid_size} colored {self.orientation} of varying heights.
The colors present are {color_list}.
The {height_description} are: {', '.join(map(str, heights))}.
The colored cells occupy {np.sum(grid > 0)} cells ({(np.sum(grid > 0) / total_cells) * 100:.1f}% of the grid), 
while the remaining {np.sum(grid == 0)} cells ({(np.sum(grid == 0) / total_cells) * 100:.1f}% of the grid) are black background."""

        input_description = describe_grid(input_grid, "input grid")
        output_description = describe_grid(output_grid, "output grid")

        difference = f"""
The main differences between the input and output grids are:
1. The {self.orientation} have been sorted by height in {self.sort_order} order.
2. Each {self.orientation[:-1]} has been {'left' if self.orientation == 'rows' else 'top'}-aligned.
3. The order of colors has changed due to the sorting operation.
4. The total number of cells of each color remains the same, but their distribution across {self.orientation} has changed.
5. The overall pattern of the grid has changed from random heights to a sorted arrangement.
"""

        return f"{input_description}\n\n{output_description}\n\n{difference}"

    def generate_pattern_description(self) -> str:
        return f"""
The pattern that transforms the input grid to the output grid involves sorting and aligning {self.orientation} based on their heights. Specifically:

1. Grid Composition:
   - The grid is {self.grid_size}x{self.grid_size} in size.
   - It contains {self.grid_size} colored {self.orientation}, each with a unique height.
   - The colors used are: {', '.join([self.color_map[color] for color in self.colors])}.

2. Height Measurement:
   - The height of each {self.orientation[:-1]} is determined by the number of colored cells it contains.
   - In the input grid, these heights are in a random order.

3. Sorting Operation:
   - The {self.orientation} are sorted based on their heights in {self.sort_order} order.
   - This means the {self.orientation} with the {'most' if self.sort_order == 'descending' else 'least'} colored cells will be placed {'first' if self.orientation == 'rows' else 'on the left'}.

4. Alignment:
   - After sorting, each {self.orientation[:-1]} is {'left' if self.orientation == 'rows' else 'top'}-aligned.
   - This means all colored cells are moved to the {'left side of their row' if self.orientation == 'rows' else 'top of their column'}, with any empty cells filled with the background color.

5. Color Preservation:
   - The color of each {self.orientation[:-1]} is maintained during the sorting and alignment process.
   - Only the position of the {self.orientation} changes, not their color or the number of colored cells they contain.

6. Pattern Formation:
   - The resulting grid shows a clear progression of {self.orientation} heights, either increasing or decreasing based on the sort order.
   - This creates a distinctive stair-step or pyramid-like pattern in the grid.

This transformation results in a grid where the {self.orientation} are organized by height, creating a visually striking pattern that clearly displays the distribution of colored cells across the grid.
"""


class GravityTask(GridTask):
    def __init__(self, grid_size, colors):
        super().__init__()
        self.grid_size = grid_size
        self.colors = colors

    def sample(self):
        grid = np.random.choice(
            [0] + list(self.colors),
            size=(self.grid_size, self.grid_size),
            p=[0.5] + [0.5 / len(self.colors)] * len(self.colors),
        )
        return grid

    def execute(self, grid: np.ndarray) -> (np.ndarray, str):
        new_grid = np.zeros_like(grid)
        for col in range(grid.shape[1]):
            col_data = grid[:, col]
            col_data = col_data[col_data != 0]  # Filter out the black (0) cells
            new_grid[-len(col_data) :, col] = (
                col_data  # Place at the bottom of the column
            )

        description = self.generate_description(grid, new_grid)
        pattern = self.generate_pattern_description()
        return new_grid, f"{description}\n\n{pattern}"

    def generate_description(
        self, input_grid: np.ndarray, output_grid: np.ndarray
    ) -> str:
        def describe_grid(grid, grid_name):
            unique, counts = np.unique(grid, return_counts=True)
            color_counts = dict(zip(unique, counts))
            total_cells = self.grid_size * self.grid_size

            color_descriptions = []
            for color in sorted(color_counts.keys()):
                if color != 0:  # Exclude the background color
                    percentage = (color_counts[color] / total_cells) * 100
                    color_descriptions.append(
                        f"{self.color_map[color]} ({color_counts[color]} cells, {percentage:.1f}%)"
                    )

            color_list = (
                ", ".join(color_descriptions[:-1]) + f" and {color_descriptions[-1]}"
                if len(color_descriptions) > 1
                else color_descriptions[0]
            )

            return f"""The {grid_name} is a {self.grid_size}x{self.grid_size} square with a black background. 
The colors present are {color_list}.
The colored cells occupy {np.sum(grid > 0)} cells ({(np.sum(grid > 0) / total_cells) * 100:.1f}% of the grid), 
while the remaining {np.sum(grid == 0)} cells ({(np.sum(grid == 0) / total_cells) * 100:.1f}% of the grid) are black background."""

        input_description = describe_grid(input_grid, "input grid")
        output_description = describe_grid(output_grid, "output grid")

        difference = f"""
The main differences between the input and output grids are:
1. All colored cells have "fallen" to the bottom of their respective columns due to gravity.
2. The total number of cells of each color remains the same, but their vertical distribution has changed.
3. The black background cells are now concentrated at the top of the grid.
4. The overall pattern of the grid has changed from a random distribution to a bottom-heavy arrangement.
5. The horizontal distribution of colors remains unchanged, only the vertical positions have been affected.
"""

        return f"{input_description}\n\n{output_description}\n\n{difference}"

    def generate_pattern_description(self) -> str:
        return f"""
The pattern that transforms the input grid to the output grid involves applying a gravity-like effect to all colored cells. Specifically:

1. Grid Composition:
   - The grid is {self.grid_size}x{self.grid_size} in size.
   - It contains colored cells and black background cells.
   - The colors used are: {', '.join([self.color_map[color] for color in self.colors])}.

2. Gravity Effect:
   - Each column of the grid is treated independently.
   - Within each column, all colored cells "fall" to the bottom of the grid.
   - The order of colors within each column is preserved during this fall.

3. Color Preservation:
   - The number of cells of each color remains the same after the transformation.
   - Only the vertical positions of the colored cells change.

4. Background Redistribution:
   - As colored cells fall, the black background cells are pushed to the top of each column.
   - This creates a clear separation between the colored and background sections of the grid.

5. Horizontal Stability:
   - The horizontal distribution of colors remains unchanged.
   - Each column in the output grid contains the same colors as in the input grid, just in a different vertical arrangement.

6. Pattern Formation:
   - The resulting grid shows a concentration of colors at the bottom.
   - This creates a distinctive "settled" or "bottom-heavy" appearance in the grid.

This transformation results in a grid where all colored cells appear to have been affected by gravity, creating a new pattern that clearly separates the colored and background sections of the grid while maintaining the original color distribution within each column.
"""


class BuoyancyTask(GridTask):
    def __init__(self, grid_size, water_level, colors, mode):
        super().__init__()
        self.grid_size = grid_size
        self.water_level = water_level
        self.colors = colors
        self.mode = mode
        self.water_color = np.random.choice(self.colors)

    def sample(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        grid[self.water_level + 1 :, :] = self.water_color

        start_x = 0
        num_objects_placed = 0

        while start_x < self.grid_size and num_objects_placed < 10:
            size = np.random.randint(1, max(2, self.grid_size // 2))
            color = np.random.choice([c for c in self.colors if c != self.water_color])

            if start_x + size > self.grid_size:
                break

            start_y = 0

            while start_y + size <= self.water_level:
                if start_y + size <= self.water_level:
                    grid[start_y : start_y + size, start_x : start_x + size] = color
                    start_y += size
                    num_objects_placed += 1
                else:
                    break

            start_x += size

        return grid

    def execute(self, grid: np.ndarray) -> (np.ndarray, str):
        new_grid = np.copy(grid)

        objects = {}
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (
                    grid[i, j] > 0
                    and grid[i, j] != self.water_color
                    and i <= self.water_level
                ):
                    color = grid[i, j]
                    if color not in objects:
                        objects[color] = []
                    objects[color].append((i, j))

        for color, coords in objects.items():
            if self.mode == "sink":
                max_y = max(coord[0] for coord in coords)
                offset = self.grid_size - 1 - max_y
            elif self.mode == "float_above":
                min_y = min(coord[0] for coord in coords)
                offset = self.water_level - min_y
            elif self.mode == "float_below":
                min_y = min(coord[0] for coord in coords)
                offset = self.water_level + 1 - min_y

            for i, j in coords:
                new_grid[i, j] = 0  # Clear original position
                if i + offset < self.grid_size:
                    new_grid[i + offset, j] = color  # Move to new position

        description = self.generate_description(grid, new_grid)
        pattern = self.generate_pattern_description()
        return new_grid, f"{description}\n\n{pattern}"

    def generate_description(
        self, input_grid: np.ndarray, output_grid: np.ndarray
    ) -> str:
        def describe_grid(grid, grid_name):
            unique, counts = np.unique(grid, return_counts=True)
            color_counts = dict(zip(unique, counts))
            total_cells = self.grid_size * self.grid_size

            color_descriptions = []
            for color in sorted(color_counts.keys()):
                if color != 0:  # Exclude the background color
                    percentage = (color_counts[color] / total_cells) * 100
                    color_descriptions.append(
                        f"{self.color_map[color]} ({color_counts[color]} cells, {percentage:.1f}%)"
                    )

            color_list = (
                ", ".join(color_descriptions[:-1]) + f" and {color_descriptions[-1]}"
                if len(color_descriptions) > 1
                else color_descriptions[0]
            )

            water_description = f"The water level is at row {self.water_level} (counting from 0 at the top), colored {self.color_map[self.water_color]}."

            return f"""The {grid_name} is a {self.grid_size}x{self.grid_size} square with a black background. 
The colors present are {color_list}.
{water_description}
The colored cells (including water) occupy {np.sum(grid > 0)} cells ({(np.sum(grid > 0) / total_cells) * 100:.1f}% of the grid), 
while the remaining {np.sum(grid == 0)} cells ({(np.sum(grid == 0) / total_cells) * 100:.1f}% of the grid) are black background."""

        input_description = describe_grid(input_grid, "input grid")
        output_description = describe_grid(output_grid, "output grid")

        difference = f"""
The main differences between the input and output grids are:
1. The objects (non-water colored cells) have moved according to the '{self.mode}' mode:
   {"- All objects have sunk to the bottom of the grid." if self.mode == "sink" else
    "- All objects have floated up so their tops are just above the water level." if self.mode == "float_above" else
    "- All objects have floated up so their tops are at the water level, fully submerged."}
2. The water level and color remain unchanged.
3. The total number of cells of each color remains the same, but their vertical distribution has changed.
4. The overall pattern of the grid has changed to reflect the new positions of the objects relative to the water.
5. The horizontal positions of the objects remain unchanged, only their vertical positions have been affected.
"""

        return f"{input_description}\n\n{output_description}\n\n{difference}"

    def generate_pattern_description(self) -> str:
        return f"""
The pattern that transforms the input grid to the output grid involves applying a buoyancy effect to objects in water. Specifically:

1. Grid Composition:
   - The grid is {self.grid_size}x{self.grid_size} in size.
   - It contains water (colored {self.color_map[self.water_color]}) and objects of various colors.
   - The water level is fixed at row {self.water_level} (counting from 0 at the top).
   - The colors used for objects are: {', '.join([self.color_map[color] for color in self.colors if color != self.water_color])}.

2. Buoyancy Effect ('{self.mode}' mode):
   {"- All objects sink to the bottom of the grid, regardless of their initial position." if self.mode == "sink" else
    "- All objects float up so that their top edge is just above the water level." if self.mode == "float_above" else
    "- All objects float up so that their top edge is at the water level, fully submerged."}
   - Each column is treated independently during this process.

3. Object Integrity:
   - The shape and size of each object is preserved during the transformation.
   - Objects may be partially cut off if they would extend beyond the grid boundaries after moving.

4. Water Stability:
   - The water level and color remain constant throughout the transformation.

5. Color Preservation:
   - The number of cells of each color (including water) remains the same after the transformation.
   - Only the vertical positions of the non-water colored cells change.

6. Horizontal Stability:
   - The horizontal positions of objects remain unchanged.
   - Each column in the output grid contains the same colors as in the input grid, just in a different vertical arrangement.

7. Pattern Formation:
   - The resulting grid shows a clear separation between water and objects, with objects positioned according to the buoyancy mode.
   - This creates a distinctive pattern that simulates the behavior of objects in water.

This transformation results in a grid where all objects appear to have been affected by buoyancy in water, creating a new pattern that reflects the specified mode of object behavior while maintaining the original water level and overall color distribution.
"""


class ScaleUpShapeTask(GridTask):
    def __init__(self, grid_size, shape_size, color, corner):
        super().__init__()
        self.grid_size = grid_size
        self.shape_size = shape_size
        self.color = color
        self.corner = corner

    def sample(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        shape = generate_shape(self.grid_size, self.shape_size)
        shape[shape == 1] = self.color
        grid = np.maximum(grid, shape)
        return grid

    def execute(self, grid: np.ndarray) -> (np.ndarray, str):
        shape_coords = np.argwhere(grid == self.color)
        if shape_coords.size == 0:
            return grid, "No shape found to scale"

        min_x, min_y = shape_coords.min(axis=0)
        max_x, max_y = shape_coords.max(axis=0)
        shape_component = grid[min_x : max_x + 1, min_y : max_y + 1]

        scaled_shape = np.kron(shape_component, np.ones((2, 2), dtype=int))
        new_grid = np.zeros_like(grid)

        if self.corner == "top-left":
            end_x = min(min_x + scaled_shape.shape[0], self.grid_size)
            end_y = min(min_y + scaled_shape.shape[1], self.grid_size)
            new_grid[min_x:end_x, min_y:end_y] = scaled_shape[
                : end_x - min_x, : end_y - min_y
            ]
        elif self.corner == "top-right":
            end_x = min(min_x + scaled_shape.shape[0], self.grid_size)
            start_y = max(max_y - scaled_shape.shape[1] + 1, 0)
            new_grid[min_x:end_x, start_y : max_y + 1] = scaled_shape[
                : end_x - min_x, -(max_y - start_y + 1) :
            ]
        elif self.corner == "bottom-left":
            start_x = max(max_x - scaled_shape.shape[0] + 1, 0)
            end_y = min(min_y + scaled_shape.shape[1], self.grid_size)
            new_grid[start_x : max_x + 1, min_y:end_y] = scaled_shape[
                -(max_x - start_x + 1) :, : end_y - min_y
            ]
        elif self.corner == "bottom-right":
            start_x = max(max_x - scaled_shape.shape[0] + 1, 0)
            start_y = max(max_y - scaled_shape.shape[1] + 1, 0)
            new_grid[start_x : max_x + 1, start_y : max_y + 1] = scaled_shape[
                -(max_x - start_x + 1) :, -(max_y - start_y + 1) :
            ]

        description = self.generate_description(grid, new_grid)
        pattern = self.generate_pattern_description()
        return new_grid, f"{description}\n\n{pattern}"

    def generate_description(
        self, input_grid: np.ndarray, output_grid: np.ndarray
    ) -> str:
        def describe_grid(grid, grid_name):
            shape_cells = np.sum(grid == self.color)
            total_cells = self.grid_size * self.grid_size

            shape_coords = np.argwhere(grid == self.color)
            if shape_coords.size > 0:
                min_x, min_y = shape_coords.min(axis=0)
                max_x, max_y = shape_coords.max(axis=0)
                shape_width = max_y - min_y + 1
                shape_height = max_x - min_x + 1
                shape_position = f"({min_x}, {min_y})"
            else:
                shape_width = shape_height = 0
                shape_position = "N/A"

            return f"""The {grid_name} is a {self.grid_size}x{self.grid_size} square with a black background. 
It contains a {self.color_map[self.color]} shape of {shape_cells} cells ({(shape_cells / total_cells) * 100:.1f}% of the grid).
The shape's bounding box is {shape_width}x{shape_height}, with the top-left corner at {shape_position}.
The remaining {total_cells - shape_cells} cells ({((total_cells - shape_cells) / total_cells) * 100:.1f}% of the grid) are black background."""

        input_description = describe_grid(input_grid, "input grid")
        output_description = describe_grid(output_grid, "output grid")

        difference = f"""
The main differences between the input and output grids are:
1. The {self.color_map[self.color]} shape has been scaled up by a factor of 2 from the {self.corner} corner.
2. The number of cells occupied by the shape has increased from {np.sum(input_grid == self.color)} to {np.sum(output_grid == self.color)}.
3. The shape's dimensions have doubled, but may be partially cut off if it extends beyond the grid boundaries.
4. The position of the shape has changed relative to the {self.corner} corner of the grid.
5. The overall pattern of the grid has changed, with the scaled shape occupying a larger portion of the grid.
"""

        return f"{input_description}\n\n{output_description}\n\n{difference}"

    def generate_pattern_description(self) -> str:
        return f"""
The pattern that transforms the input grid to the output grid involves scaling up a shape. Specifically:

1. Grid Composition:
   - The grid is {self.grid_size}x{self.grid_size} in size.
   - It contains a single shape of color {self.color_map[self.color]}.
   - The original shape is generated with a maximum size of {self.shape_size}x{self.shape_size}.

2. Scaling Operation:
   - The shape is scaled up by a factor of 2 in both dimensions.
   - The scaling is anchored at the {self.corner} corner of the original shape.

3. Scaling Method:
   - Each cell of the original shape is replaced by a 2x2 block of cells in the scaled shape.
   - This results in the shape's dimensions doubling in both width and height.

4. Position Adjustment:
   - The scaled shape's position is adjusted based on the {self.corner} corner:
    * For "{self.corner}", the shape expands {"down and right" if self.corner == "top-left" else "down and left" if self.corner == "top-right" else "up and right" if self.corner == "bottom-left" else "up and left"}.
5. Color Preservation:
   - The color of the shape ({self.color_map[self.color]}) remains the same after scaling.

6. Grid Boundaries:
   - If the scaled shape would extend beyond the grid boundaries, it is partially cut off.
   - The parts of the scaled shape that fit within the grid are always displayed.

7. Background Preservation:
   - The black background remains in areas not occupied by the scaled shape.

This transformation results in a grid where the original shape appears larger and potentially repositioned, creating a new pattern that emphasizes the scaled shape while maintaining its original color and basic form.
"""


class AddGridOverlayTask(GridTask):
    def __init__(self, grid_size, noise_level, cell_size, grid_color):
        super().__init__()
        self.grid_size = grid_size
        self.noise_level = noise_level
        self.cell_size = cell_size
        self.grid_color = grid_color

    def sample(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # Add noise
        num_noise_cells = int(self.noise_level * self.grid_size * self.grid_size)
        for _ in range(num_noise_cells):
            x, y = np.random.randint(0, self.grid_size, 2)
            noise_color = np.random.choice(range(1, 7))
            grid[x, y] = noise_color

        return grid

    def execute(self, grid: np.ndarray) -> (np.ndarray, str):
        original_size = grid.shape[0]
        new_size = original_size + (original_size // self.cell_size)
        new_grid = np.zeros((new_size, new_size), dtype=int)

        # Copy the original grid into the new grid and insert columns
        for i in range(original_size):
            new_i = i + (i // self.cell_size)
            for j in range(original_size):
                new_j = j + (j // self.cell_size)
                new_grid[new_i, new_j] = grid[i, j]

        # Add vertical grid lines
        for i in range(self.cell_size, new_size, self.cell_size + 1):
            new_grid[:, i] = self.grid_color

        # Add horizontal grid lines
        for i in range(self.cell_size, new_size, self.cell_size + 1):
            new_grid[i, :] = self.grid_color

        description = self.generate_description(grid, new_grid)
        pattern = self.generate_pattern_description()
        return new_grid, f"{description}\n\n{pattern}"

    def generate_description(
        self, input_grid: np.ndarray, output_grid: np.ndarray
    ) -> str:
        def describe_grid(grid, grid_name):
            unique, counts = np.unique(grid, return_counts=True)
            color_counts = dict(zip(unique, counts))
            total_cells = grid.shape[0] * grid.shape[1]

            color_descriptions = []
            for color in sorted(color_counts.keys()):
                if color != 0:  # Exclude the background color
                    percentage = (color_counts[color] / total_cells) * 100
                    color_descriptions.append(
                        f"{self.color_map[color]} ({color_counts[color]} cells, {percentage:.1f}%)"
                    )

            color_list = (
                ", ".join(color_descriptions[:-1]) + f" and {color_descriptions[-1]}"
                if len(color_descriptions) > 1
                else color_descriptions[0]
            )

            return f"""The {grid_name} is a {grid.shape[0]}x{grid.shape[1]} square with a black background. 
The colors present are {color_list}.
The colored cells occupy {np.sum(grid > 0)} cells ({(np.sum(grid > 0) / total_cells) * 100:.1f}% of the grid), 
while the remaining {np.sum(grid == 0)} cells ({(np.sum(grid == 0) / total_cells) * 100:.1f}% of the grid) are black background."""

        input_description = describe_grid(input_grid, "input grid")
        output_description = describe_grid(output_grid, "output grid")

        difference = f"""
The main differences between the input and output grids are:
1. The output grid size has increased from {input_grid.shape[0]}x{input_grid.shape[0]} to {output_grid.shape[0]}x{output_grid.shape[0]}.
2. A grid overlay of color {self.color_map[self.grid_color]} has been added to the output grid.
3. The grid overlay divides the output grid into {self.cell_size}x{self.cell_size} cells.
4. The original content of the input grid is preserved within these cells in the output grid.
5. The number of {self.color_map[self.grid_color]} cells (grid lines) in the output grid is {np.sum(output_grid == self.grid_color)}.
6. The overall pattern of the grid has changed from random noise to a structured grid with the original noise preserved within cells.
"""

        return f"{input_description}\n\n{output_description}\n\n{difference}"

    def generate_pattern_description(self) -> str:
        return f"""
The pattern that transforms the input grid to the output grid involves adding a grid overlay. Specifically:

1. Grid Expansion:
   - The input grid size of {self.grid_size}x{self.grid_size} is expanded to accommodate the grid overlay.
   - The new size is calculated as: original size + (original size // cell size).

2. Content Preservation:
   - The original content (colored noise) from the input grid is preserved in the output grid.
   - Each cell from the input grid is mapped to a corresponding position in the output grid.

3. Grid Overlay Addition:
   - A grid overlay of color {self.color_map[self.grid_color]} is added to the output grid.
   - The grid lines divide the output grid into cells of size {self.cell_size}x{self.cell_size}.
   - Vertical grid lines are added at intervals of {self.cell_size + 1} cells.
   - Horizontal grid lines are added at intervals of {self.cell_size + 1} cells.

4. Noise Distribution:
   - The original noise from the input grid, which occupies approximately {self.noise_level * 100}% of the cells, is maintained within the grid cells of the output.

5. Color Preservation:
   - The colors of the noise cells from the input grid remain unchanged in the output grid.
   - The only new color introduced is the {self.color_map[self.grid_color]} of the grid overlay.

6. Pattern Formation:
   - The resulting output grid shows a structured pattern of {self.cell_size}x{self.cell_size} cells.
   - Each cell contains the original noise pattern from the input grid.
   - The cells are separated by the {self.color_map[self.grid_color]} grid lines.

This transformation results in a grid that preserves the original noise pattern while imposing a structured grid overlay, creating a new pattern that combines randomness within a regular structure.
"""


class RainwaterTask(GridTask):
    def __init__(self, grid_size, max_column_height, column_color, water_color):
        super().__init__()
        self.grid_size = grid_size
        self.max_column_height = max_column_height
        self.column_color = column_color
        self.water_color = water_color

    def sample(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        num_columns = np.random.randint(3, self.grid_size)
        for _ in range(num_columns):
            height = np.random.randint(1, self.max_column_height + 1)
            col = np.random.randint(0, self.grid_size)
            grid[-height:, col] = self.column_color
        # fill last row
        grid[-1, :] = self.column_color
        return grid

    def execute(self, grid: np.ndarray) -> (np.ndarray, str):
        new_grid = np.copy(grid)
        column_heights = np.zeros(self.grid_size, dtype=int)

        for col in range(self.grid_size):
            if np.any(grid[:, col] == self.column_color):
                column_heights[col] = self.grid_size - np.argmax(
                    grid[:, col] == self.column_color
                )

        for col in range(1, self.grid_size - 1):
            left_max = max(column_heights[:col])
            right_max = max(column_heights[col + 1 :])
            if column_heights[col] < left_max and column_heights[col] < right_max:
                fill_height = min(left_max, right_max) - column_heights[col]
                new_grid[
                    -column_heights[col] - fill_height : -column_heights[col], col
                ] = self.water_color

        description = self.generate_description(grid, new_grid)
        pattern = self.generate_pattern_description()
        return new_grid, f"{description}\n\n{pattern}"

    def generate_description(
        self, input_grid: np.ndarray, output_grid: np.ndarray
    ) -> str:
        def describe_grid(grid, grid_name):
            unique, counts = np.unique(grid, return_counts=True)
            color_counts = dict(zip(unique, counts))
            total_cells = self.grid_size * self.grid_size

            color_descriptions = []
            for color in sorted(color_counts.keys()):
                if color != 0:  # Exclude the background color
                    percentage = (color_counts[color] / total_cells) * 100
                    color_descriptions.append(
                        f"{self.color_map[color]} ({color_counts[color]} cells, {percentage:.1f}%)"
                    )

            color_list = (
                ", ".join(color_descriptions[:-1]) + f" and {color_descriptions[-1]}"
                if len(color_descriptions) > 1
                else color_descriptions[0]
            )

            return f"""Looking at the {grid_name}, we see a {self.grid_size}x{self.grid_size} square grid. 
The grid has a black background, which takes up most of the space.
There are {self.color_map[self.column_color]} cells arranged in vertical columns of varying heights.
Interestingly, the bottom row is completely filled with {self.color_map[self.column_color]} cells.
The colors present in this grid are {color_list}.
The colored cells occupy {np.sum(grid > 0)} cells ({(np.sum(grid > 0) / total_cells) * 100:.1f}% of the grid), 
while the remaining {np.sum(grid == 0)} cells ({(np.sum(grid == 0) / total_cells) * 100:.1f}% of the grid) are black background.
This arrangement of columns on a solid base reminds me of a simplified terrain or cityscape."""

        input_description = describe_grid(input_grid, "input grid")
        output_description = describe_grid(output_grid, "output grid")

        water_cells = np.sum(output_grid == self.water_color)
        total_cells = self.grid_size * self.grid_size
        water_percentage = (water_cells / total_cells) * 100

        difference = f"""
When comparing the input and output grids, we notice several key differences:

1. The most striking change is the appearance of {self.color_map[self.water_color]} cells in the output grid. These weren't present in the input grid.
2. These new {self.color_map[self.water_color]} cells seem to be located in the spaces between the {self.color_map[self.column_color]} columns.
3. The {self.color_map[self.water_color]} cells don't fill all the spaces between columns. Instead, they only partially fill these gaps.
4. It appears that the height of the {self.color_map[self.water_color]} cells in each gap is related to the heights of the adjacent columns.
5. Specifically, {water_cells} cells ({water_percentage:.1f}% of the grid) have been filled with this new {self.color_map[self.water_color]} color.
6. The original {self.color_map[self.column_color]} columns and the black background remain unchanged from the input to the output grid.
7. The overall pattern has changed from just columns to columns with something filling the spaces between them, but only up to certain heights.

Given these observations, it seems like the transformation is simulating some kind of filling process between the columns, possibly representing the accumulation of a liquid like water between terrain features.
"""

        return f"{input_description}\n\n{output_description}\n\n{difference}"

    def generate_pattern_description(self) -> str:
        return f"""
After observing the changes between the input and output grids, we can describe the pattern of transformation as follows:

1. Grid Structure:
   - We start with a {self.grid_size}x{self.grid_size} grid.
   - The grid contains vertical columns of {self.color_map[self.column_color]} cells.
   - These columns have varying heights, but none exceed {self.max_column_height} cells.
   - The bottom row is always filled with {self.color_map[self.column_color]} cells, forming a base.

2. New Element Introduction:
   - In the output grid, we see the introduction of {self.color_map[self.water_color]} cells.
   - These new cells appear only in the spaces between the existing {self.color_map[self.column_color]} columns.

3. Fill Pattern:
   - The {self.color_map[self.water_color]} cells don't completely fill the gaps between columns.
   - Instead, they fill these gaps partially, up to a certain height.
   - This height seems to be determined by the columns on either side of the gap.

4. Height Determination:
   - For each gap, the fill height appears to be related to the shorter of the two adjacent columns.
   - The {self.color_map[self.water_color]} cells never overflow the shorter column.

5. Column Preservation:
   - The original {self.color_map[self.column_color]} columns remain unchanged in the transformation.
   - The {self.color_map[self.water_color]} cells are only added to the empty spaces, never replacing existing column cells.

6. Resulting Pattern:
   - The output grid shows a pattern that looks like {self.color_map[self.column_color]} terrain with {self.color_map[self.water_color]} liquid trapped between the elevations.

Given these observations, we can infer that this transformation is likely simulating the accumulation of a liquid (probably water, given the color) between columns of varying heights. This could represent rainwater or flood water collecting in a landscape with different elevations.

The pattern effectively transforms a "dry" terrain into one where water has accumulated in the low-lying areas between higher points, but only up to the level of the lowest surrounding "barrier".
"""


class BorderAdditionTask(GridTask):
    def __init__(self, grid_size, num_shapes, colors, border_color):
        super().__init__()
        self.grid_size = grid_size
        self.num_shapes = num_shapes
        self.colors = colors
        self.border_color = border_color

    def sample(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for color in self.colors:
            shape = generate_shape(self.grid_size, self.grid_size // 4)
            shape[shape == 1] = color
            grid = np.maximum(grid, shape)
        return grid

    def execute(self, grid: np.ndarray) -> (np.ndarray, str):
        new_grid = np.copy(grid)

        # Find all exterior cells using flood fill
        exterior = np.zeros_like(grid, dtype=bool)

        def flood_fill(x, y):
            stack = [(x, y)]
            while stack:
                cx, cy = stack.pop()
                if cx < 0 or cx >= self.grid_size or cy < 0 or cy >= self.grid_size:
                    continue
                if exterior[cx, cy] or grid[cx, cy] > 0:
                    continue
                exterior[cx, cy] = True
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    stack.append((cx + dx, cy + dy))

        for i in range(self.grid_size):
            if grid[i, 0] == 0:
                flood_fill(i, 0)
            if grid[i, self.grid_size - 1] == 0:
                flood_fill(i, self.grid_size - 1)
        for j in range(self.grid_size):
            if grid[0, j] == 0:
                flood_fill(0, j)
            if grid[self.grid_size - 1, j] == 0:
                flood_fill(self.grid_size - 1, j)

        # Add border around shapes by checking exterior cells
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if grid[i, j] > 0:
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + dx, j + dy
                        if (
                            0 <= ni < self.grid_size
                            and 0 <= nj < self.grid_size
                            and exterior[ni, nj]
                        ):
                            new_grid[ni, nj] = self.border_color

        description = self.generate_description(grid, new_grid)
        pattern = self.generate_pattern_description()
        return new_grid, f"{description}\n\n{pattern}"

    def generate_description(
        self, input_grid: np.ndarray, output_grid: np.ndarray
    ) -> str:
        def describe_grid(grid, grid_name):
            unique, counts = np.unique(grid, return_counts=True)
            color_counts = dict(zip(unique, counts))
            total_cells = self.grid_size * self.grid_size

            color_descriptions = []
            for color in sorted(color_counts.keys()):
                if color != 0:  # Exclude the background color
                    percentage = (color_counts[color] / total_cells) * 100
                    color_descriptions.append(
                        f"{self.color_map[color]} ({color_counts[color]} cells, {percentage:.1f}%)"
                    )

            color_list = (
                ", ".join(color_descriptions[:-1]) + f" and {color_descriptions[-1]}"
                if len(color_descriptions) > 1
                else color_descriptions[0]
            )

            return f"""Looking at the {grid_name}, we see a {self.grid_size}x{self.grid_size} square grid. 
The grid has a black background, which forms the majority of the space.
There are {self.num_shapes} distinct colored shape{'s' if self.num_shapes > 1 else ''} on this background.
The colors present in this grid are {color_list}.
The colored cells occupy {np.sum(grid > 0)} cells ({(np.sum(grid > 0) / total_cells) * 100:.1f}% of the grid), 
while the remaining {np.sum(grid == 0)} cells ({(np.sum(grid == 0) / total_cells) * 100:.1f}% of the grid) are black background.
The shapes appear to be randomly positioned and have irregular forms."""

        input_description = describe_grid(input_grid, "input grid")
        output_description = describe_grid(output_grid, "output grid")

        border_cells = np.sum(output_grid == self.border_color)
        total_cells = self.grid_size * self.grid_size
        border_percentage = (border_cells / total_cells) * 100

        difference = f"""
When comparing the input and output grids, we notice several key differences:

1. The most noticeable change is the appearance of {self.color_map[self.border_color]} cells in the output grid. These weren't present in the input grid.
2. These new {self.color_map[self.border_color]} cells seem to be located around the edges of the existing shapes.
3. The {self.color_map[self.border_color]} cells form a continuous line around each shape, like an outline or border.
4. The original shapes remain intact, with their colors and positions unchanged.
5. Specifically, {border_cells} cells ({border_percentage:.1f}% of the grid) have been filled with this new {self.color_map[self.border_color]} color.
6. The black background cells that aren't adjacent to any shape remain unchanged.
7. The overall pattern has changed from distinct shapes to shapes with a clear border around them.

Given these observations, it appears that the transformation is adding a border or outline to each shape in the grid, using a consistent color for all borders.
"""

        return f"{input_description}\n\n{output_description}\n\n{difference}"

    def generate_pattern_description(self) -> str:
        return f"""
After carefully observing the changes between the input and output grids, we can describe the pattern of transformation as follows:

1. Grid Structure:
   - We start with a {self.grid_size}x{self.grid_size} grid.
   - The grid contains {self.num_shapes} colored shape{'s' if self.num_shapes > 1 else ''} on a black background.
   - These shapes have irregular forms and are randomly positioned.

2. New Element Introduction:
   - In the output grid, we see the introduction of {self.color_map[self.border_color]} cells.
   - These new cells appear only around the edges of the existing shapes.

3. Border Pattern:
   - The {self.color_map[self.border_color]} cells form a continuous line around each shape.
   - This line is exactly one cell thick.
   - The border follows the contours of each shape precisely.

4. Border Placement:
   - The border is added to the cells that are adjacent to the shape cells.
   - It's placed in the black background cells that touch the shape, not on top of the shape itself.
   - This means the border is always outside the original shape.

5. Shape Preservation:
   - The original shapes remain completely unchanged in the transformation.
   - Their colors, sizes, and positions are maintained.

6. Consistent Border Color:
   - All borders, regardless of the shape they surround, use the same {self.color_map[self.border_color]} color.

7. Resulting Pattern:
   - The output grid shows the original shapes, now each surrounded by a {self.color_map[self.border_color]} border.
   - This creates a clear visual separation between the shapes and the background.

Given these observations, we can infer that this transformation is adding a border or outline to each shape in the grid. This border helps to define the shapes more clearly and separate them from the background.

The pattern effectively transforms a grid of distinct shapes into one where each shape is highlighted by a surrounding border, enhancing the visual definition of the shapes within the grid.
"""

    def generate_pattern_description(self) -> str:
        return f"""
After carefully observing the changes between the input and output grids, we can describe the pattern of transformation as follows:

1. Grid Structure:
   - We start with a {self.grid_size}x{self.grid_size} grid.
   - The grid contains {self.num_shapes} colored shape{'s' if self.num_shapes > 1 else ''} on a black background.
   - These shapes have irregular forms and are randomly positioned.

2. New Element Introduction:
   - In the output grid, we see the introduction of {self.color_map[self.border_color]} cells.
   - These new cells appear only around the edges of the existing shapes.

3. Border Pattern:
   - The {self.color_map[self.border_color]} cells form a continuous line around each shape.
   - This line is exactly one cell thick.
   - The border follows the contours of each shape precisely.

4. Border Placement:
   - The border is added to the cells that are adjacent to the shape cells.
   - It's placed in the black background cells that touch the shape, not on top of the shape itself.
   - This means the border is always outside the original shape.

5. Shape Preservation:
   - The original shapes remain completely unchanged in the transformation.
   - Their colors, sizes, and positions are maintained.

6. Consistent Border Color:
   - All borders, regardless of the shape they surround, use the same {self.color_map[self.border_color]} color.

7. Resulting Pattern:
   - The output grid shows the original shapes, now each surrounded by a {self.color_map[self.border_color]} border.
   - This creates a clear visual separation between the shapes and the background.

Given these observations, we can infer that this transformation is adding a border or outline to each shape in the grid. This border helps to define the shapes more clearly and separate them from the background.

The pattern effectively transforms a grid of distinct shapes into one where each shape is highlighted by a surrounding border, enhancing the visual definition of the shapes within the grid.
"""


class ShapeMergingTask(GridTask):
    def __init__(self, grid_size, num_shapes, colors, merged_color):
        super().__init__()
        self.grid_size = grid_size
        self.num_shapes = num_shapes
        self.colors = colors
        self.merged_color = merged_color

    def sample(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for color in self.colors:
            shape = generate_shape(self.grid_size, self.grid_size // 4)
            shape[shape == 1] = color
            grid = np.maximum(grid, shape)
        return grid

    def execute(self, grid: np.ndarray) -> (np.ndarray, str):
        new_grid = np.copy(grid)

        # Function to perform DFS and find connected components
        def dfs(x, y, original_color):
            stack = [(x, y)]
            shape_coords = []
            while stack:
                cx, cy = stack.pop()
                if (
                    (cx, cy) in visited
                    or cx < 0
                    or cx >= self.grid_size
                    or cy < 0
                    or cy >= self.grid_size
                ):
                    continue
                if grid[cx, cy] == original_color:
                    visited.add((cx, cy))
                    shape_coords.append((cx, cy))
                    stack.extend(
                        [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)]
                    )
            return shape_coords

        visited = set()
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if grid[i, j] > 0 and (i, j) not in visited:
                    shape_coords = dfs(i, j, grid[i, j])
                    for x, y in shape_coords:
                        new_grid[x, y] = self.merged_color

        description = self.generate_description(grid, new_grid)
        pattern = self.generate_pattern_description()
        return new_grid, f"{description}\n\n{pattern}"

    def generate_description(
        self, input_grid: np.ndarray, output_grid: np.ndarray
    ) -> str:
        def describe_grid(grid, grid_name):
            unique, counts = np.unique(grid, return_counts=True)
            color_counts = dict(zip(unique, counts))
            total_cells = self.grid_size * self.grid_size

            color_descriptions = []
            for color in sorted(color_counts.keys()):
                if color != 0:  # Exclude the background color
                    percentage = (color_counts[color] / total_cells) * 100
                    color_descriptions.append(
                        f"{self.color_map[color]} ({color_counts[color]} cells, {percentage:.1f}%)"
                    )

            color_list = (
                ", ".join(color_descriptions[:-1]) + f" and {color_descriptions[-1]}"
                if len(color_descriptions) > 1
                else color_descriptions[0]
            )

            return f"""Looking at the {grid_name}, we see a {self.grid_size}x{self.grid_size} square grid. 
The grid has a black background, which forms the majority of the space.
There are colored cells forming various shapes on this background.
The colors present in this grid are {color_list}.
The colored cells occupy {np.sum(grid > 0)} cells ({(np.sum(grid > 0) / total_cells) * 100:.1f}% of the grid), 
while the remaining {np.sum(grid == 0)} cells ({(np.sum(grid == 0) / total_cells) * 100:.1f}% of the grid) are black background.
The shapes appear to be randomly positioned and have irregular forms. Some shapes might be overlapping or touching each other."""

        input_description = describe_grid(input_grid, "input grid")
        output_description = describe_grid(output_grid, "output grid")

        merged_cells = np.sum(output_grid == self.merged_color)
        total_cells = self.grid_size * self.grid_size
        merged_percentage = (merged_cells / total_cells) * 100

        difference = f"""
When comparing the input and output grids, we notice several key differences:

1. The most striking change is that all the colored cells in the output grid are now {self.color_map[self.merged_color]}.
2. The original variety of colors seen in the input grid has been replaced by a single color.
3. The overall shape formed by the colored cells seems to be preserved, but now it's all one color.
4. The black background cells remain unchanged.
5. Specifically, {merged_cells} cells ({merged_percentage:.1f}% of the grid) are now {self.color_map[self.merged_color]}.
6. The boundaries between different colored shapes in the input grid are no longer visible in the output grid.
7. Any gaps or separations between shapes of the same color in the input grid are still present in the output grid.

Given these observations, it appears that the transformation is merging all the colored shapes into a single color, while maintaining their overall structure and position in the grid.
"""

        return f"{input_description}\n\n{output_description}\n\n{difference}"

    def generate_pattern_description(self) -> str:
        return f"""
After carefully observing the changes between the input and output grids, we can describe the pattern of transformation as follows:

1. Grid Structure:
   - We start with a {self.grid_size}x{self.grid_size} grid.
   - The input grid contains multiple shapes of various colors on a black background.
   - These shapes have irregular forms and are randomly positioned, with some possibly overlapping or touching.

2. Color Unification:
   - In the output grid, all colored cells have been changed to a single color: {self.color_map[self.merged_color]}.
   - This color change affects all non-black cells, regardless of their original color.

3. Shape Preservation:
   - The overall structure and position of the shapes remain unchanged.
   - The boundaries of the shapes, where they meet the black background, are maintained.

4. Merging Effect:
   - Where different colored shapes were touching or overlapping in the input grid, they now appear as a single, continuous shape in the output grid.
   - However, any gaps or separations between shapes of the same color in the input grid are still present in the output grid.

5. Background Preservation:
   - The black background cells remain unchanged in their position and color.

6. Loss of Internal Boundaries:
   - Any internal boundaries between touching shapes of different colors in the input grid are no longer visible in the output grid.

7. Resulting Pattern:
   - The output grid shows what appears to be a single, complex shape (or multiple disconnected shapes) all in {self.color_map[self.merged_color]}.
   - This creates a unified visual entity from what were previously distinct colored shapes.

Given these observations, we can infer that this transformation is merging all colored shapes into a single color entity. This process unifies the visual appearance of the shapes while maintaining their overall structure and position within the grid.

The pattern effectively transforms a grid with multiple, distinctly colored shapes into one where all these shapes are visually unified by a single color, potentially creating new, larger composite shapes in the process.
"""


class ShapeMergingTask(GridTask):
    def __init__(self, grid_size, num_shapes, colors, merged_color):
        super().__init__()
        self.grid_size = grid_size
        self.num_shapes = num_shapes
        self.colors = colors
        self.merged_color = merged_color

    def sample(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for color in self.colors:
            shape = generate_shape(self.grid_size, self.grid_size // 4)
            shape[shape == 1] = color
            grid = np.maximum(grid, shape)
        return grid

    def execute(self, grid: np.ndarray) -> (np.ndarray, str):
        new_grid = np.copy(grid)

        # Function to perform DFS and find connected components
        def dfs(x, y, original_color):
            stack = [(x, y)]
            shape_coords = []
            while stack:
                cx, cy = stack.pop()
                if (
                    (cx, cy) in visited
                    or cx < 0
                    or cx >= self.grid_size
                    or cy < 0
                    or cy >= self.grid_size
                ):
                    continue
                if grid[cx, cy] == original_color:
                    visited.add((cx, cy))
                    shape_coords.append((cx, cy))
                    stack.extend(
                        [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)]
                    )
            return shape_coords

        visited = set()
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if grid[i, j] > 0 and (i, j) not in visited:
                    shape_coords = dfs(i, j, grid[i, j])
                    for x, y in shape_coords:
                        new_grid[x, y] = self.merged_color

        description = self.generate_description(grid, new_grid)
        pattern = self.generate_pattern_description()
        return new_grid, f"{description}\n\n{pattern}"

    def generate_description(
        self, input_grid: np.ndarray, output_grid: np.ndarray
    ) -> str:
        def describe_grid(grid, grid_name):
            unique, counts = np.unique(grid, return_counts=True)
            color_counts = dict(zip(unique, counts))
            total_cells = self.grid_size * self.grid_size

            color_descriptions = []
            for color in sorted(color_counts.keys()):
                if color != 0:  # Exclude the background color
                    percentage = (color_counts[color] / total_cells) * 100
                    color_descriptions.append(
                        f"{self.color_map[color]} ({color_counts[color]} cells, {percentage:.1f}%)"
                    )

            color_list = (
                ", ".join(color_descriptions[:-1]) + f" and {color_descriptions[-1]}"
                if len(color_descriptions) > 1
                else color_descriptions[0]
            )

            return f"""Looking at the {grid_name}, we see a {self.grid_size}x{self.grid_size} square grid. 
The grid has a black background, which forms the majority of the space.
There are colored cells forming various shapes on this background.
The colors present in this grid are {color_list}.
The colored cells occupy {np.sum(grid > 0)} cells ({(np.sum(grid > 0) / total_cells) * 100:.1f}% of the grid), 
while the remaining {np.sum(grid == 0)} cells ({(np.sum(grid == 0) / total_cells) * 100:.1f}% of the grid) are black background.
The shapes appear to be randomly positioned and have irregular forms. Some shapes might be overlapping or touching each other."""

        input_description = describe_grid(input_grid, "input grid")
        output_description = describe_grid(output_grid, "output grid")

        merged_cells = np.sum(output_grid == self.merged_color)
        total_cells = self.grid_size * self.grid_size
        merged_percentage = (merged_cells / total_cells) * 100

        difference = f"""
When comparing the input and output grids, we notice several key differences:

1. The most striking change is that all the colored cells in the output grid are now {self.color_map[self.merged_color]}.
2. The original variety of colors seen in the input grid has been replaced by a single color.
3. The overall shape formed by the colored cells seems to be preserved, but now it's all one color.
4. The black background cells remain unchanged.
5. Specifically, {merged_cells} cells ({merged_percentage:.1f}% of the grid) are now {self.color_map[self.merged_color]}.
6. The boundaries between different colored shapes in the input grid are no longer visible in the output grid.
7. Any gaps or separations between shapes of the same color in the input grid are still present in the output grid.

Given these observations, it appears that the transformation is merging all the colored shapes into a single color, while maintaining their overall structure and position in the grid.
"""

        return f"{input_description}\n\n{output_description}\n\n{difference}"

    def generate_pattern_description(self) -> str:
        return f"""
After carefully observing the changes between the input and output grids, we can describe the pattern of transformation as follows:

1. Grid Structure:
   - We start with a {self.grid_size}x{self.grid_size} grid.
   - The input grid contains multiple shapes of various colors on a black background.
   - These shapes have irregular forms and are randomly positioned, with some possibly overlapping or touching.

2. Color Unification:
   - In the output grid, all colored cells have been changed to a single color: {self.color_map[self.merged_color]}.
   - This color change affects all non-black cells, regardless of their original color.

3. Shape Preservation:
   - The overall structure and position of the shapes remain unchanged.
   - The boundaries of the shapes, where they meet the black background, are maintained.

4. Merging Effect:
   - Where different colored shapes were touching or overlapping in the input grid, they now appear as a single, continuous shape in the output grid.
   - However, any gaps or separations between shapes of the same color in the input grid are still present in the output grid.

5. Background Preservation:
   - The black background cells remain unchanged in their position and color.

6. Loss of Internal Boundaries:
   - Any internal boundaries between touching shapes of different colors in the input grid are no longer visible in the output grid.

7. Resulting Pattern:
   - The output grid shows what appears to be a single, complex shape (or multiple disconnected shapes) all in {self.color_map[self.merged_color]}.
   - This creates a unified visual entity from what were previously distinct colored shapes.

Given these observations, we can infer that this transformation is merging all colored shapes into a single color entity. This process unifies the visual appearance of the shapes while maintaining their overall structure and position within the grid.

The pattern effectively transforms a grid with multiple, distinctly colored shapes into one where all these shapes are visually unified by a single color, potentially creating new, larger composite shapes in the process.
"""


class AddNoiseAndLinesTask(GridTask):
    def __init__(self, grid_size, noise_level, num_lines, line_color, new_line_color):
        super().__init__()
        self.grid_size = grid_size
        self.noise_level = noise_level
        self.num_lines = num_lines
        self.line_color = line_color
        self.new_line_color = new_line_color

    def sample(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # Add noise
        num_noise_cells = int(self.noise_level * self.grid_size * self.grid_size)
        for _ in range(num_noise_cells):
            x, y = np.random.randint(0, self.grid_size, 2)
            noise_color = np.random.choice(range(3, 7))
            grid[x, y] = noise_color

        # Ensure at least one column padding between lines
        available_columns = list(range(0, self.grid_size, 2))
        np.random.shuffle(available_columns)
        line_positions = available_columns[: self.num_lines]
        line_positions.sort()

        for start_col in line_positions:
            if start_col < self.grid_size - 1:
                start_row = np.random.randint(0, self.grid_size - 1)
                grid[start_row, start_col] = self.line_color
                grid[start_row + 1, start_col] = self.line_color

        return grid

    def execute(self, grid: np.ndarray) -> (np.ndarray, str):
        new_grid = np.copy(grid)

        for row in range(self.grid_size - 1):
            for col in range(self.grid_size):
                if (
                    grid[row, col] == self.line_color
                    and grid[row + 1, col] == self.line_color
                ):
                    new_grid[row, col] = self.new_line_color
                    new_grid[row + 1, col] = 0
                    if col + 1 < self.grid_size:
                        new_grid[row, col + 1] = self.new_line_color

        description = self.generate_description(grid, new_grid)
        pattern = self.generate_pattern_description()
        return new_grid, f"{description}\n\n{pattern}"

    def generate_description(
        self, input_grid: np.ndarray, output_grid: np.ndarray
    ) -> str:
        def describe_grid(grid, grid_name):
            unique, counts = np.unique(grid, return_counts=True)
            color_counts = dict(zip(unique, counts))
            total_cells = self.grid_size * self.grid_size

            color_descriptions = []
            for color in sorted(color_counts.keys()):
                if color != 0:  # Exclude the background color
                    percentage = (color_counts[color] / total_cells) * 100
                    color_descriptions.append(
                        f"{self.color_map[color]} ({color_counts[color]} cells, {percentage:.1f}%)"
                    )

            color_list = (
                ", ".join(color_descriptions[:-1]) + f" and {color_descriptions[-1]}"
                if len(color_descriptions) > 1
                else color_descriptions[0]
            )

            return f"""Looking at the {grid_name}, we see a {self.grid_size}x{self.grid_size} square grid. 
The grid has a black background, which forms the majority of the space.
There are colored cells scattered throughout the grid, which appear to be of two types:
1. Random single cells of various colors, which look like noise.
2. Short vertical lines of {self.color_map[self.line_color]} color, each consisting of two cells.
The colors present in this grid are {color_list}.
The colored cells occupy {np.sum(grid > 0)} cells ({(np.sum(grid > 0) / total_cells) * 100:.1f}% of the grid), 
while the remaining {np.sum(grid == 0)} cells ({(np.sum(grid == 0) / total_cells) * 100:.1f}% of the grid) are black background."""

        input_description = describe_grid(input_grid, "input grid")
        output_description = describe_grid(output_grid, "output grid")

        difference = f"""
When comparing the input and output grids, we notice several key differences:

1. The most noticeable change is that the short vertical {self.color_map[self.line_color]} lines in the input grid have been replaced by short horizontal {self.color_map[self.new_line_color]} lines in the output grid.
2. Each vertical line of two {self.color_map[self.line_color]} cells has been transformed into a horizontal line of two {self.color_map[self.new_line_color]} cells.
3. The transformation seems to pivot around the top cell of each vertical line:
   - The top cell changes from {self.color_map[self.line_color]} to {self.color_map[self.new_line_color]}.
   - The bottom cell of the vertical line becomes black (empty).
   - A new {self.color_map[self.new_line_color]} cell appears to the right of the top cell.
4. The random colored noise cells remain unchanged in both position and color.
5. The total number of {self.color_map[self.line_color]} cells in the input grid equals the total number of {self.color_map[self.new_line_color]} cells in the output grid.
6. The overall distribution of colored cells has shifted slightly due to this transformation.

This transformation appears to be rotating and changing the color of specific line segments in the grid, while leaving other elements untouched.
"""

        return f"{input_description}\n\n{output_description}\n\n{difference}"

    def generate_pattern_description(self) -> str:
        return f"""
After carefully observing the changes between the input and output grids, we can describe the pattern of transformation as follows:

1. Grid Structure:
   - We start with a {self.grid_size}x{self.grid_size} grid.
   - The grid contains random noise of various colors and short vertical lines of a specific color.

2. Noise Preservation:
   - All random colored noise cells remain unchanged in both position and color.

3. Line Transformation:
   - The transformation focuses on the short vertical lines of {self.color_map[self.line_color]} color.
   - Each vertical line, consisting of two cells (top and bottom), is changed in the following way:
     a. The top cell changes color from {self.color_map[self.line_color]} to {self.color_map[self.new_line_color]}.
     b. The bottom cell becomes black (empty).
     c. A new {self.color_map[self.new_line_color]} cell is added to the right of the top cell.

4. Rotation Effect:
   - This transformation effectively rotates each vertical line 90 degrees clockwise.
   - The pivot point for this rotation is the top-left corner of the original vertical line.

5. Color Change:
   - Along with the rotation, the color of the line changes from {self.color_map[self.line_color]} to {self.color_map[self.new_line_color]}.

6. Consistent Application:
   - This pattern is applied consistently to all vertical {self.color_map[self.line_color]} lines in the grid.

7. Spatial Consideration:
   - If a vertical line is at the right edge of the grid, the transformation might result in the right cell of the new horizontal line being cut off.

8. Resulting Pattern:
   - The output grid shows the same random noise as the input, but with short horizontal {self.color_map[self.new_line_color]} lines replacing the original vertical {self.color_map[self.line_color]} lines.

This transformation effectively rotates and recolors specific elements of the grid while maintaining its overall structure and noise distribution. It creates a subtle but noticeable change in the pattern of the grid, shifting from vertical to horizontal line segments amidst the background noise.
"""


import numpy as np


class SymmetryCreationTask(GridTask):
    def __init__(self, grid_size, shape_color, line_color):
        super().__init__()
        self.grid_size = grid_size
        self.shape_color = shape_color
        self.line_color = line_color

    def sample(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # Create a random shape on the left half
        shape_size = self.grid_size // 2 - 1
        shape = generate_shape(shape_size, shape_size // 2)
        shape[shape == 1] = self.shape_color

        # Place the shape on the left side of the grid
        start_x = np.random.randint(0, self.grid_size - shape_size)
        grid[start_x : start_x + shape_size, :shape_size] = shape

        # Add the dividing line
        mid = self.grid_size // 2
        grid[:, mid] = self.line_color

        return grid

    def execute(self, grid: np.ndarray) -> (np.ndarray, str):
        new_grid = np.copy(grid)
        mid = self.grid_size // 2

        # Remove the dividing line
        new_grid[:, mid] = 0

        # Mirror the left half to the right half
        new_grid[:, mid + 1 :] = np.fliplr(new_grid[:, :mid])

        description = self.generate_description(grid, new_grid)
        pattern = self.generate_pattern_description()
        return new_grid, f"{description}\n\n{pattern}"

    def generate_description(
        self, input_grid: np.ndarray, output_grid: np.ndarray
    ) -> str:
        def describe_grid(grid, grid_name):
            unique, counts = np.unique(grid, return_counts=True)
            color_counts = dict(zip(unique, counts))
            total_cells = self.grid_size * self.grid_size

            color_descriptions = []
            for color in sorted(color_counts.keys()):
                if color != 0:  # Exclude the background color
                    percentage = (color_counts[color] / total_cells) * 100
                    color_descriptions.append(
                        f"{self.color_map[color]} ({color_counts[color]} cells, {percentage:.1f}%)"
                    )

            color_list = (
                ", ".join(color_descriptions[:-1]) + f" and {color_descriptions[-1]}"
                if len(color_descriptions) > 1
                else color_descriptions[0]
            )

            if grid_name == "input grid":
                shape_description = (
                    f"a {self.color_map[self.shape_color]} shape on the left half"
                )
                line_description = f"a vertical {self.color_map[self.line_color]} line exactly in the middle"
                return f"""Looking at the {grid_name}, we see a {self.grid_size}x{self.grid_size} square grid. 
The grid is divided into two halves by {line_description}, which runs from top to bottom.
On the left side of this line, we see {shape_description}. This shape appears to be randomly generated and positioned.
The right side of the grid is completely black, creating a stark contrast with the left side.
The colors present in this grid are {color_list}.
The colored cells (including the line) occupy {np.sum(grid > 0)} cells ({(np.sum(grid > 0) / total_cells) * 100:.1f}% of the grid), 
while the remaining {np.sum(grid == 0)} cells ({(np.sum(grid == 0) / total_cells) * 100:.1f}% of the grid) are black background.
This arrangement gives the grid a distinctly asymmetrical appearance."""
            else:
                return f"""Looking at the {grid_name}, we see a {self.grid_size}x{self.grid_size} square grid. 
The grid has a black background, with a symmetrical {self.color_map[self.shape_color]} shape spanning both halves of the grid.
The colors present in this grid are {color_list}.
The colored cells occupy {np.sum(grid > 0)} cells ({(np.sum(grid > 0) / total_cells) * 100:.1f}% of the grid), 
while the remaining {np.sum(grid == 0)} cells ({(np.sum(grid == 0) / total_cells) * 100:.1f}% of the grid) are black background.
The shape appears to be perfectly mirrored across the vertical center line of the grid."""

        input_description = describe_grid(input_grid, "input grid")
        output_description = describe_grid(output_grid, "output grid")

        difference = f"""
When comparing the input and output grids, we notice several key differences:

1. The most obvious change is the removal of the {self.color_map[self.line_color]} vertical line that was in the middle of the input grid. This line is completely absent in the output grid.
2. The {self.color_map[self.shape_color]} shape, which was only on the left half of the input grid, now appears on both halves of the output grid.
3. The right half of the output grid, which was entirely black in the input, now contains a mirrored version of the shape from the left half.
4. The total number of {self.color_map[self.shape_color]} cells has approximately doubled, while the {self.color_map[self.line_color]} cells have disappeared entirely.
5. The overall structure of the grid has changed from a distinctly asymmetrical design (with a clear divide between left and right) to a symmetrical one.
6. The center of the grid, where the line used to be, now forms the axis of symmetry for the shape.

This transformation has created a perfect left-right symmetry in the output grid, based on the shape present in the left half of the input grid, while removing the dividing line.
"""

        return f"{input_description}\n\n{output_description}\n\n{difference}"

    def generate_pattern_description(self) -> str:
        return f"""
After carefully observing the changes between the input and output grids, we can describe the pattern of transformation as follows:

1. Grid Structure:
   - We start with a {self.grid_size}x{self.grid_size} grid.
   - The input grid is clearly divided into two halves by a {self.color_map[self.line_color]} vertical line in the middle.
   - The left half contains a {self.color_map[self.shape_color]} shape, while the right half is entirely black.

2. Line Removal:
   - The {self.color_map[self.line_color]} vertical line in the middle of the grid is completely removed.
   - This erases the clear division between the left and right halves of the grid.

3. Shape Mirroring:
   - The {self.color_map[self.shape_color]} shape on the left half of the input grid is used as a basis for the symmetry.
   - This shape is mirrored onto the right half of the grid, filling the previously empty space.

4. Symmetry Creation:
   - The mirroring process creates perfect left-right symmetry in the output grid.
   - Every {self.color_map[self.shape_color]} cell on the left has a corresponding {self.color_map[self.shape_color]} cell on the right, equidistant from the center line.

5. Background Preservation:
   - The black background cells that are not part of the shape remain unchanged.
   - This includes cells that were originally black on both the left and right sides.

6. Consistency:
   - The mirroring is applied consistently across the entire height of the grid.
   - The vertical center line, where the {self.color_map[self.line_color]} line used to be, becomes the axis of symmetry.

7. Resulting Pattern:
   - The output grid shows a symmetrical {self.color_map[self.shape_color]} shape or pattern, balanced equally on both sides of the center.
   - The grid no longer has any {self.color_map[self.line_color]} cells, only {self.color_map[self.shape_color]} and black.

This transformation effectively takes an asymmetrical input with a clear dividing line and converts it into a symmetrical output by mirroring the left half of the shape and removing the dividing line. It demonstrates how symmetry can be created from partial information, using mirroring as the key operation and treating the former dividing line as a new axis of symmetry.
"""


import numpy as np


class VennDiagramTask(GridTask):
    def __init__(self, grid_size, num_squares, colors):
        super().__init__()
        self.grid_size = grid_size
        self.num_squares = num_squares
        self.colors = colors
        self.intersection_color = max(colors) + 1  # New color for intersections

    def sample(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        for i in range(self.num_squares):
            size = np.random.randint(self.grid_size // 4, self.grid_size // 2)
            color = self.colors[i]
            start_x = np.random.randint(0, self.grid_size - size)
            start_y = np.random.randint(0, self.grid_size - size)
            grid[start_x : start_x + size, start_y : start_y + size] = color

        return grid

    def execute(self, grid: np.ndarray) -> (np.ndarray, str):
        new_grid = np.copy(grid)

        # Find intersections
        intersection = np.zeros_like(grid, dtype=bool)
        for color in self.colors:
            intersection |= grid == color

        # Color intersections
        new_grid[np.where((intersection) & (grid == 0))] = self.intersection_color

        description = self.generate_description(grid, new_grid)
        pattern = self.generate_pattern_description()
        return new_grid, f"{description}\n\n{pattern}"

    def generate_description(
        self, input_grid: np.ndarray, output_grid: np.ndarray
    ) -> str:
        def describe_grid(grid, grid_name):
            unique, counts = np.unique(grid, return_counts=True)
            color_counts = dict(zip(unique, counts))
            total_cells = self.grid_size * self.grid_size

            color_descriptions = []
            for color in sorted(color_counts.keys()):
                if color != 0:  # Exclude the background color
                    percentage = (color_counts[color] / total_cells) * 100
                    color_descriptions.append(
                        f"{self.color_map[color]} ({color_counts[color]} cells, {percentage:.1f}%)"
                    )

            color_list = (
                ", ".join(color_descriptions[:-1]) + f" and {color_descriptions[-1]}"
                if len(color_descriptions) > 1
                else color_descriptions[0]
            )

            if grid_name == "input grid":
                squares_description = ", ".join(
                    [f"a {self.color_map[color]} square" for color in self.colors]
                )
                return f"""Looking at the {grid_name}, we see a {self.grid_size}x{self.grid_size} square grid. 
The grid has a black background, on which {self.num_squares} colored squares are placed: {squares_description}.
These squares are of different sizes and are positioned randomly across the grid.
Some of these squares overlap with each other, creating areas where multiple colors seem to be competing for the same space.
In the overlapping areas, the color of the square that was placed last is visible, obscuring the colors underneath.
The colors present in this grid are {color_list}.
The colored cells occupy {np.sum(grid > 0)} cells ({(np.sum(grid > 0) / total_cells) * 100:.1f}% of the grid), 
while the remaining {np.sum(grid == 0)} cells ({(np.sum(grid == 0) / total_cells) * 100:.1f}% of the grid) are black background.
This arrangement creates a complex pattern of overlapping shapes, reminiscent of a simplified, colorful Venn diagram."""
            else:
                return f"""Looking at the {grid_name}, we see a {self.grid_size}x{self.grid_size} square grid. 
The grid has a black background, with {self.num_squares} colored squares and additional {self.color_map[self.intersection_color]} areas.
The colors present in this grid are {color_list}.
The colored cells occupy {np.sum(grid > 0)} cells ({(np.sum(grid > 0) / total_cells) * 100:.1f}% of the grid), 
while the remaining {np.sum(grid == 0)} cells ({(np.sum(grid == 0) / total_cells) * 100:.1f}% of the grid) are black background.
The {self.color_map[self.intersection_color]} areas appear in locations where squares overlap, creating a clear visual representation of intersections."""

        input_description = describe_grid(input_grid, "input grid")
        output_description = describe_grid(output_grid, "output grid")

        intersection_cells = np.sum(output_grid == self.intersection_color)
        total_cells = self.grid_size * self.grid_size
        intersection_percentage = (intersection_cells / total_cells) * 100

        difference = f"""
When comparing the input and output grids, we notice several key differences:

1. The most striking change is the appearance of {self.color_map[self.intersection_color]} cells in the output grid, which were not present in the input grid.
2. These new {self.color_map[self.intersection_color]} cells appear in areas where the colored squares overlap in the input grid.
3. In the input grid, overlapping areas were simply colored with the last placed square's color. Now, these areas are distinctly marked in {self.color_map[self.intersection_color]}.
4. The original colored squares remain in their same positions, but their overlapping areas are now filled with the new {self.color_map[self.intersection_color]} color.
5. Specifically, {intersection_cells} cells ({intersection_percentage:.1f}% of the grid) are now {self.color_map[self.intersection_color]}, representing intersections between squares.
6. The black background cells that are not part of any square or intersection remain unchanged.
7. The overall structure of the grid has changed from simple overlapping squares to a true Venn diagram-like representation, clearly showing all intersections.

This transformation has effectively created a visual representation of set intersections, similar to a Venn diagram, by highlighting the overlapping areas of the colored squares with a distinct color.
"""

        return f"{input_description}\n\n{output_description}\n\n{difference}"

    def generate_pattern_description(self) -> str:
        return f"""
After carefully observing the changes between the input and output grids, we can describe the pattern of transformation as follows:

1. Grid Structure:
   - We start with a {self.grid_size}x{self.grid_size} grid.
   - The input grid contains {self.num_squares} colored squares of various sizes, randomly positioned and potentially overlapping.

2. Intersection Identification:
   - The transformation identifies areas where two or more colored squares overlap.
   - In the input, these areas were simply colored with the last placed square's color.

3. New Color Introduction:
   - A new color, {self.color_map[self.intersection_color]}, is introduced specifically for intersections.
   - This color was not present in the input grid.

4. Intersection Coloring:
   - All identified intersection areas are filled with the new {self.color_map[self.intersection_color]} color.
   - This includes areas where two or more squares overlap, regardless of their original colors.
   - The intersection color is applied consistently, whether it's an overlap of two, three, or more squares.

5. Original Square Preservation:
   - The original colored squares remain in their initial positions.
   - Their non-overlapping areas retain their original colors.
   - This means that the outer edges of each square (where they don't overlap with others) keep their initial colors.

6. Background Preservation:
   - The black background cells that are not part of any square or intersection remain unchanged.
   - This maintains the overall layout and positioning of the squares within the grid.

7. Consistency:
   - This intersection coloring is applied consistently across the entire grid.
   - Every overlapping area, no matter how small or large, is treated the same way.

8. Resulting Pattern:
   - The output grid now resembles a true Venn diagram, clearly showing:
     a. The original squares in their respective colors, representing distinct sets.
     b. Intersections between squares in the new {self.color_map[self.intersection_color]} color, representing overlaps between sets.
     c. Non-overlapping areas of squares in their original colors, representing unique elements of each set.
   - This creates a clear visual distinction between unique areas (single colors) and shared areas (intersection color).

9. Visual Clarity:
   - The transformation enhances the visual representation of set relationships.
   - It becomes immediately clear which areas are shared between different colored squares (sets) and which are unique to each.

10. Potential for Analysis:
    - This transformed grid allows for easy identification of:
      - How many squares overlap in any given area (by looking at the presence of the intersection color).
      - Which specific squares are overlapping (by observing which original colors are adjacent to an intersection).
      - The extent of overlap between different squares (by the size of the intersection areas).

This transformation effectively converts a simple arrangement of overlapping colored squares into a more complex and informative representation that highlights all intersections. It visually demonstrates the concept of set intersections, similar to how Venn diagrams are used in mathematics and logic to represent relationships between different sets. The pattern reveals hidden information about overlaps that was not immediately apparent in the input grid.
"""

    def execute(self, grid: np.ndarray) -> (np.ndarray, str):
        new_grid = np.copy(grid)

        # Find all areas covered by any square
        covered = np.zeros_like(grid, dtype=bool)
        for color in self.colors:
            covered |= grid == color

        # Find intersections (areas covered by more than one square)
        intersection = np.zeros_like(grid, dtype=bool)
        for color in self.colors:
            temp = np.zeros_like(grid, dtype=bool)
            temp[grid == color] = True
            intersection |= temp & (covered ^ temp)

        # Color intersections
        new_grid[intersection] = self.intersection_color

        description = self.generate_description(grid, new_grid)
        pattern = self.generate_pattern_description()
        return new_grid, f"{description}\n\n{pattern}"


import os

# Create a directory to save images
if not os.path.exists("task_images"):
    os.makedirs("task_images")

# List of all tasks
tasks = [
    # ColorReplacementTask,
    # ShiftGridTask,
    # DrawSquaresTask,
    # ChangeStrokeColorTask,
    # ChangeFillColorTask,
    # CopyShapeWithPaddingTask,
    # PatternIntersectionUnionTask,
    # RotateShapeTask,
    # MoveShapeTask,
    # MirrorShapeTask,
    # CompleteDiagonalTask,
    # SortColumnsRowsByHeightTask,
    # GravityTask,
    # BuoyancyTask,
    # ScaleUpShapeTask,
    # AddGridOverlayTask,
    # RainwaterTask,
    # BorderAdditionTask,
    # ShapeMergingTask,
    # SimpleShapePatternFillingTask,
    AddNoiseAndLinesTask
]

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Function to plot four tasks
def plot_four_tasks(task_instances, heading="Task Comparisons"):
    fig, axes = plt.subplots(4, 2, figsize=(12, 28))
    fig.suptitle(heading, fontsize=20)

    cmap = mcolors.ListedColormap(
        ["black", "red", "green", "blue", "yellow", "purple", "orange"]
    )
    bounds = np.arange(8) - 0.5
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    for i, task in enumerate(task_instances):
        attempt = 0
        success = False
        while attempt < 10 and not success:
            try:
                input_grid = task.sample()
                if isinstance(input_grid, tuple):
                    output_grid, instruction = task.execute(*input_grid)
                else:
                    output_grid, instruction = task.execute(input_grid)
                success = True
            except Exception as e:
                attempt += 1
                print(f"Error in {task.__class__.__name__} - Attempt {attempt}: {e}")

        if not success:
            print(
                f"Failed to generate grids for {task.__class__.__name__} after 10 attempts."
            )
            continue

        def plot_grid(ax, grid, title):
            cax = ax.imshow(grid, cmap=cmap, norm=norm)
            ax.set_title(title)
            ax.set_xticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
            ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
            ax.tick_params(which="minor", size=0)
            fig.colorbar(cax, ax=ax, boundaries=bounds, ticks=np.arange(7))
            ax.set_xticks(np.arange(0, grid.shape[0], 1))
            ax.set_yticks(np.arange(0, grid.shape[0], 1))

        plot_grid(
            axes[i, 0],
            input_grid if not isinstance(input_grid, tuple) else input_grid[0],
            "Input Grid",
        )
        plot_grid(axes[i, 1], output_grid, "Output Grid")

        # Add instruction text below the output grid
        axes[i, 1].text(
            0.5,
            -0.15,
            instruction,
            ha="center",
            va="top",
            fontsize=12,
            transform=axes[i, 1].transAxes,
        )

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    return fig


# # Loop over all tasks and save the plots
# for task_cls in tasks:
#     task_name = task_cls.__name__.replace("Task", "").replace("_", " ")
#     task_instances = [task_cls() for _ in range(4)]
#     fig = plot_four_tasks(task_instances, heading=f"{task_name} Task Examples")
#     fig.savefig(f"task_images/{task_name}_examples.png")
#     plt.close(fig)

import numpy as np
import json
import os
import threading
from queue import Queue


# Task classes definition goes here (use the previously provided code)

# Constants
NUM_EXAMPLES = 100
NUM_TASKS = len(tasks)
EXAMPLES_PER_TASK = NUM_EXAMPLES // NUM_TASKS
NUM_THREADS = 8

# Create a directory to save the JSONL file
output_dir = "task_examples"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# File path for the output JSONL file
output_file = os.path.join(output_dir, "task_example_eval_red_linea.jsonl")

# Thread lock for file writing
file_lock = threading.Lock()
# set numpy seed
np.random.seed(12321)


# Worker function for threading
def worker(task_cls, num_examples, thread_id):
    local_examples = []
    for _ in range(num_examples):
        task = task_cls()
        attempt = 0
        success = False
        while attempt < 10 and not success:
            try:
                input_grid = task.sample()
                if isinstance(input_grid, tuple):
                    output_grid, instruction = task.execute(*input_grid)
                else:
                    output_grid, instruction = task.execute(input_grid)
                success = True
            except Exception as e:
                attempt += 1
                print(f"Error in {task.__class__.__name__} - Attempt {attempt}: {e}")

        if success:
            example = {
                "input": (
                    input_grid.tolist()
                    if not isinstance(input_grid, tuple)
                    else input_grid[0].tolist()
                ),
                "output": output_grid.tolist(),
                "instruction": instruction,
            }

            example["input"] = str(
                ["".join([str(i) for i in row]) for row in example["input"]]
            )
            example["output"] = str(
                ["".join([str(i) for i in row]) for row in example["output"]]
            )
            example["task"] = task.__class__.__name__
            local_examples.append(example)

    with file_lock:
        with open(output_file, "a") as f:
            for example in local_examples:
                f.write(json.dumps(example) + "\n")


# Function to start and manage threads for a specific task
def start_threads_for_task(task_cls, num_examples_per_thread):
    threads = []
    for thread_id in range(NUM_THREADS):
        thread = threading.Thread(
            target=worker, args=(task_cls, num_examples_per_thread, thread_id)
        )
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()


# Clear the output file if it exists
if os.path.exists(output_file):
    os.remove(output_file)

# Loop over all tasks and start threads for each task
for task_cls in tasks:
    start_threads_for_task(task_cls, EXAMPLES_PER_TASK // NUM_THREADS)

# shuffle the lines in the file
lines = open(output_file).readlines()
np.random.shuffle(lines)
open(output_file, "w").writelines(lines)

print(f"Generated examples and saved to {output_file}")

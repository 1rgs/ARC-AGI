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


class GridTask(ABC):
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
        pass

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def execute(self, grid: np.ndarray) -> (np.ndarray, str):
        # returns a new grid and a string instruction
        pass

    def visualize(
        self, input_grid: np.ndarray, output_grid: np.ndarray, instruction: str
    ):
        grid_size = input_grid.shape[0]
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Create a custom colormap
        cmap = mcolors.ListedColormap(
            ["black"] + [self.color_map[i] for i in range(1, 7)]
        )
        bounds = np.arange(8) - 0.5
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        def plot_grid(ax, grid, title):
            cax = ax.imshow(grid, cmap=cmap, norm=norm)
            ax.set_title(title)
            ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
            ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
            ax.tick_params(which="minor", size=0)
            fig.colorbar(cax, ax=ax, boundaries=bounds, ticks=np.arange(7))
            ax.set_xticks(np.arange(0, grid_size, 1))
            ax.set_yticks(np.arange(0, grid_size, 1))

        plot_grid(axes[0], input_grid, "Input Grid")
        plot_grid(axes[1], output_grid, "Output Grid")

        plt.suptitle(instruction)
        plt.show()


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
        color1, color2 = np.random.choice(self.colors, size=2, replace=False)
        new_grid = np.copy(grid)
        new_grid[grid == color1] = color2
        instruction = f"Replace {color1} with {color2}"
        return new_grid, instruction


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
        n = np.random.randint(1, self.grid_size // 2)
        direction = np.random.choice(self.directions)
        new_grid = np.full(grid.shape, self.fill_color)

        if direction == "right":
            new_grid[:, n:] = grid[:, :-n]
        elif direction == "left":
            new_grid[:, :-n] = grid[:, n:]
        elif direction == "down":
            new_grid[n:, :] = grid[:-n, :]
        elif direction == "up":
            new_grid[:-n, :] = grid[n:, :]

        instruction = f"Shift everything {direction} by {n} pixels and fill with color {self.fill_color}"
        return new_grid, instruction


class DrawSquaresTask(GridTask):
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
            size = np.random.randint(1, max(2, self.grid_size // 2))
            color = np.random.choice(self.colors)
            start_x = np.random.randint(0, self.grid_size - size)
            start_y = np.random.randint(0, self.grid_size - size)
            grid[start_x : start_x + size, start_y : start_y + size] = color
        return grid

    def execute(self, grid: np.ndarray) -> (np.ndarray, str):
        color1, color2 = np.random.choice(self.colors, size=2, replace=False)
        new_grid = np.copy(grid)
        new_grid[grid == color1] = color2
        instruction = f"Replace {self.color_map[color1]} with {self.color_map[color2]}"
        return new_grid, instruction


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
        stroke_color = np.random.choice(self.colors)
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
                            new_grid[i, j] = stroke_color
                            break

        instruction = f"Change stroke color to {self.color_map[stroke_color]}"
        return new_grid, instruction


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
        fill_color = np.random.choice(self.colors)
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
                    new_grid[i, j] = fill_color
        instruction = f"Change fill color to {self.color_map[fill_color]}, leaving the stroke color unchanged"
        return new_grid, instruction


class CopyShapeWithPaddingTask(GridTask):
    def __init__(self, single_color=True):
        super().__init__()
        self.grid_size = np.random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE + 1)
        self.shape_size = np.random.randint(2, max(3, self.grid_size // 2 + 1))

        self.single_color = single_color
        self.color = np.random.randint(1, 7) if single_color else None

    def sample(self):
        grid = generate_shape(self.grid_size, self.shape_size)
        if self.single_color:
            grid[grid == 1] = self.color
        else:
            colors = np.random.choice(
                range(1, 7), size=np.count_nonzero(grid), replace=True
            )
            grid[grid == 1] = colors
        return grid

    def execute(self, grid: np.ndarray) -> (np.ndarray, str):
        shape_coords = np.argwhere(grid > 0)
        if shape_coords.size == 0:
            return grid, "No shape found to copy"

        min_x, min_y = shape_coords.min(axis=0)
        max_x, max_y = shape_coords.max(axis=0)
        shape_component = grid[min_x : max_x + 1, min_y : max_y + 1]
        shape_height, shape_width = shape_component.shape

        new_grid = np.copy(grid)

        # Define all possible positions including diagonals
        all_positions = {
            "Above": (min_x - shape_height - 1, min_y),
            "Right": (min_x, min_y + shape_width + 1),
            "Below": (max_x + 2, min_y),
            "Left": (min_x, min_y - shape_width - 1),
            "Top-left": (min_x - shape_height - 1, min_y - shape_width - 1),
            "Top-right": (min_x - shape_height - 1, min_y + shape_width + 1),
            "Bottom-left": (max_x + 2, min_y - shape_width - 1),
            "Bottom-right": (max_x + 2, min_y + shape_width + 1),
        }

        # Determine the selection strategy
        strategy = np.random.choice(
            ["all", "top_left_bottom_right", "diagonal", "random"],
            p=[0.25, 0.25, 0.25, 0.25],
        )

        if strategy == "all":
            positions = all_positions
            instruction = "Copy the original shape, and then copy it above, right, below, and left with padding 1px"
        elif strategy == "top_left_bottom_right":
            positions = {
                key: all_positions[key] for key in ["Above", "Right", "Below", "Left"]
            }
            instruction = np.random.choice(
                [
                    "Copy the original shape, and then copy it above, right, below, and left with padding 1px",
                    "Copy the original shape to the top, left, bottom, and right with padding 1px",
                ]
            )
        elif strategy == "diagonal":
            positions = {
                key: all_positions[key]
                for key in ["Top-left", "Top-right", "Bottom-left", "Bottom-right"]
            }
            instruction = "Copy the original shape, and then copy it to the top-left, top-right, bottom-left, and bottom-right with padding 1px"
        else:
            num_directions = np.random.randint(1, len(all_positions) + 1)
            selected_keys = np.random.choice(
                list(all_positions.keys()), num_directions, replace=False
            )
            positions = {key: all_positions[key] for key in selected_keys}
            instruction = f"Copy the original shape to the following positions with padding 1px: {', '.join(positions.keys())}"

        for direction, pos in positions.items():
            pos_x, pos_y = pos
            if pos_x < 0 or pos_y < 0:
                # Compute the intersection of the shape with the grid
                intersect_x_start = max(0, pos_x)
                intersect_y_start = max(0, pos_y)
                intersect_x_end = min(self.grid_size, pos_x + shape_height)
                intersect_y_end = min(self.grid_size, pos_y + shape_width)

                if (
                    intersect_x_start < intersect_x_end
                    and intersect_y_start < intersect_y_end
                ):
                    new_grid[
                        intersect_x_start:intersect_x_end,
                        intersect_y_start:intersect_y_end,
                    ] = shape_component[
                        intersect_x_start - pos_x : intersect_x_end - pos_x,
                        intersect_y_start - pos_y : intersect_y_end - pos_y,
                    ]
            else:
                end_x = min(pos_x + shape_height, self.grid_size)
                end_y = min(pos_y + shape_width, self.grid_size)
                new_grid[pos_x:end_x, pos_y:end_y] = shape_component[
                    : end_x - pos_x, : end_y - pos_y
                ]

        return new_grid, instruction


class PatternIntersectionUnionTask(GridTask):
    def __init__(self, n=None):
        self.mode = np.random.choice(["intersection", "union", "combined"])
        self.orientation = np.random.choice(["vertical", "horizontal"])
        self.max_colors = np.random.randint(2, 7)
        self.n = n if n else np.random.randint(2, 11)

        if self.orientation == "vertical":
            self.grid_size = (self.n, 2 * self.n + 1)
        else:  # horizontal
            self.grid_size = (2 * self.n + 1, self.n)

    def sample(self):
        grid = np.zeros(self.grid_size, dtype=int)

        if self.orientation == "vertical":
            left_side = np.random.randint(0, self.max_colors, (self.n, self.n))
            right_side = np.random.randint(0, self.max_colors, (self.n, self.n))

            grid[:, : self.n] = left_side
            grid[:, self.n + 1 :] = right_side

            random_color = np.random.randint(1, 7)
            grid[:, self.n] = random_color
        else:  # horizontal
            top_side = np.random.randint(0, self.max_colors, (self.n, self.n))
            bottom_side = np.random.randint(0, self.max_colors, (self.n, self.n))

            grid[: self.n, :] = top_side
            grid[self.n + 1 :, :] = bottom_side

            random_color = np.random.randint(1, 7)
            grid[self.n, :] = random_color

        return grid

    def execute(self, grid: np.ndarray) -> (np.ndarray, str):
        if self.orientation == "vertical":
            left_side = grid[:, : self.n]
            right_side = grid[:, self.n + 1 :]

            # copy the divider color
            divider_color = grid[:, self.n]

            intersection = np.logical_and(left_side, right_side).astype(int)
            union = np.logical_or(left_side, right_side).astype(int)

            if self.mode == "intersection":
                result_grid = intersection
                instruction = "Perform an intersection of the left and right sides and return an n x n grid."
            elif self.mode == "union":
                result_grid = union
                instruction = "Perform a union of the left and right sides and return an n x n grid."
            elif self.mode == "combined":
                result_grid = np.zeros(self.grid_size, dtype=int)
                result_grid[:, : self.n] = union
                result_grid[:, self.n + 1 :] = intersection
                # add divider color
                result_grid[:, self.n] = divider_color

                instruction = "The left side is the union of the left and right sides. There is a vertical divider in the middle. The right side is the intersection of the left and right sides."
            else:
                raise ValueError(
                    "Invalid mode. Choose from 'intersection', 'union', or 'combined'."
                )
        else:  # horizontal
            top_side = grid[: self.n, :]
            bottom_side = grid[self.n + 1 :, :]

            # copy the divider color
            divider_color = grid[self.n, :]
            intersection = np.logical_and(top_side, bottom_side).astype(int)
            union = np.logical_or(top_side, bottom_side).astype(int)

            if self.mode == "intersection":
                result_grid = intersection
                instruction = "Perform an intersection of the top and bottom sides and return an n x n grid."
            elif self.mode == "union":
                result_grid = union
                instruction = "Perform a union of the top and bottom sides and return an n x n grid."
            elif self.mode == "combined":

                result_grid = np.zeros(self.grid_size, dtype=int)
                result_grid[: self.n, :] = union
                result_grid[self.n + 1 :, :] = intersection
                # add divider color
                result_grid[self.n, :] = divider_color
                instruction = "The top side is the union of the top and bottom sides. There is a horizontal divider in the middle. The bottom side is the intersection of the top and bottom sides."
            else:
                raise ValueError(
                    "Invalid mode. Choose from 'intersection', 'union', or 'combined'."
                )

        return result_grid, instruction


class RotateShapeTask(GridTask):
    def __init__(self):
        super().__init__()
        self.grid_size = np.random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE + 1)
        self.num_shapes = np.random.randint(1, 9)
        self.colors = np.random.choice(range(1, 7), self.num_shapes, replace=True)
        self.mode = np.random.choice(["clockwise", "counter-clockwise", "180"])
        self.shape_size = np.random.randint(2, max(3, self.grid_size // 2 + 1))

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
            instruction = "Rotate the entire grid 90 degrees clockwise."

        elif self.mode == "counter-clockwise":
            new_grid = np.rot90(grid, 1)  # Rotate 90 degrees counter-clockwise
            instruction = "Rotate the entire grid 90 degrees counter-clockwise."

        elif self.mode == "180":
            new_grid = np.rot90(grid, 2)  # Rotate 180 degrees
            instruction = "Rotate the entire grid 180 degrees."

        return new_grid, instruction


class MoveShapeTask(GridTask):
    def __init__(self):
        super().__init__()
        self.grid_size = np.random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE + 1)
        self.num_shapes = np.random.randint(1, 9)
        self.colors = np.random.choice(range(1, 7), self.num_shapes, replace=True)
        self.shape_size = np.random.randint(2, 5)
        self.directions = [
            "top",
            "bottom",
            "left",
            "right",
            "top-right",
            "top-left",
            "bottom-right",
            "bottom-left",
        ]
        self.modes = ["move-only", "move-and-copy"]

    def sample(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for color in self.colors:
            shape = generate_shape(self.grid_size, self.shape_size)
            shape[shape == 1] = color
            grid = np.maximum(grid, shape)
        return grid

    def execute(self, grid: np.ndarray) -> (np.ndarray, str):
        color = np.random.choice(self.colors)
        n = np.random.randint(1, self.grid_size // 2)
        direction = np.random.choice(self.directions)
        mode = np.random.choice(self.modes)
        new_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        if mode == "move-and-copy":
            for other_color in self.colors:
                if other_color != color:
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

        move_shape(new_grid, grid, color, n, direction)

        instruction = f"Move all shapes of color {self.color_map[color]} {n} pixels to the {direction}."
        if mode == "move-and-copy":
            instruction += " Copy other shapes to their original positions."
        else:
            instruction += " Remove other shapes."

        return new_grid, instruction


class MirrorShapeTask(GridTask):
    def __init__(self):
        super().__init__()
        self.grid_size = np.random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE + 1)
        self.num_shapes = np.random.randint(1, 9)
        self.colors = np.random.choice(range(1, 7), self.num_shapes, replace=True)
        self.axis = np.random.choice(["horizontal", "vertical"])
        self.shape_size = np.random.randint(2, max(3, self.grid_size // 2 + 1))

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
            instruction = "Mirror the entire grid horizontally."
        else:
            new_grid = np.flipud(grid)
            instruction = "Mirror the entire grid vertically."

        return new_grid, instruction


class CompleteDiagonalTask(GridTask):
    def __init__(self):
        super().__init__()
        self.grid_size = np.random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE + 1)
        self.color = np.random.choice(list(range(1, 7)))
        self.diagonal_color = np.random.choice(list(range(1, 7)))
        while self.color == self.diagonal_color:
            self.diagonal_color = np.random.choice(list(range(1, 7)))

    def sample(self):
        grid = np.full((self.grid_size, self.grid_size), self.color)
        num_diagonals = np.random.randint(1, 3)
        for _ in range(num_diagonals):
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

        instruction = "Extend all diagonals through the grid."
        return new_grid, instruction


class SortColumnsRowsByHeightTask(GridTask):
    def __init__(self):
        super().__init__()
        self.grid_size = np.random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE + 1)
        self.orientation = np.random.choice(["rows", "columns"])
        self.sort_order = np.random.choice(["ascending", "descending"])
        self.colors = np.random.choice(
            list(range(1, 7)), size=self.grid_size, replace=True
        )

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
            instruction = (
                f"Sort rows by height in {self.sort_order} order and left-align."
            )

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
            instruction = (
                f"Sort columns by height in {self.sort_order} order and top-align."
            )

        return new_grid, instruction


class GravityTask(GridTask):
    def __init__(self):
        super().__init__()
        self.grid_size = np.random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE + 1)
        self.colors = np.random.choice(
            list(range(1, 7)), size=np.random.randint(2, 7), replace=False
        )

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
        instruction = "Make all colors fall to the bottom due to gravity."
        return new_grid, instruction


class BuoyancyTask(GridTask):
    def __init__(self):
        super().__init__()
        self.grid_size = np.random.randint(MIN_GRID_SIZE + 2, MAX_GRID_SIZE + 1)
        self.water_level = self.grid_size // 2
        self.colors = np.random.choice(
            range(1, 7), size=np.random.randint(2, 7), replace=False
        )
        self.mode = np.random.choice(["sink", "float_above", "float_below"])

    def sample(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        water_color = np.random.choice(self.colors)
        grid[self.water_level + 1 :, :] = water_color

        start_x = 0
        num_objects_placed = 0

        while (
            start_x < self.grid_size and num_objects_placed < 10
        ):  # Limit the number of objects
            size = np.random.randint(1, max(2, self.grid_size // 2))
            color = np.random.choice(
                [color for color in self.colors if color != water_color]
            )

            if start_x + size > self.grid_size:
                break  # No more space on the x-axis

            start_y = 0  # Reset start_y for each new column of objects

            while start_y + size <= self.water_level:
                if start_y + size <= self.water_level:
                    grid[start_y : start_y + size, start_x : start_x + size] = color
                    start_y += size  # Move start_y down by the size of the object
                    num_objects_placed += 1  # Increment the number of objects placed
                else:
                    break  # No more vertical space

            start_x += size  # Move start_x to the next position

        return grid

    def execute(self, grid: np.ndarray) -> (np.ndarray, str):
        new_grid = np.copy(grid)

        # Identify all objects and their coordinates
        objects = {}
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if grid[i, j] > 0 and i <= self.water_level:
                    color = grid[i, j]
                    if color not in objects:
                        objects[color] = []
                    objects[color].append((i, j))

        # Move each object according to the selected mode
        for color, coords in objects.items():
            if self.mode == "sink":
                max_y = max(coord[0] for coord in coords)
                offset = self.grid_size - 1 - max_y
                instruction = "Sink all objects to the bottom."

            elif self.mode == "float_above":
                # Calculate the highest point the object can fall to without crossing the water line
                min_y = min(coord[0] for coord in coords)
                max_shift = self.water_level - min_y
                instruction = "Float all objects such that their top is just above the water level."

                for i, j in coords:
                    new_grid[i, j] = 0  # Clear original position
                    new_grid[i + max_shift, j] = color  # Move to new position

            elif self.mode == "float_below":
                min_y = min(coord[0] for coord in coords)
                offset = self.water_level + 1 - min_y
                instruction = "Float all objects completely below the water level with the top of the object at the water line."

            for i, j in coords:
                if self.mode != "float_above":
                    new_grid[i, j] = 0  # Clear original position
                    if i + offset < self.grid_size:
                        new_grid[i + offset, j] = color  # Move to new position

        return new_grid, instruction


class ScaleUpShapeTask(GridTask):
    def __init__(self):
        super().__init__()
        self.grid_size = np.random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE + 1)
        self.shape_size = np.random.randint(2, max(3, self.grid_size // 4 + 1))
        self.color = np.random.choice(range(1, 7))
        self.corners = ["top-left", "top-right", "bottom-left", "bottom-right"]
        self.corner = np.random.choice(self.corners)

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

        instruction = (
            f"Scale up the shape by a factor of 2 from the {self.corner} corner."
        )
        return new_grid, instruction


class AddGridOverlayTask(GridTask):
    def __init__(self):
        super().__init__()
        self.grid_size = np.random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE + 1)
        self.noise_level = 0.05  # Percentage of noise cells
        self.cell_size = np.random.randint(
            2, 5
        )  # Size of each cell in the overlay grid
        self.grid_color = np.random.choice(range(1, 7))

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

        instruction = f"Add a grid overlay on top of the original grid with cell size {self.cell_size} and grid color {self.color_map[self.grid_color]}. The original should be preserved."

        return new_grid, instruction


class RainwaterTask(GridTask):
    def __init__(self):
        super().__init__()
        self.grid_size = np.random.randint(
            max(3, MIN_GRID_SIZE), MAX_GRID_SIZE + 1
        )  # Ensure grid_size is at least 3
        self.max_column_height = self.grid_size // 2
        self.water_color = 3  # Set the rainwater color to blue
        self.column_color = np.random.choice(range(1, 7))
        while self.column_color == self.water_color:
            self.column_color = np.random.choice(range(1, 7))

    def sample(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        num_columns = np.random.randint(3, self.grid_size) if self.grid_size > 3 else 3
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

        instruction = f"Fill the gaps between columns with rainwater color {self.color_map[self.water_color]}."
        return new_grid, instruction


class BorderAdditionTask(GridTask):
    def __init__(self):
        super().__init__()
        self.grid_size = np.random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE + 1)
        self.num_shapes = np.random.randint(1, 9)
        self.colors = np.random.choice(range(1, 7), self.num_shapes, replace=True)
        self.border_color = np.random.choice(range(1, 7))

    def sample(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for color in self.colors:
            shape = generate_shape(self.grid_size, self.grid_size // 4)
            shape[shape == 1] = color
            grid = np.maximum(grid, shape)
        return grid

    def execute(self, grid: np.ndarray) -> (np.ndarray, str):
        new_grid = np.copy(grid)
        border_color = self.border_color

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
                            new_grid[ni, nj] = border_color

        instruction = (
            f"Add a border of {self.color_map[border_color]} around all shapes."
        )
        return new_grid, instruction


class ShapeMergingTask(GridTask):
    def __init__(self):
        super().__init__()
        self.grid_size = np.random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE + 1)
        self.num_shapes = np.random.randint(2, 10)
        self.colors = np.random.choice(range(1, 7), self.num_shapes, replace=True)
        self.merged_color = np.random.choice(range(1, 7))

    def sample(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for color in self.colors:
            shape = generate_shape(self.grid_size, self.grid_size // 4)
            shape[shape == 1] = color
            grid = np.maximum(grid, shape)
        return grid

    def execute(self, grid: np.ndarray) -> (np.ndarray, str):
        new_grid = np.copy(grid)
        merged_color = self.merged_color

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
                        new_grid[x, y] = merged_color

        instruction = f"Merge all overlapping shapes into a single shape of {self.color_map[merged_color]}."
        return new_grid, instruction


class SimpleShapePatternFillingTask(GridTask):
    def __init__(self):
        super().__init__()
        self.n = np.random.randint(2, 6)  # Size of the small pattern
        self.tile_size = np.random.randint(3, 6)  # How many times to tile the pattern
        self.grid_size = self.n * self.tile_size  # Size of the larger grid
        self.pattern_color = np.random.choice(range(1, 7))
        self.background_color = 0
        self.pattern = self.generate_pattern()

    def generate_pattern(self):
        # random chance 50%
        if np.random.rand() < 0.5:
            # Generate a simple shape pattern (e.g., a cross)
            pattern = np.zeros((self.n, self.n), dtype=int)
            center = self.n // 2
            pattern[center, :] = self.pattern_color
            pattern[:, center] = self.pattern_color
        # else, generate a x pattern
        else:
            pattern = np.zeros((self.n, self.n), dtype=int)
            for i in range(self.n):
                pattern[i, i] = self.pattern_color
                pattern[i, self.n - i - 1] = self.pattern_color

        return pattern

    def sample(self):
        # Tile the pattern to cover the larger grid
        grid = np.tile(self.pattern, (self.tile_size, self.tile_size))
        self.final_grid = np.copy(grid)
        # Delete a random chunk
        chunk_size = np.random.randint(3, min(self.grid_size // 2, 6) + 1)
        start_x = np.random.randint(0, self.grid_size - chunk_size + 1)
        start_y = np.random.randint(0, self.grid_size - chunk_size + 1)
        grid[start_x : start_x + chunk_size, start_y : start_y + chunk_size] = (
            self.background_color
        )

        return grid

    def execute(self, grid: np.ndarray) -> (np.ndarray, str):
        return self.final_grid, "Fill the deleted chunk with the pattern."


import os

# Create a directory to save images
if not os.path.exists("task_images"):
    os.makedirs("task_images")

# List of all tasks
tasks = [
    ColorReplacementTask,
    ShiftGridTask,
    DrawSquaresTask,
    ChangeStrokeColorTask,
    ChangeFillColorTask,
    CopyShapeWithPaddingTask,
    PatternIntersectionUnionTask,
    RotateShapeTask,
    MoveShapeTask,
    MirrorShapeTask,
    CompleteDiagonalTask,
    SortColumnsRowsByHeightTask,
    GravityTask,
    BuoyancyTask,
    ScaleUpShapeTask,
    AddGridOverlayTask,
    RainwaterTask,
    BorderAdditionTask,
    ShapeMergingTask,
    SimpleShapePatternFillingTask,
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

# Seed for reproducibility
seed = 1
np.random.seed(seed)

# Task classes definition goes here (use the previously provided code)

# Constants
NUM_EXAMPLES = 100000
NUM_TASKS = len(tasks)
EXAMPLES_PER_TASK = NUM_EXAMPLES // NUM_TASKS
NUM_THREADS = 8

# Create a directory to save the JSONL file
output_dir = "task_examples"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# File path for the output JSONL file
output_file = os.path.join(output_dir, "task_examples.jsonl")

# Thread lock for file writing
file_lock = threading.Lock()


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

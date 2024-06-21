import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Tuple
import random

class SimpleGridDescriptionTask(ABC):
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.color_map = {
            0: "black",
            1: "red",
            2: "green",
            3: "blue",
            4: "yellow",
            5: "purple",
            6: "orange",
        }

    @abstractmethod
    def execute(self) -> tuple[np.ndarray, str]:
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
        cax = ax.imshow(indexed_grid, cmap=cmap, interpolation='nearest')
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
        ax.format_cursor_data = lambda data: ''

        plt.tight_layout()
        plt.show()

class RectangleSuperImpose(SimpleGridDescriptionTask):
    def __init__(self, grid_size, num_colors):
        super().__init__(grid_size)
        self.num_colors = min(num_colors, 6)  # Limit to 6 colors due to color_map

    def execute(self) -> tuple[np.ndarray, str]:
        while True:
            grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
            colors = np.random.choice(range(1, 7), size=3, replace=False)
            rect_coords = np.random.randint(0, self.grid_size, size=(3, 2, 2))

            # Sort rectangles based on their top-left x-coordinate
            sort_indices = np.argsort(rect_coords[:, 0, 0])
            rect_coords = rect_coords[sort_indices]
            colors = colors[sort_indices]

            original_areas = [0, 0, 0]

            for i in range(3):
                x1, y1 = rect_coords[i, 0]
                x2, y2 = rect_coords[i, 1]
                rect = grid[min(x1, x2):max(x1, x2)+1, min(y1, y2):max(y1, y2)+1]
                
                original_areas[i] = rect.size  # Store the original area of each rectangle
                rect[:] = colors[i]

            color_counts = np.array([np.count_nonzero(grid == color) for color in colors])  
            print(color_counts)

            # Check if all rectangles are partially covered
            if all((0 < color_counts[i] < area) or (i == 2) for i, (_, area) in enumerate(zip(colors, original_areas))): 
                break  # Exit the loop if the condition is met

        return grid, "A grid with three partially overlapping rectangles, {2} on top of {1} on top of {0}".format(colors[0], colors[1], colors[2])

class RectangleTreeSuperImpose(SimpleGridDescriptionTask):
    def __init__(self, grid_size, num_colors):
        super().__init__(grid_size)
        self.num_colors = min(num_colors, 6)  # Limit to 6 colors due to color_map

    def execute(self) -> tuple[np.ndarray, str]:
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        colors = np.random.choice(range(1, 7), size=3, replace=False)
        
        # Generate the largest rectangle
        x1, y1 = np.random.randint(0, self.grid_size // 2, size=2)
        x2 = np.random.randint(x1 + 2, self.grid_size)
        y2 = np.random.randint(y1 + 2, self.grid_size)
        
        grid[x1:x2+1, y1:y2+1] = colors[0]
        
        # Generate the middle rectangle
        x1_mid = np.random.randint(x1 + 1, x2 - 1)
        y1_mid = np.random.randint(y1 + 1, y2 - 1)
        x2_mid = np.random.randint(x1_mid + 1, x2)
        y2_mid = np.random.randint(y1_mid + 1, y2)
        
        grid[x1_mid:x2_mid+1, y1_mid:y2_mid+1] = colors[1]
        
        # Generate the smallest rectangle
        x1_small = np.random.randint(x1_mid + 1, x2_mid - 1)
        y1_small = np.random.randint(y1_mid + 1, y2_mid - 1)
        x2_small = np.random.randint(x1_small + 1, x2_mid)
        y2_small = np.random.randint(y1_small + 1, y2_mid)
        
        grid[x1_small:x2_small+1, y1_small:y2_small+1] = colors[2]

        description = (
            f"A grid with three nested rectangles: "
            f"a {self.color_map[colors[2]]} rectangle inside "
            f"a {self.color_map[colors[1]]} rectangle, both inside "
            f"a {self.color_map[colors[0]]} rectangle."
        )

        return grid, description

class NestedRectangleAndIndependent(SimpleGridDescriptionTask):
    def __init__(self, grid_size, num_colors):
        super().__init__(grid_size)
        self.num_colors = min(num_colors, 6)  # Limit to 6 colors due to color_map

    def create_rectangle(self, grid: np.ndarray, color: int, exclude_area: Tuple[int, int, int, int] = None) -> Tuple[int, int, int, int]:
        while True:
            x1, y1 = np.random.randint(0, self.grid_size - 2, size=2)
            x2 = np.random.randint(x1 + 1, min(x1 + 10, self.grid_size - 1))
            y2 = np.random.randint(y1 + 1, min(y1 + 10, self.grid_size - 1))
            
            if exclude_area:
                ex1, ey1, ex2, ey2 = exclude_area
                if (x1 <= ex2 and x2 >= ex1 and y1 <= ey2 and y2 >= ey1):
                    continue  # Overlap with excluded area, try again
            
            grid[x1:x2+1, y1:y2+1] = color
            return x1, y1, x2, y2

    def execute(self) -> tuple[np.ndarray, str]:
        while True:
            grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
            colors = np.random.choice(range(1, 7), size=3, replace=False)
            
            # Create outer nested rectangle
            outer_x1, outer_y1, outer_x2, outer_y2 = self.create_rectangle(grid, colors[0])
            
            # Create inner nested rectangle
            try: 
                inner_x1 = np.random.randint(outer_x1 + 1, outer_x2 - 1)
                inner_y1 = np.random.randint(outer_y1 + 1, outer_y2 - 1)
                inner_x2 = np.random.randint(inner_x1 + 1, outer_x2)
                inner_y2 = np.random.randint(inner_y1 + 1, outer_y2)
            except ValueError:
                continue

            grid[inner_x1:inner_x2+1, inner_y1:inner_y2+1] = colors[1]
            
            # Create independent rectangle
            self.create_rectangle(grid, colors[2], exclude_area=(outer_x1, outer_y1, outer_x2, outer_y2))

            description = (
                f"A grid with three rectangles: "
                f"a {self.color_map[colors[1]]} rectangle nested inside "
                f"a {self.color_map[colors[0]]} rectangle, and a separate "
                f"{self.color_map[colors[2]]} rectangle."
            )

            return grid, description


class ColoredInShapes(SimpleGridDescriptionTask):
    def __init__(self, grid_size):
        super().__init__(grid_size)
        self.num_shapes = 3

    def execute(self) -> tuple[np.ndarray, str]:
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        # Generate random shapes
        shape_color, fill_color = random.sample(range(1, 7), 2)

        for _ in range(self.num_shapes):
            shape_type = random.choice(['rectangle', 'cross'])
            
            if shape_type == 'rectangle':
                x1, y1 = np.random.randint(0, self.grid_size, size=2)
                x2 = np.random.randint(x1, self.grid_size)
                y2 = np.random.randint(y1, self.grid_size)
                grid[x1:x2+1, y1:y2+1] = fill_color
            elif shape_type == 'cross':
                center_x, center_y = np.random.randint(1, self.grid_size-1, size=2)
                size = random.randint(1, min(center_x, center_y, self.grid_size-center_x-1, self.grid_size-center_y-1))
                grid[center_x-size:center_x+size+1, center_y] = fill_color
                grid[center_x, center_y-size:center_y+size+1] = fill_color

        # Outline the shapes with the shape_color
        padded = np.pad(grid, pad_width=1, mode='constant', constant_values=0)
        outline = np.zeros_like(grid)
        outline[((padded[:-2, 1:-1] != padded[1:-1, 1:-1]) | 
                 (padded[2:, 1:-1] != padded[1:-1, 1:-1]) | 
                 (padded[1:-1, :-2] != padded[1:-1, 1:-1]) | 
                 (padded[1:-1, 2:] != padded[1:-1, 1:-1])) & 
                (grid != 0)] = shape_color

        # Combine the filled shapes and their outlines
        grid = np.where(outline != 0, outline, grid)

        description = f"A grid with several shapes colored in {self.color_map[fill_color]} and outlined in {self.color_map[shape_color]}"
        return grid, description

class RectangleWithHole(SimpleGridDescriptionTask):
    def __init__(self, grid_size, num_colors):
        super().__init__(grid_size)

    def execute(self) -> tuple[np.ndarray, str]:
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Generate random rectangle dimensions and position
        width = random.randint(4, self.grid_size - 1)  # Minimum width of 4 to ensure non-corner holes
        height = random.randint(4, self.grid_size - 1)  # Minimum height of 4 to ensure non-corner holes
        x = random.randint(0, self.grid_size - width)
        y = random.randint(0, self.grid_size - height)
        
        # Choose a random color for the rectangle
        color = random.randint(1, 6)
        
        # Create the rectangle boundary
        grid[y, x:x+width] = color  # Top edge
        grid[y+height-1, x:x+width] = color  # Bottom edge
        grid[y:y+height, x] = color  # Left edge
        grid[y:y+height, x+width-1] = color  # Right edge
        
        # Remove a single non-corner element from the boundary
        side = random.choice(['top', 'bottom', 'left', 'right'])
        if side == 'top':
            hole_x = random.randint(x + 1, x + width - 2)
            hole_y = y
        elif side == 'bottom':
            hole_x = random.randint(x + 1, x + width - 2)
            hole_y = y + height - 1
        elif side == 'left':
            hole_x = x
            hole_y = random.randint(y + 1, y + height - 2)
        else:  # right
            hole_x = x + width - 1
            hole_y = random.randint(y + 1, y + height - 2)
        
        grid[hole_y, hole_x] = 0
        
        # Generate description
        description = f"A {self.color_map[color]} rectangle outline with a single hole on its {side} edge, located at ({x}, {y}) with width {width} and height {height}. The hole is at ({hole_x}, {hole_y})."
        
        return grid, description
    
class RectangleWithHoles(SimpleGridDescriptionTask):
    def __init__(self, grid_size, num_colors):
        super().__init__(grid_size)

    def execute(self) -> tuple[np.ndarray, str]:
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Generate random rectangle dimensions and position
        width = random.randint(6, self.grid_size - 1)  # Minimum width of 6 to ensure space for multiple holes
        height = random.randint(6, self.grid_size - 1)  # Minimum height of 6 to ensure space for multiple holes
        x = random.randint(0, self.grid_size - width)
        y = random.randint(0, self.grid_size - height)
        
        # Choose a random color for the rectangle
        color = random.randint(1, 6)
        
        # Create the rectangle boundary
        grid[y, x:x+width] = color  # Top edge
        grid[y+height-1, x:x+width] = color  # Bottom edge
        grid[y:y+height, x] = color  # Left edge
        grid[y:y+height, x+width-1] = color  # Right edge
        
        # Choose a random number of holes between 2 and 4
        num_holes = random.randint(2, 4)
        
        holes = []
        sides = ['top', 'bottom', 'left', 'right']
        
        for _ in range(num_holes):
            while True:
                side = random.choice(sides)
                if side == 'top':
                    hole_x = random.randint(x + 1, x + width - 2)
                    hole_y = y
                elif side == 'bottom':
                    hole_x = random.randint(x + 1, x + width - 2)
                    hole_y = y + height - 1
                elif side == 'left':
                    hole_x = x
                    hole_y = random.randint(y + 1, y + height - 2)
                else:  # right
                    hole_x = x + width - 1
                    hole_y = random.randint(y + 1, y + height - 2)
                
                # Check if this hole position is already taken
                if grid[hole_y, hole_x] != 0:
                    grid[hole_y, hole_x] = 0
                    holes.append((side, hole_x, hole_y))
                    break
        
        # Generate description
        holes_description = ", ".join([f"one on the {side} edge at ({hx}, {hy})" for side, hx, hy in holes])
        description = f"A {self.color_map[color]} rectangle outline with {num_holes} holes: {holes_description}. The rectangle is located at ({x}, {y}) with width {width} and height {height}."
        
        return grid, description
class ShapeShooter(SimpleGridDescriptionTask):
    def __init__(self, grid_size):
        super().__init__(grid_size)
        self.shape_size = 3

    def generate_base_shape(self, color):
        shape = np.zeros((self.shape_size, self.shape_size), dtype=int)
        
        # Always include the center
        shape[1, 1] = color
        
        # List of all possible positions excluding the center
        positions = [(i, j) for i in range(3) for j in range(3) if (i, j) != (1, 1)]
        
        # Randomly choose at least 4 more positions to fill (5 total including center)
        num_to_fill = random.randint(4, 8)
        fill_positions = random.sample(positions, num_to_fill)
        
        for pos in fill_positions:
            shape[pos] = color
        
        return shape

    def place_shape(self, grid, shape, x, y):
        for i in range(self.shape_size):
            for j in range(self.shape_size):
                if 0 <= x+i < self.grid_size and 0 <= y+j < self.grid_size:
                    if shape[i, j] != 0:  # Only place non-zero values
                        grid[x+i, y+j] = shape[i, j]

    def shoot_shape(self, grid, base_shape, start_x, start_y, direction) -> bool:
        dx, dy = direction
        x, y = start_x, start_y

        # Generate a new color for this direction, different from the base color
        new_color = random.randint(1, 6)
        while new_color == base_shape[1, 1]:  # Compare with center color
            new_color = random.randint(1, 6)
        
        # Create a new shape with the same pattern but new color
        new_shape = np.where(base_shape != 0, new_color, 0)
        shape_placed = False
        while True:
            x += dx * (self.shape_size + 1)
            y += dy * (self.shape_size + 1)
            
            # Check if the shape is completely out of bounds
            if (x >= self.grid_size or x + self.shape_size <= 0 or
                y >= self.grid_size or y + self.shape_size <= 0):
                break
            shape_placed = True
            self.place_shape(grid, new_shape, x, y)
        return shape_placed

    def execute(self) -> tuple[np.ndarray, str]:
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Generate monocolored base shape
        base_color = random.randint(1, 6)
        base_shape = self.generate_base_shape(base_color)

        # Place the initial shape randomly
        start_x = random.randint(0, self.grid_size - self.shape_size)
        start_y = random.randint(0, self.grid_size - self.shape_size)
        self.place_shape(grid, base_shape, start_x, start_y)

        # Choose between cardinal and diagonal directions
        if random.choice([True, False]):
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # Cardinal
            direction_name = "cardinal"
        else:
            directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]  # Diagonal
            direction_name = "diagonal"

        # Shoot shapes in random directions
        num_directions = random.randint(1, len(directions))
        chosen_directions = random.sample(directions, num_directions)
        
        final_direction_count = 0
        for direction in chosen_directions:
            final_direction_count += int(self.shoot_shape(grid, base_shape, start_x, start_y, direction))

        description = f"A grid with a random 3x3 base shape shooting in {final_direction_count} {direction_name} direction(s) with same-patterned copies in different colors extending to the board edges"
        return grid, description

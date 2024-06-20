import random


def sample_island(n, m, max_land_cells=None):
    # creates a super connected island of size at most n x m
    # returns as a list of int[][]
    # the island is guaranteed to be connected
    # the island is guaranteed to be at most n x m

    # helper function to check if a cell is in the grid
    def in_grid(x, y):
        return 0 <= x < n and 0 <= y < m

    # directions for moving in the grid (right, down, left, up)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    # initialize the grid with water (0)
    grid = [[0 for _ in range(m)] for _ in range(n)]

    # start from a random cell
    start_x = random.randint(0, n - 1)
    start_y = random.randint(0, m - 1)
    grid[start_x][start_y] = 1  # mark the starting cell as land (1)

    # list of cells to expand from
    cells_to_expand = [(start_x, start_y)]

    # counter for the number of land cells
    land_cells = 1

    # set the maximum number of land cells if not specified
    if max_land_cells is None:
        max_land_cells = n * m

    while cells_to_expand and land_cells < max_land_cells:
        x, y = cells_to_expand.pop()
        random.shuffle(
            directions
        )  # shuffle directions to randomize the shape of the island
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if in_grid(new_x, new_y) and grid[new_x][new_y] == 0:
                grid[new_x][new_y] = 1  # mark the new cell as land
                cells_to_expand.append((new_x, new_y))
                land_cells += 1
                if land_cells >= max_land_cells:
                    break

    return grid


import random
import matplotlib.pyplot as plt


def hollow_sample_island(n, m, num_extensions=5):
    # initialize the grid with water (0)
    grid = [[0 for _ in range(m)] for _ in range(n)]

    def in_grid(x, y):
        return 0 <= x < n and 0 <= y < m

    # create initial random rectangle
    rect_w = random.randint(1, m // 3)
    rect_h = random.randint(1, n // 3)
    top_left_x = random.randint(0, n - rect_h)
    top_left_y = random.randint(0, m - rect_w)

    for i in range(top_left_x, top_left_x + rect_h):
        for j in range(top_left_y, top_left_y + rect_w):
            grid[i][j] = 1

    # function to extend the shape
    def extend_shape():
        # pick a random direction to extend
        direction = random.choice(["up", "down", "left", "right"])
        if direction == "up" and top_left_x > 0:
            new_top_left_x = max(0, top_left_x - random.randint(1, rect_h))
            for i in range(new_top_left_x, top_left_x):
                for j in range(top_left_y, top_left_y + rect_w):
                    if in_grid(i, j):
                        grid[i][j] = 1
        elif direction == "down" and top_left_x + rect_h < n:
            new_bottom_x = min(n, top_left_x + rect_h + random.randint(1, rect_h))
            for i in range(top_left_x + rect_h, new_bottom_x):
                for j in range(top_left_y, top_left_y + rect_w):
                    if in_grid(i, j):
                        grid[i][j] = 1
        elif direction == "left" and top_left_y > 0:
            new_top_left_y = max(0, top_left_y - random.randint(1, rect_w))
            for i in range(top_left_x, top_left_x + rect_h):
                for j in range(new_top_left_y, top_left_y):
                    if in_grid(i, j):
                        grid[i][j] = 1
        elif direction == "right" and top_left_y + rect_w < m:
            new_right_y = min(m, top_left_y + rect_w + random.randint(1, rect_w))
            for i in range(top_left_x, top_left_x + rect_h):
                for j in range(top_left_y + rect_w, new_right_y):
                    if in_grid(i, j):
                        grid[i][j] = 1

    # extend the initial shape randomly
    for _ in range(num_extensions):
        extend_shape()

    # hollow out the inside
    def hollow_out():
        for i in range(1, n - 1):
            for j in range(1, m - 1):
                if grid[i][j] == 1 and all(
                    grid[x][y] == 1
                    for x, y in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                ):
                    grid[i][j] = 0

    hollow_out()
    return grid


# Example usage:
n, m = 10, 10
island = hollow_sample_island(n, m, num_extensions=5)

# Plotting the result
plt.figure(figsize=(8, 8))
plt.imshow(island, cmap="Blues")
plt.title("Extended Island with Hollow Inside")
plt.show()

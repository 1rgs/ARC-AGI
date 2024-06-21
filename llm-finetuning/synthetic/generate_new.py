import openai
#import instructor
import functools
import re

class_defn = """
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
        
"""

OPENAI_API_KEY = "<YOUR_API_KEY>"  # Skylake
# Define the model name
GPT4O_MODEL_NAME = "gpt-4o"

DEFAULT_MODEL = GPT4O_MODEL_NAME

simple_pattern = open("simplepattern.py", "r").read()

def extract_python_code(text: str) -> str:
    # Regular expression to find code enclosed in triple backticks
    code_block_pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
    matches = code_block_pattern.findall(text)

    if matches:
        return matches[0]
    else:
        return None

def generate_tasks(num_tasks: int = 1):    
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    messages = [{"role": "system", "content": "You are a top 1% software engineer who is highly compensated for their work. Please do these tasks very carefully."}]
    messages.append({"role": "user", "content": f"Generate a new subclass of SimpleGridDescriptionTask that should correspond to a natural task. Make sure that it does not involve CIRCLES; if it involves CIRCLES think through something new. Think through an idea first, and then start writing the code. Here's the code {simple_pattern}"})

    result = (
        openai_client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            max_tokens=4096,
            temperature=0.8,
            n = num_tasks,
        )
    )

    all_results = [extract_python_code(result.choices[i].message.content) for i in range(num_tasks)]
    for task in range(num_tasks):
        with open(f"generated_generators/task_{task}.py", "w") as f:
            f.write(class_defn)
            f.write(all_results[task])

generate_tasks(num_tasks=8)
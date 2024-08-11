import anthropic
import functools
import re
import dotenv
import os
import json

dotenv.load_dotenv()

# Read class definition from "simplepattern.py"
class_defn = open("synthetic/simple_grid_definition_task.py", "r").read()

CLAUDE_MODEL = "claude-3-sonnet-20240229"

# Boolean flag to control the mode
GENERATE_IDEAS_ONLY = False


def extract_python_code(text: str) -> str:
    code_block_pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
    matches = code_block_pattern.findall(text)
    return matches[0] if matches else None


def generate_ideas(num_ideas: int = 8):
    client = anthropic.Anthropic()

    prompt = f"""
    Generate {num_ideas} unique ideas for subclasses of SimpleGridDescriptionTask. 
    Each idea should describe a specific pattern or arrangement that can be represented on a grid.
    Return the ideas as a Python list of strings, enclosed in a Python code block.
    """

    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )

    print("ideas", message.to_json())
    return message.to_json()


def generate_task_code(idea: str):
    client = anthropic.Anthropic()

    docstring = """
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
            - if you are placing shapes, it's very important that you are clear about where it is going. If you place a shape and another shape is over it
                make sure to mention that the shapes are overlapping and on which sides they are touching. Be clear in your code 
                when placing multiple shapes on how they interact. Check if they are overlapping or touching and mention it in the description, and avoid it if possible.


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
            - if two shapes touch each other, make sure to mention that they are touching, and on which sides. the same goes for overlapping shapes.
            - if you don't want to overly complicate the description, avoid overlaps or touching shapes. if they do touch or overlap, make sure to mention it.
    """

    prompt = f"""
    Generate a new subclass of SimpleGridDescriptionTask based on the following idea:
    {idea}

    Here's the base class definition to refer to:
    {class_defn}

    Please implement the execute method according to the following docstring: 
    {docstring}
    
    Make sure to import the existing ABC class, don't reimplement it.
    Return only the Python code for the new subclass, enclosed in a Python code block.
    Ensure that the description generated in the execute method follows the guidelines in the docstring.

    Also, add some code at the end to execute the implemented task. Here's an example:
    if __name__ == "__main__":
        task = ColoredRectanglesTask()
        grid, description = task.execute()
        print(description)
        task.visualize(grid, description)
    """

    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    print("task_code", message)
    return extract_python_code(message.content[0].text)


ideas = [
    "Single Square: Place a single square of random color at a random position in the grid.",
    "Single Rectangle: Generate a single rectangle with random dimensions and color, positioned randomly in the grid.",
    "Single Pyramid: Create a single pyramid (triangle pointing upwards) of random color and size, placed at a random position in the grid.",
    "Single Right Triangle: Draw a single right-angled triangle with random color and size, positioned randomly in one of the four corners of the grid.",
    "Centered Square: Place a single square of random color and size exactly in the center of the grid.",
    "Corner Square: Position a single square of random color in one of the four corners of the grid, with random size.",
    "Two Squares: Place two squares of different colors at random positions in the grid.",
    "Square and Triangle: Draw a square and a right triangle of different colors at random positions.",
    "Colored Border: Create a single-cell wide border around the grid with a random color.",
    "Diagonal Line: Draw a diagonal line of random color from one corner to the opposite corner.",
    "Cross Shape: Create a cross shape ('+') of random color centered in the grid.",
    "L-Shape: Form an L-shape of random color positioned in one of the corners.",
    "T-Shape: Generate a T-shape of random color positioned along one of the edges.",
    "Checkerboard Corner: Fill one quarter of the grid (a corner) with a 2x2 checkerboard pattern.",
    "Parallel Lines: Draw two parallel lines of different colors across the grid.",
    "Central Plus: Create a plus sign ('+') of one color with a different color in the center cell.",
    "Four Corners: Place a small square of a different color in each of the four corners of the grid.",
]


def main():
    global ideas
    if GENERATE_IDEAS_ONLY:
        # First pass: Generate ideas
        ideas = generate_ideas()
        # Save ideas to a file, create a new file if it doesn't exist
        with open("generated_ideas.json", "w") as f:
            json.dump(ideas, f, indent=2)
        print("Ideas saved to generated_ideas.json")
    else:
        # Second pass: Generate code for each idea
        for i, idea in enumerate(ideas, 1):
            code = generate_task_code(idea)
            if code:
                # create a new file for each task, if it doesn't exist make a new file
                with open(f"synthetic/gen/task_{i}.py", "w") as f:
                    f.write(
                        f"from ..simple_grid_definition_task import SimpleGridDescriptionTask\n\n"
                    )
                    f.write(code)
                print(f"Generated code for task {i}")
            else:
                print(f"Failed to generate code for task {i}")


if __name__ == "__main__":
    main()

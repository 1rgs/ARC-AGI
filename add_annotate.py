import pathlib
import json
import pandas as pd

def load_json(folder) -> pd.DataFrame:
    data = []
    for f in pathlib.Path(folder).iterdir():
        with open(f) as file:
            json_data = json.load(file)
            json_data['filename'] = f.stem  # Add filename without the .json extension
            data.append(json_data)
    # Normalize the JSON data
    normalized_data = pd.json_normalize(data)
    return pd.DataFrame(normalized_data).drop(columns = ["name"])

# Assuming the folder is 'data/training'
result = load_json('data/training')
result.to_csv('training_annotate.csv', index=False)



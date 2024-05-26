import json
import os

def json_to_list(data_path):
    with open(data_path) as f:
        data = json.load(f)
    return data
import json
import os

def json_to_list(data_path):
    with open(data_path) as f:
        data = json.load(f)
    return data


# Extracts the model results between start and end tokens returned
def extract_between_tags(text):
    start_tag = '<pad>'
    end_tag = '</s>'

    # Find the positions of the start and end tags
    start_pos = text.find(start_tag)
    end_pos = text.find(end_tag)
    
    # Check none of the tags are present, just return text
    if start_pos == -1 or end_pos == -1:
        return text
    
    # Extract the content between the tags
    extracted_content = text[start_pos + len(start_tag):end_pos].strip()
    return extracted_content

# TODO: Convert dict to Huggingface Dataset
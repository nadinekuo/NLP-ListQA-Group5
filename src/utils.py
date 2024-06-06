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


# Remove tokens for <pad> and </s> which correspond to 0 and 1
# NOTE: check this per model tokenizer!
def remove_pad_tokens(tokens_lst):
    try:
        # Find the index of the first occurrence of 0
        start_index = tokens_lst.index(0)
        # Find the index of the first occurrence of 1 after the 0
        end_index = tokens_lst.index(1, start_index + 1)
        # Slice the list from the first 0 to the first 1 (inclusive)
        return tokens_lst[start_index + 1:end_index]
    except ValueError:
        # If 0 or 1 is not found, return the original list
        return tokens_lst
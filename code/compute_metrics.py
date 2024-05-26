import re
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

# NOTE: Each tokenized list typically starts with 784, 31 and ends with 908, 1 because of the [''] structure

# Exact match: strict matching on each list item (entity-time pair)
def compute_em(ground_truth_list, pred_list):
    # Calculate the match score for each ground truth item
    matches = [1 if gt in pred_list else 0 for gt in ground_truth_list]
    
    # Compute the average exact match score based on the length of the ground truth list
    exact_match_score = sum(matches) / len(ground_truth_list)
    
    return exact_match_score


# TODO: reuse for time (and entities)
def compute_precision(ground_truth, prediction):
    # Convert the lists to sets to facilitate comparison
    ground_truth_set = set(ground_truth)
    prediction_set = set(prediction)
    
    true_positives = len(ground_truth_set & prediction_set)
    if len(prediction_set) == 0:
        return 0.0
    return true_positives / len(prediction_set)


# Evaluates completeness
def compute_recall(ground_truth, prediction):
    # Convert the lists to sets to facilitate comparison
    ground_truth_set = set(ground_truth)
    prediction_set = set(prediction)
    
    true_positives = len(ground_truth_set & prediction_set)
    if len(ground_truth_set) == 0:
        return 0.0
    return true_positives / len(ground_truth_set)


def compute_f1_score(ground_truth, prediction):
    # Convert the lists to sets to facilitate comparison
    ground_truth_set = set(ground_truth)
    prediction_set = set(prediction)
    
    # Calculate precision and recall
    precision = compute_precision(ground_truth_set, prediction_set)
    recall = compute_recall(ground_truth_set, prediction_set)
    
    # Calculate F1 score
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    return f1_score
    

def extract_time_ranges(entity_list):
    time_ranges = []
    for entity in entity_list:
        # Extract the time ranges using a regular expression
        match = re.search(r'\((\d{4}.*)\)', entity)
        if match:
            time_range = f"({match.group(1)})"
            time_ranges.append(time_range)
    return time_ranges

def compute_time_precision_score(ground_truth, prediction):
    # Extract time ranges from the ground truth and prediction
    ground_truth_times = extract_time_ranges(ground_truth)
    prediction_times = extract_time_ranges(prediction)
    
    # Calculate true positives (correctly predicted time intervals)
    true_positives = len(set(ground_truth_times) & set(prediction_times))
    
    # Calculate precision
    if len(prediction_times) == 0:
        return 0.0
    return true_positives / len(prediction_times)


def extract_entities(entity_list):
    entities = []
    for entity in entity_list:
        # Extract the part before the parentheses or part within parentheses that does not contain digits
        match = re.match(r'(.*?)(?: \((\D+?)\))? \((\d{4}.*)\)', entity)
        if match:
            entity_name = match.group(1).strip()
            entities.append(entity_name)
        else:
            # Check if there's a part within parentheses that does not contain digits
            match = re.match(r'(.*?) \((\D+?)\)$', entity)
            if match:
                entity_name = f"{match.group(1).strip()} ({match.group(2).strip()})"
                entities.append(entity_name)
    return entities

def compute_entity_precision_score(ground_truth, prediction):
    # Extract entities from the ground truth and prediction
    ground_truth_entities = extract_entities(ground_truth)
    prediction_entities = extract_entities(prediction)
    
    # Calculate true positives (correctly predicted entities)
    true_positives = len(set(ground_truth_entities) & set(prediction_entities))
    
    # Calculate precision
    if len(prediction_entities) == 0:
        return 0.0
    return true_positives / len(prediction_entities)
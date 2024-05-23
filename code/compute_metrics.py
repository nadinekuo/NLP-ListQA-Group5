import re

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


def compute_em(ground_truth_list, pred_list):
    # Convert the lists to sets to compare - we don't care about ordering
    ground_truth_set = set(ground_truth_list)
    prediction_set = set(pred_list)
    # Calculate the exact match score
    if ground_truth_set == prediction_set:
        return 1.0
    else:
        return 0.0


def precision_score(ground_truth, prediction):
    # Convert the lists to sets to facilitate comparison
    ground_truth_set = set(ground_truth)
    prediction_set = set(prediction)
    
    true_positives = len(ground_truth_set & prediction_set)
    if len(prediction_set) == 0:
        return 0.0
    return true_positives / len(prediction_set)


# Evaluates completeness
def recall_score(ground_truth, prediction):
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
    precision = precision_score(ground_truth_set, prediction_set)
    recall = recall_score(ground_truth_set, prediction_set)
    
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
        match = re.search(r'\((.*?)\)', entity)
        if match:
            times = match.group(1).split(', ')
            time_ranges.extend(times)
    print(time_ranges)
    return set(time_ranges)

def time_precision_score(ground_truth, prediction):
    # Extract time ranges from the ground truth and prediction
    ground_truth_times = extract_time_ranges(ground_truth)
    prediction_times = extract_time_ranges(prediction)
    
    # Calculate true positives (correctly predicted time intervals)
    true_positives = len(ground_truth_times & prediction_times)
    
    # Calculate precision
    if len(prediction_times) == 0:
        return 0.0
    return true_positives / len(prediction_times)


def extract_entities(entity_list):
    entities = []
    for entity in entity_list:
        # Extract the part before the parentheses using a regular expression
        match = re.match(r'(.*?) \(', entity)
        if match:
            entities.append(match.group(1).strip())
    print(entities)
    return set(entities)

def entity_precision_score(ground_truth, prediction):
    # Extract entities from the ground truth and prediction
    ground_truth_entities = extract_entities(ground_truth)
    prediction_entities = extract_entities(prediction)
    
    # Calculate true positives (correctly predicted entities)
    true_positives = len(ground_truth_entities & prediction_entities)
    
    # Calculate precision
    if len(prediction_entities) == 0:
        return 0.0
    return true_positives / len(prediction_entities)
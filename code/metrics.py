import re
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

# NOTE: Each tokenized list typically starts with 784, 31 and ends with 908, 1 because of the [''] structure
# We assume these have been removed before metrics computation (see helpers in utils.py)


# Exact match: strict all or nothing matching on each list item (entity-time pair)
# This is not computed on token-level
def compute_em(ground_truth_list, pred_list):
    # Calculate the match score for each ground truth item
    matches = [1 if gt in pred_list else 0 for gt in ground_truth_list]
    
    # Compute the average exact match score based on the length of the ground truth list
    exact_match_score = sum(matches) / len(ground_truth_list)
    
    return exact_match_score


# F1: word overlap between the labeled and the predicted answer
# For both precision and recall, match on TOKENS over the full list of Entity-Time pairs
def compute_f1(gt_tokens, pred_tokens):
    precision = compute_precision(gt_tokens, pred_tokens)
    recall = compute_recall(gt_tokens, pred_tokens)
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    return f1_score
    

# Computed as # correct tokens / # pred tokens
def compute_precision(gt_tokens, pred_tokens):    
    true_positives = len(set(gt_tokens) & set(pred_tokens))
    if len(pred_tokens) == 0:
        return 0.0
    return true_positives / len(pred_tokens)


# Evaluates completeness: # correct answers / # GT answers 
def compute_recall(gt_tokens, pred_tokens):        
    true_positives = len(set(gt_tokens) & set(pred_tokens))
    if len(gt_tokens) == 0:
        return 0.0
    return true_positives / len(gt_tokens)


def extract_time_ranges(entity_list):
    time_ranges = []
    for entity in entity_list:
        # Extract the time ranges using a regular expression
        match = re.search(r'\((\d{4}.*)\)', entity)
        if match:
            time_range = f"({match.group(1)})"
            time_ranges.append(time_range)
    return time_ranges


# TimePrecision = # correct time ranges / # time ranges given
# Computed over time ranges (NOT tokenized, as we wanna match on full year)
def compute_time_precision(ground_truth, prediction):
    ground_truth_times = extract_time_ranges(ground_truth)
    prediction_times = extract_time_ranges(prediction)
    true_positives = len(set(ground_truth_times) & set(prediction_times))
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

# TODO: tokenize extracted entities

# TODO: use NLTK to compute BLEU 
def compute_entity_bleu(gt_tokens, pred_tokens):
    # BLEU_entity: over entity items (tokenized and concatenated) -> use sentence_blue or corpus_bleu
    # It calculates a precision score for each n-gram size (typically 1 to 4) and then computes a geometric mean of these precisions.
    # entity_bleu_score = corpus_bleu(example_gt, example_pred_list)
    # print(entity_bleu_score)
    return 0

# def compute_entity_precision_score(ground_truth, prediction):
#     # Extract entities from the ground truth and prediction
#     ground_truth_entities = extract_entities(ground_truth)
#     prediction_entities = extract_entities(prediction)
    
#     # Calculate true positives (correctly predicted entities)
#     true_positives = len(set(ground_truth_entities) & set(prediction_entities))
    
#     # Calculate precision
#     if len(prediction_entities) == 0:
#         return 0.0
#     return true_positives / len(prediction_entities)
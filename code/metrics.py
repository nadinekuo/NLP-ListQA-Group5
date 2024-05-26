import re
from nltk.translate.bleu_score import sentence_bleu
from transformers import BertTokenizer, BertForMaskedLM, BertModel
from bert_score import BERTScorer

# NOTE: Each tokenized list typically starts with 784, 31 and ends with 908, 1 because of the [''] structure
# We assume these have been removed before metrics computation (see helpers in utils.py)


# Exact match: strict all or nothing matching on each list item (entity-time pair)
# This is not computed on token-level
def compute_em(gt_list, pred_list):
    # Calculate the match score for each ground truth item
    matches = [1 if gt in pred_list else 0 for gt in gt_list]
    
    # Compute the average exact match score based on the length of the ground truth list
    exact_match_score = sum(matches) / len(gt_list)
    
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


# Computes BLEU: precision-based overlap between predicted time ranges and ground truth time ranges
# The NLTK implementation computes BLEU-1, BLEU-2, BLEU-3, BLEU-4 and averages
# We can do this on token-level, since each year XXXX has a unique token
def compute_time_bleu(gt_list, pred_list, tokenizer):
    gt_time_list = extract_time_ranges(gt_list)
    pred_time_list = extract_time_ranges(pred_list)

    if len(pred_time_list) == 0:
        return 0
    
    # Tokenize each timeline item '(XXXX, ...)' - String converted to List!
    gt_time_list = list(map(lambda x: tokenizer(x, return_tensors="pt").input_ids[0].tolist(), gt_time_list))
    pred_time_list = list(map(lambda x: tokenizer(x, return_tensors="pt").input_ids[0].tolist(), pred_time_list))
    
    # Remove token 1 from all inner lists, since that denotes stop token </s>
    gt_time_list = [lst[:-1] if lst and lst[-1] == 1 else lst for lst in gt_time_list]
    pred_time_list = [lst[:-1] if lst and lst[-1] == 1 else lst for lst in pred_time_list]

    # print(f"\nPred time range tokens: {pred_time_list}")
    # print(f"GT time range tokens: {gt_time_list}\n")

    total_bleu = 0
    for i, pred in enumerate(pred_time_list):
        curr_bleu = sentence_bleu(gt_time_list, pred)  # BLEU for curr pred item against all reference items
        # print(f"Curr BLEU for pred {pred}: {curr_bleu}")
        total_bleu += curr_bleu
    total_bleu = total_bleu / len(pred_time_list)
    return total_bleu

# Uses BERT tokenizer under the hood to compute similarity between prediction and GT
# As this takes into semantics (unlike BLEU e.g.) this can detect synonyms
def compute_entity_bert(gt_list, pred_list):
    gt_entity_list = extract_entities(gt_list)
    pred_entity_list = extract_entities(pred_list)

    if len(pred_entity_list) == 0:
        return 0

    # Concatenate all strings in pred and gt lists
    concat_gt = ' '.join(gt_entity_list)
    concat_pred = ' '.join(pred_entity_list)

    scorer = BERTScorer(model_type='bert-base-uncased')  # NOTE: this can take some time to load
    P, R, F1 = scorer.score([concat_gt], [concat_pred])
    # print(f"BERTScore Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")
    return F1.tolist()[0]    # Convert Tensor to List


def extract_time_ranges(entity_list):
    time_ranges = []
    for entity in entity_list:
        # Extract the time ranges using a regular expression
        match = re.search(r'\((\d{4}.*)\)', entity)
        if match:
            time_range = f"({match.group(1)})"
            time_ranges.append(time_range)
    return time_ranges

def extract_entities(entity_list):
    entities = []
    for entity in entity_list:
        # Extract the part before the parentheses or part within parentheses that does not contain digits
        match = re.match(r'(.*?)(?: \((\D+?)\))? \((\d{4}.*)\)', entity)
        if match:
            entity_name = match.group(1).strip()
            entities.append(entity_name)
        else:
            entities.append(entity)
    return entities
import metrics
import utils
import re
from datasets import Dataset
from transformers import T5Tokenizer
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Eval performance')
    parser.add_argument('--data-dir', default='../results/3_shot_flan-t5-large.hf')
    parser.add_argument('--tokenizer-name', default='google/flan-t5-large')
    parser.add_argument('--output-dir', default='../results/3_shot_flan-t5-large.txt')
    args = parser.parse_args()
    return args

def eval_performance(results_ds, tokenizer):

    num_results = len(results_ds['prompts'])  # Any arbitrary feature works
    total_em = 0
    total_f1 = 0
    total_recall = 0
    total_timebleu = 0
    total_entitybert = 0

    for i in range(num_results):
        prompt = results_ds['prompts'][i]
        pred = results_ds['outputs'][i]
        pred_tokens = results_ds['output_tokens'][i]
        gt_tokens = results_ds['ground_truth_tokens'][i]
        gt = results_ds['ground_truths'][i]

        pred_tokens = utils.remove_pad_tokens(pred_tokens)
        pred = utils.extract_between_tags(pred)
        pred = re.findall(r"'(.*?)'", pred)  # We convert the output string into list to facilitate metrics computation

        # ---------------------- Syntax-based metrics --------------------------
        em = metrics.compute_em(gt, pred)
        f1 = metrics.compute_f1(gt_tokens, pred_tokens)
        recall = metrics.compute_recall(gt_tokens, pred_tokens)
        time_bleu = metrics.compute_time_bleu(gt, pred, tokenizer)  # NOTE: tends to be relatively high, since the time range is already hinted at in the prompt
        # ---------------------- Semantics-based metrics --------------------------
        entity_bert = metrics.compute_entity_bert(gt, pred)   # Uses BERT tokenizer
        print(f"\nTest item {i}")
        print(f"EM: {em}, F1: {f1}, Recall: {recall}, Time-BLEU: {time_bleu}, Entity-BERT: {entity_bert}")

        total_em += em
        total_f1 += f1
        total_recall += recall
        total_timebleu += time_bleu
        total_entitybert += entity_bert 

    total_em = total_em / num_results * 100
    total_f1 = total_f1 / num_results * 100
    total_recall = total_recall / num_results * 100
    total_timebleu = total_timebleu / num_results * 100
    total_entitybert = total_entitybert / num_results * 100
    results_dict = {'EM': total_em, 'F1': total_f1, 'Recall': total_recall, 'Time-BLEU': total_timebleu, 'Entity-BERT': total_entitybert}
    
    return results_dict


if __name__ == '__main__':

    args = parse_args()
    print(args)
    print(f"\n\nLoading dataset...")
    data = Dataset.load_from_disk(f"{args.data_dir}")
    print(data)
    print(f"\nDataset length: {len(data)}\n\n")

    tok = T5Tokenizer.from_pretrained(args.tokenizer_name, torch_dtype=torch.float16)
    # tok = T5Tokenizer.from_pretrained("google/flan-t5-xl", torch_dtype=torch.float16)

    res_dict = eval_performance(data, tok)

    with open(f"{args.output_dir}", "w") as f:
        f.write(str(res_dict))
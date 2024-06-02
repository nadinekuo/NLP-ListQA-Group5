from knn import KnnSearch
from utils import json_to_list
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from datasets import Dataset
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
# import accelerate
import argparse

# NOTE: This script is based on TLQA_few_shot_ipynb, but adapted to run using GPU

def parse_args():
    parser = argparse.ArgumentParser(description='Few shot eval')
    parser.add_argument('--k', default=3)
    parser.add_argument('--model-name', default='google/flan-t5-large')
    args = parser.parse_args()
    return args


# Extracts all questions from the (train) set used for getting neighbours
def get_transfer_questions(transfer_data):
    transfer_questions = []
    for index, data in enumerate(transfer_data):
        transfer_questions.append(data["question"])
    return transfer_questions


def simplify_dict_list(dict_list):
    return [{'question': item['question'], 'answers': item['answers']} for item in dict_list]


def fewshot_eval(K, model_name, test_data, train_data, train_emb):  
    MAX_OUTPUT_LEN = 200
    
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_name, torch_dtype=torch.float16)
    
    results_GT_dict = {'prompts': [], 'outputs': [], 'output_tokens': [], 
                                'ground_truths': [], 'ground_truth_tokens': []}  

    # Configure formatter that will format the few-shot examples into a string
    example_prompt = PromptTemplate(
        input_variables=["question", "answers"], template="Question: {question}\n{answers}"
    )

    # Convert test set to list and loop over all items (1071 in total)
    for i, item in enumerate(test_data):
        
        # For each test question, retrieve k neighbours (try 3, 5, 7, 10)
        test_question = test_data[i]['question']
        print(f"Test question {i}: {test_question}")

        neighs = knn.get_top_n_neighbours(sentence=test_question, data_emb=train_emb, transfer_data=train_data, k=K)
        simple_neighs = simplify_dict_list(neighs)

        # Create the few-shot prompt template and feed to model
        prompt = FewShotPromptTemplate(
            examples=simple_neighs,
            example_prompt=example_prompt,
            suffix="Question: {input}",
            input_variables=["input"],
        )
        few_shot_prompt = prompt.format(input=f"{test_question} Please answer this question in the same format as the {K} examples above.")
        results_GT_dict['prompts'].append(few_shot_prompt)
        
        input_ids = tokenizer(few_shot_prompt, return_tensors="pt").input_ids.to(device)
        output_tokens = model.generate(input_ids, max_length=MAX_OUTPUT_LEN)
        output = tokenizer.decode(output_tokens[0])

        results_GT_dict['output_tokens'].append(output_tokens[0])
        results_GT_dict['outputs'].append(output)
        results_GT_dict['ground_truths'].append(test_data[i]['final_answers'])
        gt_tokens = tokenizer(str(test_data[i]['final_answers']), return_tensors="pt").input_ids[0]
        results_GT_dict['ground_truth_tokens'].append(gt_tokens)

    results_ds = Dataset.from_dict(results_GT_dict)
    results_ds.save_to_disk(f"{K}_shot_{model_name}.hf")  # TODO: Dataset name may have to changed for other experiments to prevent overwriting!



if __name__ == '__main__':

    # TODO: Try K-values [3, 5, 7, 10]
    # TODO: Models: "google/flan-t5-large" and "google/flan-t5-xl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    k = args.k
    model = args.model_name

    knn = KnnSearch()
    test_set = json_to_list("../data/test_TLQA.json")
    train_set = json_to_list("../data/train_TLQA.json")
    train_questions = get_transfer_questions(train_set)   # Keep questions only to embed (to use in similarity metric)
    train_questions_emb = knn.get_embeddings_for_data(train_questions)

    print(f"\n\nStarting {k}-shot evaluation on FlanT5-large...\n\n")
    fewshot_eval(K=k, model_name=model, test_data=test_set, train_data=train_set, train_emb=train_questions_emb)


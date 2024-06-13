#8
import os
from knn import KnnSearch
from utils import json_to_list
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from datasets import Dataset
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from sentence_transformers import SentenceTransformer, util
import json
import argparse

# Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(description='Few shot eval with context retrieval')
    parser.add_argument('--k', default=3, type=int)
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

# Few-shot Evaluation with Context Retrieval
def fewshot_eval_with_context(K, model_name, test_data, train_data, train_emb, infoboxes, retriever):  
    MAX_OUTPUT_LEN = 200
    MAX_SEQUENCE_LENGTH = 512  # Model's max sequence length
    HALF_MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH // 2
    
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_name, torch_dtype=torch.float16)
    
    results_GT_dict = {'prompts': [], 'outputs': [], 'output_tokens': [], 
                                'ground_truths': [], 'ground_truth_tokens': []}  
    # Configure formatter that will format the few-shot examples into a string
    example_prompt = PromptTemplate(
        input_variables=["question", "answers"], template="Question: {question}\n{answers}"
    )
    # Convert test set to list and loop over all items
    for i, item in enumerate(test_data):
        
        # For each test question, retrieve k neighbours
        test_question = test_data[i]['question']
        print(f"Test question {i}: {test_question}")
        # Retrieve k-nearest neighbors from training data
        neighs = knn.get_top_n_neighbours(sentence=test_question, data_emb=train_emb, transfer_data=train_data, k=K)
        simple_neighs = simplify_dict_list(neighs)
        # Retrieve top-K relevant contexts from infoboxes
        infobox_texts = [infobox['infobox'] for infobox in infoboxes]
        infobox_embeddings = retriever.encode(infobox_texts, convert_to_tensor=True)
        query_embedding = retriever.encode(test_question, convert_to_tensor=True)

        hits = util.semantic_search(query_embedding, infobox_embeddings, top_k=K)[0]
        top_infoboxes = [infoboxes[hit['corpus_id']]['infobox'] for hit in hits]

        # Concatenate top-K infoboxes ensuring the total length does not exceed the limit
        combined_infobox = ""
        total_length = 0
        for infobox in top_infoboxes:
            if total_length + len(infobox) > HALF_MAX_SEQUENCE_LENGTH:
                break
            combined_infobox += infobox + "\n"
            total_length += len(infobox)

        # Debug statement to verify the combined infobox content
        print(f"Combined infobox for Test Question {i}:\n{combined_infobox}\n")

        # Create the few-shot prompt template and feed to model
        prompt = FewShotPromptTemplate(
            examples=simple_neighs,  # No. of few shot examples is defined by sysarg K
            example_prompt=example_prompt,
            suffix="<s>[INST] <<SYS>>\nUse the following context to answer the question at the end. Do not use any other information. If you can't find the relevant information in the context, just say you don't have enough information to answer the question. Don't try to make up an answer.\n\n<</SYS>>\n\n{context}\n\nQuestion: {input} [/INST]",
            input_variables=["input", "context"],
        )
      
        few_shot_prompt = prompt.format(
            input=f"{test_question}\nPlease answer this question in the same format as the {K} examples above.",
            context=combined_infobox
        )

        # Print the prompt to see how it looks
        print(f"Few-shot Prompt for Test Question {i}:\n{few_shot_prompt}\n")
        
        results_GT_dict['prompts'].append(few_shot_prompt)
        
        input_ids = tokenizer(few_shot_prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQUENCE_LENGTH).input_ids.to(device)
        output_tokens = model.generate(input_ids, max_length=MAX_OUTPUT_LEN)
        output = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        results_GT_dict['output_tokens'].append(output_tokens[0])
        results_GT_dict['outputs'].append(output)
        results_GT_dict['ground_truths'].append(test_data[i]['final_answers'])
        gt_tokens = tokenizer(str(test_data[i]['final_answers']), return_tensors="pt").input_ids[0]
        results_GT_dict['ground_truth_tokens'].append(gt_tokens)
    results_ds = Dataset.from_dict(results_GT_dict)
    results_ds.save_to_disk(f"{K}_shot_{model_name}_with_context.hf")  # Ensure different name to prevent overwriting

if __name__ == '__mn__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    k = args.k
    model = args.model_name
    knn = KnnSearch()
    # Define absolute path to the data directory
    data_dir = os.path.abspath("../data")
    test_file_path = os.path.join(data_dir, "test_TLQA.json")
    train_file_path = os.path.join(data_dir, "train_TLQA.json")
    infoboxes_file_path = os.path.join(data_dir, "extracted_infoboxes_7500.json")
    # Print the current working directory and the contents of the data directory
    print("Current working directory:", os.getcwd())
    print("Absolute path of data directory:", data_dir)
    print("Contents of the data directory:")
    print(os.listdir(data_dir))
    if not os.path.exists(test_file_path):
        raise FileNotFoundError(f"{test_file_path} not found.")
    if not os.path.exists(train_file_path):
        raise FileNotFoundError(f"{train_file_path} not found.")
    if not os.path.exists(infoboxes_file_path):
        raise FileNotFoundError(f"{infoboxes_file_path} not found.")
    
    test_set = json_to_list(test_file_path)
    train_set = json_to_list(train_file_path)
    train_questions = get_transfer_questions(train_set)   # Keep questions only to embed (to use in similarity metric)
    train_questions_emb = knn.get_embeddings_for_data(train_questions)
    
    # Load infoboxes
    with open(infoboxes_file_path, 'r') as f:
        infoboxes = json.load(f)
    
    # Initialize retriever model
    retriever = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-v4')
    print(f"\n\nStarting {k}-shot evaluation on {model} with context retrieval...\n\n")
    fewshot_eval_with_context(K=k, model_name=model, test_data=test_set, train_data=train_set, train_emb=train_questions_emb, infoboxes=infoboxes, retriever=retriever)

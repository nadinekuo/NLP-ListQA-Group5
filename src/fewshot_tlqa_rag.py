#1
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

# Few-shot Evaluation with Context Retrieval
def fewshot_eval_with_context(K, model_name, test_data, train_data, train_emb, infoboxes, retriever):  
    MAX_OUTPUT_LEN = 200
    
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

        # Retrieve top-1 relevant context from infoboxes
        infobox_texts = [infobox['infobox'] for infobox in infoboxes]
        infobox_embeddings = retriever.encode(infobox_texts, convert_to_tensor=True)
        query_embedding = retriever.encode(test_question, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, infobox_embeddings, top_k=1)[0]
        top_infobox = infoboxes[hits[0]['corpus_id']]['infobox']
        
        # Create the few-shot prompt template and feed to model
        prompt = FewShotPromptTemplate(
            examples=simple_neighs,
            example_prompt=example_prompt,
            suffix="<s>[INST] <<SYS>>\nUse the following context to answer the question at the end. Do not use any other information. If you can't find the relevant information in the context, just say you don't have enough information to answer the question. Don't try to make up an answer.\n\n<</SYS>>\n\n{context}\n\nQuestion: {input} [/INST]",
            input_variables=["input", "context"],
        )
        few_shot_prompt = prompt.format(input=test_question, context=top_infobox)
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
    results_ds.save_to_disk(f"{K}_shot_{model_name}_with_context.hf")  # Ensure different name to prevent overwriting


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    k = args.k
    model = args.model_name

    knn = KnnSearch()

    # Define absolute path to the data directory
    data_dir = os.path.abspath("../data")
    test_file_path = os.path.join(data_dir, "test_TLQA.json")
    train_file_path = os.path.join(data_dir, "train_TLQA.json")
    infoboxes_file_path = os.path.join(data_dir, "extracted_infoboxes.json")

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




'''
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import torch
from datasets import Dataset
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
import argparse
import numpy as np
import faiss

def parse_args():
    parser = argparse.ArgumentParser(description='Few shot eval with RAG')
    parser.add_argument('--k', default=3, type=int, help='Number of contexts to retrieve')
    parser.add_argument('--model-name', default='google/flan-t5-large', help='Name of the generative model')
    parser.add_argument('--retriever-name', default='sentence-transformers/msmarco-distilbert-base-tas-b', help='Name of the retriever model')
    parser.add_argument('--test-data', default='../data/test_TLQA.json', help='Path to the test dataset')
    parser.add_argument('--train-data', default='../data/train_TLQA.json', help='Path to the train dataset')
    parser.add_argument('--train-emb', default='../data/train_TLQA_emb.npy', help='Path to the train embeddings')
    parser.add_argument('--infobox-data', default='../data/extracted_infoboxes.json', help='Path to the extracted infoboxes data')
    args = parser.parse_args()
    return args

def json_to_list(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data

def get_transfer_questions(transfer_data):
    return [data["question"] for data in transfer_data]

def simplify_dict_list(dict_list):
    return [{'question': item['question'], 'answers': item['answers']} for item in dict_list]

def retrieve_context(query, corpus, retriever, corpus_embeddings, top_k=1):
    query_embedding = retriever.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
    return [corpus[hit['corpus_id']] for hit in hits[0]]

def generate_answer_with_context(question, context, model, tokenizer):
    combined_context = " ".join(context)
    input_text = f"Context: {combined_context}\n\nQuestion: {question}\n\nAnswer (please format as a timeline):"
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512)
    outputs = model.generate(inputs.input_ids, max_length=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def knn_search(query_embedding, embeddings, k):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    distances, indices = index.search(np.array([query_embedding]), k)
    return indices[0]

def fewshot_eval(K, model_name, retriever_name, test_data, train_data, train_emb, infobox_data):  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    retriever = SentenceTransformer(retriever_name)
    
    with open(infobox_data) as f:
        infoboxes = json.load(f)

    corpus = [f"{item['title']}: {item['infobox']}" for item in infoboxes if 'title' in item and 'infobox' in item]
    corpus_embeddings = retriever.encode(corpus, convert_to_tensor=True)
    
    results_GT_dict = {'prompts': [], 'outputs': [], 'output_tokens': [], 
                       'ground_truths': [], 'ground_truth_tokens': []}

    for i, item in enumerate(test_data):
        test_question = item['question']
        
        # Retrieve few-shot examples using KNN
        test_question_embedding = retriever.encode(test_question, convert_to_tensor=True).cpu().numpy()
        few_shot_indices = knn_search(test_question_embedding, train_emb, k=K)
        few_shot_examples = [train_data[i] for i in few_shot_indices]
        
        # Retrieve top-k contexts
        top_k_contexts = retrieve_context(test_question, corpus, retriever, corpus_embeddings, top_k=K)
        
        # Create the few-shot prompt template and feed to model
        simple_neighs = simplify_dict_list(few_shot_examples)
        prompt = FewShotPromptTemplate(
            examples=simple_neighs,
            example_prompt=PromptTemplate(input_variables=["question", "answers"], template="Question: {question}\n{answers}"),
            suffix="Question: {input}",
            input_variables=["input"],
        )
        few_shot_prompt = prompt.format(input=f"{test_question} Please answer this question in the same format as the {K} examples above.")
        
        # Integrate context into the prompt
        full_prompt = f"Context: {' '.join(top_k_contexts)}\n\n{few_shot_prompt}"
        results_GT_dict['prompts'].append(full_prompt)
        
        input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(device)
        output_tokens = model.generate(input_ids, max_length=200)
        output = tokenizer.decode(output_tokens[0])

        results_GT_dict['output_tokens'].append(output_tokens[0])
        results_GT_dict['outputs'].append(output)
        results_GT_dict['ground_truths'].append(item['final_answers'])
        gt_tokens = tokenizer(str(item['final_answers']), return_tensors="pt").input_ids[0]
        results_GT_dict['ground_truth_tokens'].append(gt_tokens)

    results_ds = Dataset.from_dict(results_GT_dict)
    results_ds.save_to_disk(f"{K}_shot_{model_name}.hf")

def main():
    args = parse_args()
    
    # Load datasets
    test_set = json_to_list(args.test_data)
    train_set = json_to_list(args.train_data)
    train_questions = get_transfer_questions(train_set)
    train_questions_emb = np.load(args.train_emb)
    
    fewshot_eval(K=args.k, model_name=args.model_name, retriever_name=args.retriever_name, test_data=test_set, train_data=train_set, train_emb=train_questions_emb, infobox_data=args.infobox_data)

if __name__ == "__main__":
    main()





from knn import KnnSearch
from utils import json_to_list
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification
import torch
from datasets import Dataset
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
import argparse
import faiss
import numpy as np

# NOTE: This script is based on TLQA_few_shot_ipynb, but adapted to run using GPU

def parse_args():
    parser = argparse.ArgumentParser(description='Few shot eval with RAG')
    parser.add_argument('--k', default=3, type=int, help='Number of contexts to retrieve')
    parser.add_argument('--model-name', default='google/flan-t5-large', help='Name of the generative model')
    parser.add_argument('--retriever-name', default='facebook/contriever', help='Name of the retriever model')
    parser.add_argument('--test-data', default='../data/test_TLQA.json', help='Path to the test dataset')
    parser.add_argument('--train-data', default='../data/train_TLQA.json', help='Path to the train dataset')
    parser.add_argument('--train-emb', default='../data/train_TLQA_emb.npy', help='Path to the train embeddings')
    args = parser.parse_args()
    return args

def get_transfer_questions(transfer_data):
    return [data["question"] for data in transfer_data]

def simplify_dict_list(dict_list):
    return [{'question': item['question'], 'answers': item['answers']} for item in dict_list]

def retrieve_top_k_contexts(query, corpus, retriever, tokenizer, k=1):
    inputs = tokenizer(query, corpus, return_tensors='pt', padding=True, truncation=True)
    outputs = retriever(**inputs)
    top_k_indices = outputs.logits.topk(k).indices
    return [corpus[i] for i in top_k_indices]

def generate_answer_with_context(question, context, model, tokenizer):
    combined_context = " ".join(context)
    input_text = f"Context: {combined_context}\n\nQuestion: {question}\n\nAnswer (please format as a timeline):"
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512)
    outputs = model.generate(inputs.input_ids, max_length=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def knn_search(query_embedding, embeddings, k):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    distances, indices = index.search(np.array([query_embedding]), k)
    return indices[0]

def fewshot_eval(K, model_name, retriever_name, test_data, train_data, train_emb):  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    retriever = AutoModelForSequenceClassification.from_pretrained(retriever_name).to(device)
    retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_name)
    
    results_GT_dict = {'prompts': [], 'outputs': [], 'output_tokens': [], 
                       'ground_truths': [], 'ground_truth_tokens': []}

    for i, item in enumerate(test_data):
        test_question = item['question']
        
        # Retrieve few-shot examples using KNN
        test_question_embedding = retriever_tokenizer(test_question, return_tensors='pt').input_ids
        test_question_embedding = retriever(**test_question_embedding).logits.detach().cpu().numpy().flatten()
        few_shot_indices = knn_search(test_question_embedding, train_emb, k=K)
        few_shot_examples = [train_data[i] for i in few_shot_indices]
        
        # Retrieve top-k contexts
        top_k_contexts = retrieve_top_k_contexts(test_question, train_data, retriever, retriever_tokenizer, k=K)
        
        # Create the few-shot prompt template and feed to model
        simple_neighs = simplify_dict_list(few_shot_examples)
        prompt = FewShotPromptTemplate(
            examples=simple_neighs,
            example_prompt=PromptTemplate(input_variables=["question", "answers"], template="Question: {question}\n{answers}"),
            suffix="Question: {input}",
            input_variables=["input"],
        )
        few_shot_prompt = prompt.format(input=f"{test_question} Please answer this question in the same format as the {K} examples above.")
        
        # Integrate context into the prompt
        full_prompt = f"Context: {' '.join(top_k_contexts)}\n\n{few_shot_prompt}"
        results_GT_dict['prompts'].append(full_prompt)
        
        input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(device)
        output_tokens = model.generate(input_ids, max_length=200)
        output = tokenizer.decode(output_tokens[0])

        results_GT_dict['output_tokens'].append(output_tokens[0])
        results_GT_dict['outputs'].append(output)
        results_GT_dict['ground_truths'].append(item['final_answers'])
        gt_tokens = tokenizer(str(item['final_answers']), return_tensors="pt").input_ids[0]
        results_GT_dict['ground_truth_tokens'].append(gt_tokens)

    results_ds = Dataset.from_dict(results_GT_dict)
    results_ds.save_to_disk(f"{K}_shot_{model_name}.hf")

def main():
    args = parse_args()
    
    # Load datasets
    test_set = json_to_list(args.test_data)
    train_set = json_to_list(args.train_data)
    train_questions = get_transfer_questions(train_set)
    train_questions_emb = np.load(args.train_emb)
    
    knn = KnnSearch()
    
    fewshot_eval(K=args.k, model_name=args.model_name, retriever_name=args.retriever_name, test_data=test_set, train_data=train_set, train_emb=train_questions_emb)

if __name__ == "__main__":
    main()
'''

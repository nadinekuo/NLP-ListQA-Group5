import json
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification
import torch
from datasets import Dataset
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
import argparse
from sentence_transformers import SentenceTransformer, util
import numpy as np

# NOTE: This script is based on TLQA_few_shot_ipynb, but adapted to run using GPU

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
    
    knn = KnnSearch()
    
    fewshot_eval(K=args.k, model_name=args.model_name, retriever_name=args.retriever_name, test_data=test_set, train_data=train_set, train_emb=train_questions_emb, infobox_data=args.infobox_data)

if __name__ == "__main__":
    main()
'''
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

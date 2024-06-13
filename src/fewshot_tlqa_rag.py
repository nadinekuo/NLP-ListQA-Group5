# 4
import os
from knn import KnnSearch
from utils import json_to_list
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from datasets import Dataset
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from sentence_transformers import SentenceTransformer, util
import mwxml
import mwparserfromhell
import json
import bz2
import argparse
import re
from datetime import datetime
import statistics
import numpy as np
import heapq

KNN_SEARCH = KnnSearch()


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


def extract_years_and_convert_to_datetime(sentence):
    year_pattern = r'\b\d{4}\b'
    years = re.findall(year_pattern, sentence)
    timestamps = []
    for year in years:
        date_string = f"January 1, {year}"  # Assuming January 1 for simplicity
        date_object = datetime.strptime(date_string, "%B %d, %Y")
        timestamps.append(date_object.timestamp())

    return datetime.fromtimestamp(np.mean(np.array(timestamps)))


def temporal_score(question_date: datetime, infobox_date: datetime):
    alpha = 1e+7  # This parameter was manually tuned to have at least some impact on our model
    if question_date < infobox_date:
        return -1e+10
    return alpha / ((question_date.timestamp() - infobox_date.timestamp()) + 1e-10)


def mean_std_temporal(all_test_question_dates, all_infoboxes_dates):
    temporal_scores = []
    for question_date in all_test_question_dates:
        for infobox_date in all_infoboxes_dates:
            if question_date >= infobox_date:
                temporal_scores.append(temporal_score(question_date, infobox_date))

    scores = np.array(temporal_scores)
    return np.mean(scores), np.std(scores)


def mean_std_semantic(all_test_questions, all_infoboxes_text):
    semantic_scores = []
    for question_emb in all_test_questions:
        for infobox_emb in all_infoboxes_text:
            score = util.cos_sim(question_emb, infobox_emb).tolist()[0][0]
            semantic_scores.append(score)

    scores = np.array(semantic_scores)
    return np.mean(scores), np.std(scores)


# Few-shot Evaluation with Context Retrieval
def fewshot_eval_with_context(K, model_name, test_data, train_data, train_emb, infoboxes, retriever, is_temporal_enabled=False):
    MAX_OUTPUT_LEN = 200
    MAX_SEQUENCE_LENGTH = 512  # Model's max sequence length

    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_name, torch_dtype=torch.float16)

    results_GT_dict = {'prompts': [], 'outputs': [], 'output_tokens': [],
                       'ground_truths': [], 'ground_truth_tokens': []}

    # Configure formatter that will format the few-shot examples into a string
    example_prompt = PromptTemplate(
        input_variables=["question", "answers"], template="Question: {question}\n{answers}"
    )

    infobox_texts = [infobox['infobox'] for infobox in infoboxes]
    all_test_questions = {test_d['question']: retriever.encode(test_d['question'], convert_to_tensor=True) for test_d in test_data}
    infobox_embeddings = retriever.encode(infobox_texts, convert_to_tensor=True)

    if is_temporal_enabled:
        all_test_question_dates = [extract_years_and_convert_to_datetime(test_d['question']) for test_d in test_data]
        all_infoboxes_dates = [infobox['mean_date'] for infobox in infoboxes]
        temporal_mean, temporal_std = mean_std_temporal(all_test_question_dates, all_infoboxes_dates)
        semantic_mean, semantic_std = mean_std_semantic(all_test_questions.values(), infobox_embeddings)

    # Convert test set to list and loop over all items
    for i, item in enumerate(test_data):
        # For each test question, retrieve k neighbours
        test_question = test_data[i]['question']
        test_question_date = extract_years_and_convert_to_datetime(test_question)

        print(f"Test question {i}: {test_question} of date {test_question_date}")

        # Retrieve k-nearest neighbors from training data
        neighs = KNN_SEARCH.get_top_n_neighbours(sentence=test_question, data_emb=train_emb, transfer_data=train_data,
                                                 k=K)
        simple_neighs = simplify_dict_list(neighs)
        # Retrieve top-1 relevant context from infoboxes
        query_embedding = all_test_questions[test_question]

        if is_temporal_enabled:
            hits = util.semantic_search(query_embedding, infobox_embeddings, top_k=len(infobox_embeddings))[0]
            score_by_infobox_id = {}
            for hit in hits:
                infobox_id = int(hit['corpus_id'])
                semantic_sc = hit['score']
                temporal_sc = temporal_score(test_question_date, infoboxes[infobox_id]['mean_date'])
                temporal_sc = ((temporal_sc - temporal_mean) / temporal_std) * semantic_std + semantic_mean
                score_by_infobox_id[infobox_id] = temporal_sc + semantic_sc

            top_k = heapq.nlargest(K, score_by_infobox_id.items(), key=lambda item: item[1])
            top_k_infobox_ids = [key for key, value in top_k]
        else:
            hits = util.semantic_search(query_embedding, infobox_embeddings, top_k=K)[0]
            top_k_infobox_ids = [hit['corpus_id'] for hit in hits]

        print(top_k_infobox_ids)
        top_infobox_id = top_k_infobox_ids[0]
        top_infobox = infoboxes[top_infobox_id]['infobox']

        # Truncate the context to fit within the sequence length limit
        top_infobox = top_infobox[:MAX_SEQUENCE_LENGTH // 2]  # Adjust the truncation as needed

        # Create the few-shot prompt template and feed to model
        prompt = FewShotPromptTemplate(
            examples=simple_neighs,  # No. of few shot examples is defined by sysarg K
            example_prompt=example_prompt,
            suffix="Question: {input}",
            input_variables=["input"],
        )
        few_shot_prompt = prompt.format(input=f"{test_question}\nPlease answer this question in the same format as the {K} examples above.\n\n\
        Use the following context to answer the question at the end (do not use this structure however). \
        If you can't find the relevant information in the context, just say you don't have enough information to answer the question. \
        Don't try to make up an answer.\n\n{top_infobox}")

        # Print the prompt to see how it looks
        # print(f"Few-shot Prompt for Test Question {i}:\n{few_shot_prompt}\n")
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


def convert_to_datetime(date_str):
    # Try to convert date_str to a datetime object with multiple formats
    for fmt in ('%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y'):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Date format for {date_str} not recognized")


def extract_infoboxes(dump_file, output_file):
    dump = mwxml.Dump.from_file(bz2.open(dump_file))
    infoboxes = []

    for page in dump:
        if page.namespace != 0:  # Only process articles in the main namespace
            continue
        if page.redirect:  # Skip redirects
            continue
        for revision in page:
            if revision.text is None:
                continue
            wikicode = mwparserfromhell.parse(revision.text)
            templates = wikicode.filter_templates()
            for template in templates:
                if template.name.lower().startswith("infobox"):
                    infobox = {"title": page.title, "infobox": str(template)}
                    infoboxes.append(infobox)
            break  # Process only the latest revision

    with open(output_file, 'w') as f:
        json.dump(infoboxes, f, indent=2)


def parse_infobox(infobox_wikitext):
    wikicode = mwparserfromhell.parse(infobox_wikitext)
    templates = wikicode.filter_templates()

    infobox_data = {}
    dates = []
    date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4}|\d{2}/\d{2}/\d{4}')

    for template in templates:
        template_name = str(template.name).strip()
        fields = {}
        for param in template.params:
            param_name = str(param.name).strip()
            param_value = str(param.value).strip()
            fields[param_name] = param_value

            # Find all dates in the parameter value
            dates.extend(date_pattern.findall(param_value))

        infobox_data[template_name] = fields

    return infobox_data, dates


def calculate_mean_date(dates):
    datetime_objects = [convert_to_datetime(date) for date in dates]
    mean_timestamp = statistics.mean(dt.timestamp() for dt in datetime_objects)
    mean_datetime = datetime.fromtimestamp(mean_timestamp)
    return mean_datetime


def extract_infoboxes_from_file(input_file):
    with open(input_file, 'r') as f:
        infoboxes = json.load(f)

    parsed_infoboxes = []
    all_dates = []
    for infobox in infoboxes:
        parsed_infobox, dates = parse_infobox(infobox['infobox'])
        mean_date = calculate_mean_date(dates) if dates else None
        parsed_infoboxes.append({
            'title': infobox['title'],
            'parsed_info_box': parsed_infobox,
            'infobox': infobox['infobox'],
            'is_global_mean': False if mean_date else True,
            'mean_date': mean_date if mean_date else None  # string format
        })
        if mean_date:
            all_dates.append(mean_date)

    return parsed_infoboxes, all_dates


def parse_infoboxes_from_file(input_file):
    parsed_infoboxes, all_dates = extract_infoboxes_from_file(input_file)
    mean_date = datetime.fromtimestamp(statistics.mean(dt.timestamp() for dt in all_dates))
    for infobox in parsed_infoboxes:
        if infobox['mean_date'] is None:
            infobox['mean_date'] = mean_date

    return parsed_infoboxes


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    k = args.k
    model = args.model_name

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
    train_questions = get_transfer_questions(train_set)  # Keep questions only to embed (to use in similarity metric)
    train_questions_emb = KNN_SEARCH.get_embeddings_for_data(train_questions)

    # Load infoboxes
    infoboxes = parse_infoboxes_from_file(infoboxes_file_path)


    # Initialize retriever model
    retriever = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-v4')

    print(f"\n\nStarting {k}-shot evaluation on {model} with context retrieval...\n\n")
    fewshot_eval_with_context(K=k, model_name=model, test_data=test_set, train_data=train_set,
                              train_emb=train_questions_emb, infoboxes=infoboxes, retriever=retriever, is_temporal_enabled=True)

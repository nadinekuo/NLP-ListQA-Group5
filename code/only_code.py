import json
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate


class KnnSearch:
    def __init__(self, data=None, num_trees=None, emb_dim=None):
        self.num_trees = num_trees
        self.emb_dim = emb_dim

    def get_embeddings_for_data(self, data_ls):
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        embeddings = model.encode(data_ls)
        return embeddings

    def get_top_n_neighbours(self, sentence, data_emb, transfer_data, k):
        """
        Retrieves the top k most similar questions for "sentence" based on cosine similarity from given embeddings "data_emb".

        Parameters:
        sentence (str): The input sentence to find similar questions for.
        data_emb (np.ndarray): The embeddings for the transfer questions.
        transfer_data (list): The list of transfer questions corresponding to data_emb.
        k (int): The number of top similar questions to retrieve.

        Returns:
        list: A list of the top k similar questions from transfer_data and all similar questions from str_qa.
        """
        sent_emb = self.get_embeddings_for_data(sentence)
        top_questions = []

        text_sims = cosine_similarity(data_emb, [sent_emb]).tolist()
        results_sims = zip(range(len(text_sims)), text_sims)
        sorted_similarities = sorted(results_sims, key=lambda x: x[1], reverse=True)  # Obtain the highest similarities

        # NOTE: we only match based on questions, but include the full question-answer pair in resulting neighs
        for idx, item in sorted_similarities[:k]:
            top_questions.append(transfer_data[idx])

        return top_questions


def get_transfer_questions(transfer_data):
    transfer_questions = []
    for index, data in enumerate(transfer_data):
        transfer_questions.append(data["question"])
    return transfer_questions


def json_to_list(data_path):
    with open(data_path) as f:
        data = json.load(f)
    return data


def simplify_dict_list(dict_list):
    return [{'question': item['question'], 'answers': item['answers']} for item in dict_list]


TOKENIZER = T5Tokenizer.from_pretrained("google/flan-t5-large", torch_dtype=torch.float16)
MODEL = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
input_text = "List all Michael Jackson albums between 2000 and 2009."

input_ids = TOKENIZER(input_text, return_tensors="pt").input_ids
outputs = MODEL.generate(input_ids)

knn = KnnSearch()
train_data = json_to_list("../data/train_TLQA.json")
train_questions = get_transfer_questions(train_data)
train_questions_emb = knn.get_embeddings_for_data(train_questions)

question = "List all positions held by Mark Zuckerberg from 2000 to 2024."
neighs = knn.get_top_n_neighbours(sentence=question, data_emb=train_questions_emb, transfer_data=train_data, k=3)

simple_neighs = simplify_dict_list(neighs)
print(simple_neighs)

example_prompt = PromptTemplate(input_variables=["question", "answers"], template="Question: {question}\n{answers}")

print(example_prompt.format(**simple_neighs[1]))

prompt = FewShotPromptTemplate(
    examples=simple_neighs,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"],
)

few_shot_prompt = prompt.format(
    input="List all positions held by Mark Zuckerberg between 2000 and 2020. Please answer this question in the same format as the three examples above.")
print(few_shot_prompt)

K = 10
prompts_results_GT_dict = {'prompts': [], 'outputs': [], 'ground_truths': []}

example_prompt = PromptTemplate(
    input_variables=["question", "answers"], template="Question: {question}\n{answers}"
)


test_data = json_to_list("../data/test_TLQA.json")
for i, item in enumerate(test_data):
    test_question = test_data[i]['question']
    neighs = knn.get_top_n_neighbours(sentence=test_question, data_emb=train_questions_emb, transfer_data=train_data, k=K)
    simple_neighs = simplify_dict_list(neighs)
    prompt = FewShotPromptTemplate(
        examples=simple_neighs,
        example_prompt=example_prompt,
        suffix="Question: {input}",
        input_variables=["input"],
    )
    few_shot_prompt = prompt.format(
        input=f"{test_question} Please answer this question in the same format as the {K} examples above.")
    prompts_results_GT_dict['prompts'].append(few_shot_prompt)

    input_ids = TOKENIZER(few_shot_prompt, return_tensors="pt").input_ids
    outputs = MODEL.generate(input_ids, max_length=200)
    model_answer = TOKENIZER.decode(outputs[0])
    prompts_results_GT_dict['outputs'].append(model_answer)
    prompts_results_GT_dict['ground_truths'].append(test_data[i]['final_answers'])

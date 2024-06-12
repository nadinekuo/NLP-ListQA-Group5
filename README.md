# Timeline based List-Question Answering
Project for the course Natural Language Processing (CS4360) at Delft University of Technology


## Introduction

Question Answering (QA) systems play an important role in meeting users' precise information needs by directly providing answers with supporting evidence rather than presenting a list of snippets or hyperlinks. This improves user experience significantly in various various domains such as healthcare, education, and more, where accurate and immediate information is essential. While QA systems are good at handling a wide range of queries, generating answers in forms from single words to detailed abstracts, there is still a noticeable gap in effectively addressing questions that require structured, list-based answers, especially those involving temporal elements.

## Our Project: Research Questions

We aim to study the abilities of language models on the TLQA task, beyond traditional QA where answers are of free-form type or extractive or multiple choice types. TLQA has constraints that the system should generate the complete list of answers while also ensuring the time ranges are accurate. 

Specifically, this repository contains our experiment setup to answer the following questions:

1. **How do finetuned generative models and few-shot prompting of generative models in a closed book QA setting perform on the task of TLQA?**

2. **How do finetuned generative models and few-shot prompting of generative models with retrieved top-k evidence perform on the task of TLQA?**

3. **Does special handling of temporal markers improve performance of TLQA?**

## Getting Started

The following scripts form the backbone to our various experiments:

- `fewshot_tlqa.py`: closed book few-shot setting 
- `fewshot_tlqa_rag.py`: RAG in few-shot setting
- `eval_results.py`: evaluation of model performance based on various metrics - EM, F1, Recall, BLEU (on time ranges only) and BERT-score (on entities only)
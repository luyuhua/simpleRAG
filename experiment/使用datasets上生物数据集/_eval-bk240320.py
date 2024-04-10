from datasets import load_dataset
from langchain_community.chat_models import ChatOllama
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas import evaluate
import os 


# rag_dataset = load_dataset("neural-bridge/rag-dataset-12000")
malayalam_dataset = load_dataset("explodinggradients/amnesty_qa","malayalam")


model_name = "/home/lu/.cache/torch/sentence_transformers/BAAI_bge-small-zh-v1.5"
# model_kwargs = {}
model_kwargs = {'device': 'cuda:0'}
# encode_kwargs = {'normalize_embeddings': False}
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            cache_folder = "/home/lu/.cache/torch/sentence_transformers")

generator_llm = ChatOllama(model="qwen:14b")

result = evaluate(
    malayalam_dataset["eval"],
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ],
    raise_exceptions=False,
    llm = generator_llm,
    embeddings=embeddings
)

df = result.to_pandas()
print(df)
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context,conditional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
import os 


os.environ["OPENAI_API_KEY"] = "sk-YzvJyUYM1qBzVkbu0rAuLhiYT9tjYmDNz5vYlJRvzRCVC8Mz"
os.environ["OPENAI_BASE_URL"] = "https://api.chatanywhere.tech/v1"


loader = DirectoryLoader("docs")
documents = loader.load()
for document in documents:
    document.metadata['filename'] = document.metadata['source']

print('doc load finished')

# embedding model
# model_name = "infgrad/stella-base-zh-v3-1792d"
# model_name = "BAAI/bge-small-zh-v1.5"
# model_name = "BAAI/bge-small-en-v1.5"
model_name = "/home/lu/.cache/torch/sentence_transformers/BAAI_bge-small-en-v1.5"

# model_kwargs = {}
model_kwargs = {'device': 'cuda:0'}
# encode_kwargs = {'normalize_embeddings': False}
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            cache_folder = "/home/lu/.cache/torch/sentence_transformers")
print('init embedding finished')

# generator with openai models
# generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
# critic_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")

generator_llm = ChatOllama(model="qwen:14b")
critic_llm = ChatOllama(model="qwen:14b")
# embeddings = OpenAIEmbeddings()


generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embeddings
)

# # adapt to language
# language = "chinese"

# generator.adapt(language, evolutions=[simple, reasoning,conditional,multi_context])
# generator.save(evolutions=[simple, reasoning, multi_context,conditional])

# generate testset
testset = generator.generate_with_langchain_docs(documents, test_size=100, distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25},raise_exceptions=False)
print(testset)
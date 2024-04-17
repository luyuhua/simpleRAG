
from llama_index.readers.file import MarkdownReader,PyMuPDFReader
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core import SimpleDirectoryReader
from pathlib import Path
import os


class Loader:
    def __init__(self) -> None:
        pass
    
    def load_data(self,docs):
        documents = []
        for doc in docs:
            doc = doc if isinstance(doc,Path) else Path(doc)
            if doc.suffix.lower() == '.pdf':
                reader = PyMuPDFReader()
            elif doc.suffix.lower() == '.md':
                reader = MarkdownReader()
                docs = reader.load_data(doc,extra_info={'file_name':doc.name})
                parser = MarkdownNodeParser()
                docs = parser.get_nodes_from_documents(docs)
                documents.extend(docs)
                continue
            else:
                reader = SimpleDirectoryReader(input_files=docs,file_metadata=lambda x:{"file_name":os.path.basename(x)})
                documents.extend(reader.load_data())
                continue
            documents.extend(reader.load_data(doc,extra_info={'file_name':doc.name}))
        return documents        



if __name__ == '__main__':
    loader = Loader()

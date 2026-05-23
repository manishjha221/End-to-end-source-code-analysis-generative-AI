from git import Repo
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain

def repo_ingestion(repo_url):
     os.makedirs("repo", exist_ok=True)
     repo = "repo/"
     Repo.clone_from(repo_url, to_path=repo)

def load_repo(repo_path):
    loader = GenericLoader.from_filesystem(repo_path, 
                                           glob="**/*",
                                           suffixes=[".py"],
                                            parser=LanguageParser(language=Language.PYTHON)
                                            )
    documents = loader.load()
    return documents


def text_splitter(documents):
  documents_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, 
                                                                    chunk_size=500, chunk_overlap=20)
  texts = documents_splitter.split_documents(documents)
  return texts


def load_embedding():
    embeddings = OpenAIEmbeddings(disallow_special=())
    return embeddings
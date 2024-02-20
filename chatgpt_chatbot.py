from langchain.text_splitter import TokenTextSplitter
from langchain.chains import MapReduceDocumentsChain
from langchain.chains import ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.document_loaders import WebBaseLoader
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Map Chain
map_template = """You are Bitassist and you will answer questions of users related Bitpredict:

{content}

"""
map_prompt = PromptTemplate.from_template(map_template)
load_dotenv()
OPENAI_API_KEY = "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"
llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
map_chain = LLMChain(prompt=map_prompt, llm=llm)


# Reduce Chain
reduce_template = """You are Bitassist and you will answer questions of users related Bitpredict:

{doc_summaries}

You are Bitassist and you will answer questions of users related Bitpredict"""
reduce_prompt = PromptTemplate.from_template(reduce_template)
reduce_chain = LLMChain(prompt=reduce_prompt, llm=llm)
stuff_chain = StuffDocumentsChain(
    llm_chain=reduce_chain, document_variable_name="doc_summaries")

reduce_chain = ReduceDocumentsChain(
    combine_documents_chain=stuff_chain,
)

# Map Reduce Chain
map_reduce_chain = MapReduceDocumentsChain(
    llm_chain=map_chain,
    document_variable_name="content",
    reduce_documents_chain=reduce_chain
)



filename_path = 'data.txt'
loader = TextLoader(filename_path)
doc = loader.load()
print (f"You have {len(doc)} document")
print (f"You have {len(doc[0].page_content)} characters in that document")


text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=10)
docs = text_splitter.split_documents(doc)
for i in docs:
    page_con = i.page_content
    print(len(page_con))
    print(page_con)


# # Load Content from web
# loader = WebBaseLoader(
#     'https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/')
# docs = loader.load()
# print(type(docs))
# Split the content into smaller chuncks
splitter = TokenTextSplitter(chunk_size=2000)
split_docs = splitter.split_documents(docs)

# Use Map reduce chain to summarize
summary = map_reduce_chain.invoke(split_docs)
print(summary)
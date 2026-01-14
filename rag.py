from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import GigaChat
from settings.config import GIGACHAT_TOKEN
from settings.prompts import generator_prompt, critic_prompt, qa_prompt

embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
CREDENTIALS = GIGACHAT_TOKEN

QA_PROMPT = PromptTemplate.from_template(qa_prompt)
GENERATOR_PROMPT = PromptTemplate.from_template(generator_prompt)
CRITIC_PROMPT = PromptTemplate.from_template(critic_prompt)


def get_generator_llm():
    return GigaChat(credentials=CREDENTIALS, model="GigaChat-Pro", temperature=0.7, verify_ssl_certs=False, timeout=120)

def get_critic_llm():
    return GigaChat(credentials=CREDENTIALS, model="GigaChat-Max", temperature=0.2, verify_ssl_certs=False, timeout=120)

def get_qa_llm():
    return GigaChat(credentials=CREDENTIALS, model="GigaChat-Pro", temperature=0.4, verify_ssl_certs=False, timeout=120)


def ask(question: str):
    docs = vectorstore.similarity_search(question, k=5)
    context = "\n\n".join([f"Источник: {d.metadata.get('title','?')}\n{d.page_content}" for d in docs])
    llm = get_qa_llm()
    response = (QA_PROMPT | llm).invoke({"context": context, "question": question})
    return response.content


def generate_hypotheses(problem: str):
    docs = vectorstore.similarity_search(problem, k=10)
    context = "\n\n".join([f"[{i+1}] {d.metadata.get('title','?')}\n{d.page_content}" for i, d in enumerate(docs)])
    
    raw_hypotheses = (GENERATOR_PROMPT | get_generator_llm()).invoke({
        "problem": problem,
        "context": context
    }).content
    
    final_hypotheses = (CRITIC_PROMPT | get_critic_llm()).invoke({
        "raw_hypotheses": raw_hypotheses,
        "context": context
    }).content
    
    return final_hypotheses, raw_hypotheses, docs
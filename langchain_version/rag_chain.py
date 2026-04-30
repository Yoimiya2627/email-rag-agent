"""
LangChain-based RAG chain — parallel implementation alongside the custom pipeline.

Uses:
  - langchain_openai.ChatOpenAI → DeepSeek API
  - langchain_community.vectorstores.Chroma → same ChromaDB collection
  - langchain.chains.RetrievalQA with ConversationalRetrievalChain
  - ConversationBufferWindowMemory for multi-turn context

How to run:
  python langchain_version/rag_chain.py

Prerequisites:
  pip install langchain langchain-openai langchain-community
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config.settings as cfg
from core.embedder import _get_embedding_fn  # reuse same embedding model

# Lazy imports to avoid mandatory dependency on langchain
try:
    from langchain_openai import ChatOpenAI
    from langchain_community.vectorstores import Chroma
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.chains import ConversationalRetrievalChain
    from langchain.prompts import PromptTemplate
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False


_SYSTEM_TEMPLATE = """你是一个专业的邮件智能助手。请根据以下检索到的邮件内容，简洁、准确地回答用户的问题。
如果提供的内容不足以回答，请明确说明。引用邮件时请注明发件人和日期。

检索到的邮件内容：
{context}

对话历史：
{chat_history}

用户问题：{question}
回答："""


def build_chain(k: int = 5, window: int = 5):
    """
    Build and return a ConversationalRetrievalChain using the same ChromaDB collection.

    Args:
        k: number of documents to retrieve
        window: conversation window size (turns)
    """
    if not _HAS_LANGCHAIN:
        raise ImportError(
            "LangChain not installed. Run: pip install langchain langchain-openai langchain-community"
        )

    llm = ChatOpenAI(
        model=cfg.DEEPSEEK_MODEL,
        openai_api_key=cfg.DEEPSEEK_API_KEY,
        openai_api_base=cfg.DEEPSEEK_BASE_URL,
        temperature=0.3,
        max_tokens=1500,
        request_timeout=cfg.LLM_TIMEOUT,
    )

    embedding_fn = _get_embedding_fn()
    vectorstore = Chroma(
        collection_name=cfg.CHROMA_COLLECTION,
        embedding_function=embedding_fn,
        persist_directory=cfg.CHROMA_PERSIST_DIR,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
        k=window,
    )

    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=_SYSTEM_TEMPLATE,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        verbose=False,
    )
    return chain


def ask(chain, question: str) -> dict:
    """Run a question through the chain and return answer + sources.

    Modern langchain (>=0.2) replaces Chain.__call__/__call with .invoke().
    """
    result = chain.invoke({"question": question})
    sources = [
        {
            "content": doc.page_content[:300],
            "metadata": doc.metadata,
        }
        for doc in result.get("source_documents", [])
    ]
    return {"answer": result["answer"], "sources": sources}


def interactive_demo():
    """Simple interactive CLI demo."""
    print("Building LangChain RAG chain...")
    chain = build_chain()
    print("Ready! Type 'quit' to exit.\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("quit", "exit", "q"):
            break
        result = ask(chain, q)
        print(f"\nAssistant: {result['answer']}")
        if result["sources"]:
            print(f"\n[来源: {result['sources'][0]['metadata'].get('subject','?')}]")
        print()


if __name__ == "__main__":
    interactive_demo()

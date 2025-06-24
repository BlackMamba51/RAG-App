import os, json, logging
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize 
from tqdm import tqdm
from transformers import AutoTokenizer 
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

def delete_empty_json_files(folder):
    deleted = 0
    for root, _, files in os.walk(folder):
        for filename in files:
            if filename.endswith('.json'):
                path = os.path.join(root, filename)
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if not content.strip():
                    os.remove(path)
                    deleted += 1                                                                                    
                    print(path)
    print(f'Удалено - {deleted}')

# delete_empty_json_files('D:\RAG Data')

def load_json_files(json_file):
    docs = []
    with open(json_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                docs.append(data)
    return docs
json_data = load_json_files('D:\RAG Data\enriched_objects.json') 

model_name = 'sentence-transformers/all-MiniLM-L12-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)


# def chunk_text_by_tokens(text, max_tokens=512, overlap=50):
    # sentences = sent_tokenize(text)
    # chunks, current_chunk = [], []
    # current_length = 0

    # for sentence in sentences:
    #     sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
    #     token_len = len(sentence_tokens)

    #     if token_len > max_tokens:
    #         for i in range(0, token_len, max_tokens):
    #             part_tokens = sentence_tokens[i:i+max_tokens]
    #             part_text = tokenizer.decode(part_tokens)
    #             chunks.append(part_text.strip())
    #         current_chunk = []
    #         current_length = 0
    #         continue

    #     if current_length + token_len > max_tokens:
    #         chunks.append(' '.join(current_chunk))
    #         current_chunk = [sentence]
    #         current_length = token_len
    #     else:
    #         current_chunk.append(sentence)
    #         current_length += token_len    

    # if current_chunk:
    #     chunks.append(' '. join(current_chunk))

    # final_chunks = []
    # for i, chunk in enumerate(chunks):
    #     if i > 0:
    #         overlap_text = chunks[i-1].split()[-overlap:]
    #         combined = ' '.join(overlap_text + chunk.split())
    #     else:
    #         combined = chunk
    #     final_chunks.append(combined.strip())
    # return final_chunks

# def write_chunks_json(data, path):
#     with open(path, 'w', encoding='utf-8') as f:
#         for chunk in data:
#             json.dump(chunk, f, ensure_ascii=False)
#             f.write('\n')
#     print(f"✅ Сохранено {len(data)} чанков в {path}")

# all_chunks = []
# for record in tqdm(json_data, desc="Chunking records"):
#     if not isinstance(record, dict):
#         print(f"❗ Пропущен несловарь: {type(record)} → {record}")
#         continue
#     text = json.dumps(record, ensure_ascii=False)
#     json_chunks = chunk_text_by_tokens(text)

#     for i, chunk in enumerate(json_chunks):
#         all_chunks.append({
#             'id': record.get('id'),
#             'chunk_index': i,
#             'chunk_text': chunk
#         })
    
# write_chunks_json(all_chunks, 'D:\RAG Data\chunks.json')


model = SentenceTransformer(model_name)
model.to('cuda')
def load_chunks(folder):
    with open(folder, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

# chunks_data = load_chunks('D:\RAG Data\chunks.json')
def clean_chunks(chunks):
    seen = set()
    clean = []
    for chunk in chunks:
        uid = f'{chunk['id']}_{chunk['chunk_index']}'
        if uid not in seen:
            clean.append(chunk)
            seen.add(uid)
    return clean
# clean_chunks_data = clean_chunks(chunks_data)

client = chromadb.PersistentClient(path="D:/RAG Data/chroma_index")

collection = client.get_or_create_collection('rag_chunks')
# texts = [chunk['chunk_text'] for chunk in clean_chunks_data]
# metadatas = [{'id': chunk['id'], 'chunk_index': chunk['chunk_index']} for chunk in clean_chunks_data]
# ids = [f'{chunk['id']}_{chunk['chunk_index']}' for chunk in clean_chunks_data]
# embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

def add_in_chunks(database, docs, embs, metas, ids, max_batch_size=5000):
    total = len(ids)
    for i in range(0, total, max_batch_size):
        batch_docs = docs[i:i+max_batch_size]
        batch_embs = embs[i:i+max_batch_size]
        batch_metas = metas[i:i+max_batch_size]
        batch_ids = ids[i:i+max_batch_size]
        database.add(documents=batch_docs, embeddings=batch_embs, metadatas=batch_metas, ids=batch_ids)

# add_in_chunks(collection, texts, embeddings, metadatas, ids)


# def search_documents(query, embedder, database, top_k=5):
#     query_emb = embedder.encode(query, normalize_embeddings=True)
#     result = database.query(
#         query_embeddings = query_emb,
#         n_results = top_k,
#         include = ['documents', 'metadatas']
#     )
#     return result['documents'][0], result['metadatas'][0]
# docs = search_documents('Find portraits by female artists', model, collection)

logging.basicConfig(level=logging.INFO)

llm = ChatOllama(
    model='llama3.2:3b',
    temperature=0,
    num_predict=150
)
messages = [
    ('system', "You must answer only based on the context and list every artwork in the context that matches the question. Do not use prior knowledge. If uncertain, you may say 'I don't know'."),
    ('human', 'Context:\n{context}\n\nQuestion: {question}')
]
prompt_template = ChatPromptTemplate.from_messages(messages)
# prompt = PromptTemplate(
#     template="""
#     Answer the questions based on the text below.
#     If you cannot answer the question using the provided information answer with "I don't know".
#     context: {context}
#     Question: {question}
#     Answer: 
#     """,
#     input_variables=['context', 'question']
# )

def simple_format(doc):
    meta = doc.metadata or {}
    content = doc.page_content or ""
    try: 
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {}
    lines = []
    print(data)
    artist = data.get("artist") or meta.get("artist")
    text = data.get("text") or meta.get("text")
    title = data.get("title") or meta.get("title")
    dated = data.get("dated") or meta.get("dated")
    classification = data.get("classification") or meta.get("classification")
    department = data.get("department") or meta.get("department")
    continent = data.get("continent") or meta.get("continent")
    country = data.get("country") or meta.get("country")
    creditline = data.get("creditline") or meta.get("creditline")
    object_name = data.get("object_name") or meta.get("object_name")
    medium = data.get("medium") or meta.get("medium")
    res_context = f'"{title}" is a {dated} {classification} by {artist} created in {continent}, {country}. This work belongs to "{department}" collection and made from {medium}'
    # if title: lines.append(f'{title}')
    # if artist: lines.append(f'by {artist}')
    # if dated: lines.append(f'({dated})')
    # if classification: lines.append(classification)
    # if department: lines.append(department)
    # if continent: lines.append(continent)
    # if country: lines.append(country)
    # # if creditline: lines.append(creditline)
    # # if object_name: lines.append(object_name)
    # if medium: lines.append(f'made of {medium}')
    # if text: lines.append(text)

    return res_context

def build_simple_context(docs, max_chunks=10):
    chunks = []
    for i, doc in enumerate(docs[:max_chunks]):
        chunk = simple_format(doc)
        print(f"\n[DEBUG] chunk {i+1}:\n{chunk}\n{'-'*40}")
        chunks.append(chunk)
    return "\n".join(chunks)



embedder = HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2')

vectorstore = Chroma(
    persist_directory="D:/RAG Data/chroma_index",
    collection_name="rag_chunks",
    embedding_function=embedder
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

llm_chain = prompt_template | llm | StrOutputParser()

def rag_pipeline(question, retriever):
    steps = []
    def get_logs(text):
        logging.info(text)
        steps.append(text)

    get_logs(f'Question recieved')
    

    try:
        docs = retriever.invoke(question)
        context = build_simple_context(docs)
        get_logs(f'Find {len(docs)} Documents')
        
        
        result = llm_chain.invoke({
            'context': context,
            'question': question
        })
        
        get_logs("Answer generated.")
        return {'answer': result,'context': context, 'sources': [doc.metadata for doc in docs], 'trace': steps}
    except Exception as e:
        get_logs(f"Ошибка в RAG-конвейере: {e}")
        return {
            "answer": "Не удалось получить ответ.",
            "sources": []
        }
flag = True    
# while flag:
#     user_question = input('Your question:')
#     # prompt_value = prompt_template.invoke({'context': rag_pipeline(user_question, retriever), 'question': user_question})
#     response = rag_pipeline(user_question, retriever)
#     # print(response)
#     # print("🧠 Ответ:\n", response["answer"])
#     # print("\n📚 Источники:")
#     # for meta in response["sources"]:
#     #     print("-", meta.get("title") or meta.get("id") or "Без названия")
#     if user_question in ['exit', 'ex', 'back']:
#         flag = False
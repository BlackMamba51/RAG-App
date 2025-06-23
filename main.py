import os, json
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize 
from tqdm import tqdm
from transformers import AutoTokenizer 
from sentence_transformers import SentenceTransformer
import chromadb
from collections import Counter

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


def search_documents(query, embedder, database, top_k=5):
    query_emb = embedder.encode(query, normalize_embeddings=True)
    result = database.query(
        query_embeddings = query_emb,
        n_results = top_k,
        include = ['documents', 'metadatas']
    )
    return result['documents'][0], result['metadatas'][0]
docs = search_documents('Find portraits by female artists', model, collection)
for d in docs:
    print(d)
    print('-' * 50)
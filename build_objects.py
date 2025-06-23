import os, json
from tqdm import tqdm
DATA_DIR = 'D:\RAG Data'

OBJ_DIR  = os.path.join(DATA_DIR, 'objects')
OUTPUT_FILE = os.path.join(DATA_DIR, 'enriched_objects.json')
def build_enriched_objects(obj_dir, output_file):
    
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for root, _,  files in os.walk(obj_dir):
            for filename in tqdm(files, desc='Object processing'):
                if filename.endswith('.json'):
                    with open(os.path.join(root, filename), 'r', encoding='utf-8') as f:
                        print(filename)
                        data = json.load(f)
                        out_file.write(json.dumps(data, ensure_ascii=False) + '\n')
build_enriched_objects(OBJ_DIR, OUTPUT_FILE)

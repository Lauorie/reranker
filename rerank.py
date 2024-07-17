import os
import json
import torch
from tqdm import tqdm
from sentence_transformers import CrossEncoder
from typing import List, Dict, Union, Literal
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

bge = '/root/app/models/models--BAAI--bge-reranker-large'
jina = '/root/app/models/models--jinaai--jina-reranker-v2-base-multilingual'
model = CrossEncoder(
    bge,
    automodel_args={"torch_dtype": "auto"},
    trust_remote_code=True,
    max_length=512
)

# Read JSON data
with open('/root/app/data/data_cleaned_generated.json', 'r') as f:
    data = json.load(f)
data = data[:1024]

questions = [str(item['question']) for item in data]
ground_truths = [str(item['text']) for item in data]


hit_count = 0
total_questions = len(questions)

def rerank(query, documents, batch_size, top_k=5)-> List[Dict[Literal["corpus_id", "score", "text"], Union[int, float, str]]]:
    result = model.rank(
        query,
        documents,
        batch_size = batch_size,
        top_k=top_k,
        # return_documents=True, 
        convert_to_tensor=True,
        show_progress_bar=True
    )
    return result

# Example query and documents
# query = "Organic skincare products for sensitive skin"
# documents = [
#     "Organic skincare for sensitive skin with aloe vera and chamomile.",
#     "New makeup trends focus on bold colors and innovative techniques",
#     "Bio-Hautpflege für empfindliche Haut mit Aloe Vera und Kamille",
#     "Neue Make-up-Trends setzen auf kräftige Farben und innovative Techniken",
#     "Cuidado de la piel orgánico para piel sensible con aloe vera y manzanilla",
#     "Las nuevas tendencias de maquillaje se centran en colores vivos y técnicas innovadoras",
#     "针对敏感肌专门设计的天然有机护肤产品",
#     "新的化妆趋势注重鲜艳的颜色和创新的技巧",
#     "敏感肌のために特別に設計された天然有機スキンケア製品",
#     "新しいメイクのトレンドは鮮やかな色と革新的な技術に焦点を当てています",
# ]
# result = rerank(query, documents, 5, 5)
# print(result)


hit_count = 0
for i, question in tqdm(enumerate(questions)):
    result = rerank(question, ground_truths, 1024, 5)
    top5_ids = [item['corpus_id'] for item in result]
    if i in top5_ids:
        hit_count += 1

avaerage_hit_rate = hit_count / total_questions
print(f"Average hit rate: {avaerage_hit_rate}")

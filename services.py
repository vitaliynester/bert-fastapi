from transformers import AutoTokenizer, AutoModel
from sentence_transformers import util
import torch


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def compare(text1: str, text2: str):
    sentences = [text1, text2]

    tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_nlu_ru")
    model = AutoModel.from_pretrained("ai-forever/sbert_large_nlu_ru")

    encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=24, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    probability = util.pytorch_cos_sim(sentence_embeddings[0], sentence_embeddings[1])
    return list(probability[0].numpy())[0]

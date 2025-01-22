import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import os
from tqdm import tqdm

model_name = "../SGPT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

df = pd.read_csv("../data/dpr/psgs_w100_fixed.tsv", sep="\t", header=None)


def get_weightedmean_embedding(batch_tokens, model, device):
    batch_tokens = {key: val.to(device) for key, val in batch_tokens.items()}

    with torch.no_grad():
        last_hidden_state = model(**batch_tokens, output_hidden_states=True, return_dict=True).last_hidden_state

    weights = (
        torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
        .unsqueeze(0)
        .unsqueeze(-1)
        .expand(last_hidden_state.size())
        .float().to(device)
    )

    input_mask_expanded = (
        batch_tokens["attention_mask"]
        .unsqueeze(-1)
        .expand(last_hidden_state.size())
        .float()
    )

    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded * weights, dim=1)
    sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

    embeddings = sum_embeddings / sum_mask

    return embeddings


encode_file_path = "../SGPT/encode_result"
os.makedirs(encode_file_path, exist_ok=True)


for j, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Documents"):
    doc = str(row[1])
    batch_tokens = tokenizer(doc, return_tensors="pt", padding=True, truncation=True)
    embeddings = get_weightedmean_embedding(batch_tokens, model, device)
    filename = f"0_{j}.pt"
    torch.save(embeddings, os.path.join(encode_file_path, filename))  # 保存到指定目录

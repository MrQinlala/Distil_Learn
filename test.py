import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
# from datasets import Dataset
from tqdm import tqdm
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained('/mnt/workspace/.cache/modelscope/hub/ZhipuAI/chatglm3-6b',trust_remote_code=True).to(device)
# tokenizer = AutoTokenizer.from_pretrained('/mnt/workspace/.cache/modelscope/hub/ZhipuAI/chatglm3-6b',trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('/mnt/workspace/minimind/model/minimind_tokenizer')
model.eval()
data_path = './json_datasets/val.json'

with open(data_path, 'r') as f:
    test_data = json.load(f)
#数据预处理类
class CEVALDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.questions = [f"问题：{item['question']}\n选项：A. {item['A']} B. {item['B']} C. {item['C']} D. {item['D']}\n答案：" for item in data]
        self.answers = [item["answer"] for item in data]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.questions[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "answer": self.answers[idx]
        }

def evaluate_model(test_data):
    dataset = CEVALDataset(test_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    correct = 0

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask, 
                max_new_tokens=1,  # 只生成一个token（选项字母）
                pad_token_id=tokenizer.eos_token_id
            )
            predictions = tokenizer.batch_decode(outputs[:, -1], skip_special_tokens=True)
            correct += sum([pred.upper() == ans for pred, ans in zip(predictions, batch["answer"])])

    accuracy = correct / len(test_data)
    return accuracy

if __name__ == "__main__":
    accuracy = evaluate_model(test_data)
    print(f"模型在CEVAL数据集上的准确率：{accuracy:.2%}")
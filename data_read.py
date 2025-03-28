import json
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.data import Dataset,DataLoader

tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-4_H-768_A-12")
with open("/mnt/workspace/kl/hand_write/json_datasets/val.json","r") as f:
    test_data = json.load(f)  # 用户提供的测试集数据
#这里的test_data是一个list，每个元素是一个字典，字典的key是"question"和"answer"，分别表示问题内容和答案
#且还没有经过test_dataset = CEVALDataset(test_data, tokenizer)处理



# test_loader = DataLoader(test_data,batch_size=8)
# #遍历到前batch个字典数据中
# for batch in test_loader:
# #遍历到具体每batch字典中的k,v中
#     for k,v in batch.items():
#         print(k)


#item就是每个json数据

# class CEVALDataset(Dataset):
    # def __init__(self, data, tokenizer, max_length=512):
    #     self.questions = [f"问题：{item['question']}\n选项：A. {item['A']} B. {item['B']} C. {item['C']} D. {item['D']}\n答案：" for item in data]
    #     self.answers = [item["answer"] for item in data]
    #     self.tokenizer = tokenizer
    #     self.max_length = max_length

    # def __len__(self):
    #     return len(self.questions)

    # def __getitem__(self, idx):
    #     encoding = self.tokenizer(
    #         self.questions[idx],
    #         max_length=self.max_length,
    #         padding="max_length",
    #         truncation=True,
    #         return_tensors="pt"
    #     )
    #     return {
    #         "input_ids": encoding["input_ids"].squeeze(),
    #         "attention_mask": encoding["attention_mask"].squeeze(),
    #         "answer": self.answers[idx]
    #     }
class CEVALDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.samples = []
        for item in data:
            #item是{id:'',question:'','A':'','B':'','C':'','D':'','answer':'','explanation':''}
            prompt = f"问题：{item['question']}\n选项：\nA. {item['A']}\nB. {item['B']}\nC. {item['C']}\nD. {item['D']}\n答案："
            #A,B,C,D
            target = item["answer"]
            #'input_ids','attention_mask',"token_type_ids"
            # BatchEncoding 对象
            encoding = tokenizer(prompt, padding="max_length", truncation=True,return_tensors="pt")
            encoding["labels"] = ord(target) - 65  # 将A-D转换为0-3
            self.samples.append(encoding)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


test_dataset = CEVALDataset(test_data, tokenizer)
# print(test_dataset)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
# print(test_loader)
for data in test_loader:
    print(data)

# for item in test_data:
#     prompt = f"问题：{item['question']}\n选项：\nA. {item['A']}\nB. {item['B']}\nC. {item['C']}\nD. {item['D']}\n答案："
#     print(item)
#     target = item['answer']
#     encoding = tokenizer(prompt, return_tensors="pt")
    # print(item)
    # data = ord(target)-65
    # print(data)
    # print(target)
    # print(help(encoding))
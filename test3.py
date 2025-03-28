import torch
from transformers import AutoTokenizer, AutoModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
import json
import argparse
import os 
# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#加载数据
with open("/mnt/workspace/kl/hand_write/json_datasets/val.json","r") as f:
    test_data = json.load(f)  # 用户提供的测试集数据
## 加载教师模型（ChatGLM3-6b）
# teacher_model = AutoModel.from_pretrained(
#     "/mnt/workspace/.cache/modelscope/hub/ZhipuAI/chatglm3-6b",
#     trust_remote_code=True
# ).to(device).eval()
# 初始化学生模型（示例：TinyBERT）
#student_model = AutoModel.from_pretrained("google/bert_uncased_L-4_H-768_A-12").to(device)
#加载分词器
#tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-4_H-768_A-12")


teacher_model_name = "gpt2-medium"
student_model_name = "gpt2"  # 也可自定义更小的模型结构
 
# 加载教师和学生模型
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
teacher_model = GPT2LMHeadModel.from_pretrained(teacher_model_name)
teacher_model.eval()  # 推理/评估模式
 
student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
student_model = GPT2LMHeadModel.from_pretrained(student_model_name)


class CEVALDataset(Dataset):
    #data = test_data
    def __init__(self, data, tokenizer):
        self.samples = []
        for item in data:
            #item是{id:'',question:'','A':'','B':'','C':'','D':'','answer':'','explanation':''}
            prompt = f"问题：{item['question']}\n选项：\nA. {item['A']}\nB. {item['B']}\nC. {item['C']}\nD. {item['D']}\n答案："
            #A,B,C,D
            target = item["answer"]
            #'input_ids','attention_mask',"answer+，对prompt进行分词
            # BatchEncoding 对象
            encoding = tokenizer(prompt, padding="max_length", truncation=True,return_tensors="pt")
            label = ord(target) - 65  # 将A-D转换为0-3
            self.samples.append((prompt,label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt, label = self.samples[idx]
        return {"prompt": prompt, "label": label}
# def collate_fn(batch):
#     prompts = [item["prompt"] for item in batch]  
#     labels = torch.tensor([item["label"] for item in batch])
    
#     # 动态填充
#     inputs = tokenizer(
#         prompts,
#         padding="longest",      # 自动填充到批次内最长长度
#         truncation=True,
#         max_length=512,         # 设置最大长度限制
#         return_tensors="pt"
#     )
    
#     return {
#         "input_ids": inputs.input_ids,
#         "attention_mask": inputs.attention_mask,
#         "labels": labels
#     }

#将每个batch对齐
def collate_fn(batch):  
    prompts = [item["prompt"] for item in batch]  
    labels = torch.tensor([item["label"] for item in batch])  

    inputs = tokenizer(  
        prompts,  
        padding="longest",  
        truncation=True,  
        max_length=512,  
        return_tensors="pt"  
    )  

    return {  
        "input_ids": inputs.input_ids,  
        "attention_mask": inputs.attention_mask,  
        "labels": labels  
    }  
# 加载测试集（示例数据）


test_dataset = CEVALDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    collate_fn=collate_fn,    # 添加自定义collate函数
    shuffle=False)


# 原始蒸馏方法（标准KL散度）
def original_distillation_loss(student_logits, teacher_logits):
    return F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        F.softmax(teacher_logits, dim=-1),
        reduction="batchmean"
    )

# 改进的蒸馏损失函数（用户提供）
def improved_distillation_loss(
    student_logits,
    teacher_logits,
    # 独立控制教师与学生的温度
    teacher_temp=5.0,    # 教师温度：软化教师分布
    student_temp=2.0,     # 学生温度：控制学习强度
    # 动态温度衰减
    step=None,            # 当前训练步数（用于动态温度）
    max_steps=10000,     # 总训练步数
    reduction='batchmean',
    # 温度正则化约束
    min_temp=0.1,        # 温度下限防止数值不稳定
    temp_penalty=0.01    # 温度参数的正则化强度
):
    # 动态温度衰减策略（线性衰减到目标温度）
    def decay_temp(init_temp, final_temp=1.0):
        if step is None or max_steps <= 0:
            return init_temp
        progress = min(step / max_steps, 1.0)
        return max(init_temp * (1 - progress) + final_temp * progress, min_temp)
    
    # 应用温度衰减
    teacher_temp = decay_temp(teacher_temp, final_temp=2.0)
    student_temp = decay_temp(student_temp, final_temp=1.0)
    
    # 温度正则化项（防止温度趋近极端值）
    # temp_reg = temp_penalty * (
    #     (teacher_temp - 1.0)**2 + 
    #     (student_temp - 1.0)**2
    # )
    
    #计算温度正则化的偏方差
    temp_reg = temp_penalty*(
        (pow(teacher_temp - 1.0,2))+
        (pow(student_temp - 1.0,2))
    )
    # 温度缩放后的概率分布
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / teacher_temp, dim=-1)
    student_log_probs = F.log_softmax(student_logits / student_temp, dim=-1)
    # KL散度计算
    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction=reduction
    )
    # 温度补偿 + 正则化项
    #teacher_temp和student_temp都是通过温度动态衰减进行调整，以控制学习强度和教师分布的软化程度。
    #kl通过Logits计算 T->logits->kl
    loss = (student_temp ** 2) * kl + temp_reg
    return loss

# 训练循环
def train_distillation(teacher, student, loader, loss_fn):
    optimizer = torch.optim.AdamW(student.parameters(), lr=5e-5)
    #批量加载数据
    for batch in loader:
        #k是所有数据的key
        # inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device)
        }
        labels = batch["labels"].to(device)
        # 教师推理
        with torch.no_grad():
            teacher_outputs = teacher(**inputs).logits
            
        # 学生推理
        student_outputs = student(**inputs).logits
        
        # 计算损失
        loss = loss_fn(student_outputs, teacher_outputs)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



def evaluate(model, loader):
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)
            
            outputs = model(**inputs).logits
            preds = torch.argmax(outputs, dim=-1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return correct / total


parser = argparse.ArgumentParser()
parser.add_argument("--save_path",default = './dataset')
parser.add_argument("--save_name",default = './dl_test.pth')
#解析参数
args = parser.parse_args()
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)


# 运行两种蒸馏训练
print("训练原始蒸馏模型...")
train_distillation(teacher_model, student_model, test_loader, original_distillation_loss)
torch.save(student_model.state_dict(), f"{os.path.join(args.save_path, args.save_name)}")





# print("训练改进蒸馏模型...")
# train_distillation(teacher_model, student_model, test_loader, improved_distillation_loss)
# torch.save(student_model.state_dict(), "student_improved.pth")

# # 评估结果
# student_original = AutoModel.from_pretrained("google/bert_uncased_L-4_H-768_A-12").to(device)
# student_original.load_state_dict(torch.load("student_original.pth"))
# print(f"原始蒸馏准确率：{evaluate(student_original, test_loader):.2%}")

# student_improved = AutoModel.from_pretrained("google/bert_uncased_L-4_H-768_A-12").to(device)
# student_improved.load_state_dict(torch.load("student_improved.pth"))
# print(f"改进蒸馏准确率：{evaluate(student_improved, test_loader):.2%}")


test_batch = next(iter(test_loader))
print(f"输入尺寸检查：")
print(f"input_ids: {test_batch['input_ids'].size()}")
print(f"attention_mask: {test_batch['attention_mask'].size()}")
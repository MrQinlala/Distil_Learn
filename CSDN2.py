import torch
from torch import nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import load_dataset,Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import os
import argparse

import matplotlib.pyplot as plt

# 设置模型名称
teacher_model_name = "gpt2-medium"
# student_model_name = "gpt2"  # 也可自定义更小的模型结构
student_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu" 

# 加载教师和学生模型
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
##解决pad token
teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
teacher_model = GPT2LMHeadModel.from_pretrained(teacher_model_name).to(device)
teacher_model.eval()  # 推理/评估模式
 
student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)
# student_tokenizer.pad_token = student_tokenizer.eos_token

# 如果Qwen需要设置pad_token（可能已内置）
if student_tokenizer.pad_token is None:
    student_tokenizer.pad_token = student_tokenizer.eos_token
student_model = AutoModelForCausalLM.from_pretrained(student_model_name).to(device)
 
# 简单数据集：使用 wikitext-2 做语言建模蒸馏演示
train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split='train')
test_dataset  = load_dataset("wikitext", "wikitext-2-raw-v1", split='test')
# print(f"原始数据:{train_dataset[0]['text']}")
# print("空样本数量：", sum(1 for x in train_dataset if len(x["text"].strip()) == 0)) 
def tokenize_fn(examples):
    return teacher_tokenizer(
        examples["text"],
         truncation=True, 
         max_length=128,
         padding="max_length",
         return_tensors='pt')
#对所有数据集使用tokenize_fn进行处理
tokenized_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
tokenized_test_dataset = test_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])


# print(f"数据总量,{len(tokenized_dataset)}")
# print(f"示例数据",tokenized_dataset)
# print(type(tokenized_dataset))
# PyTorch DataLoader
#接收一个batch即列表，列表元素是字典形式
def collate_fn(batch):
    # 这里直接使用 teacher_tokenizer 的 pad 方法，也可用 student_tokenizer
    # return teacher_tokenizer.pad(batch, return_tensors="pt")
    if len(batch) == 0:
        raise ValueError("空批次数据")
    # 使用 student_tokenizer 保持数据一致性（原代码可能因 tokenizer 不一致导致问题）
    # 使用teacher_tokenizer对批次进行填充，避免数据长度等不一致
    return teacher_tokenizer.pad(  
        batch,
        return_tensors="pt",
        padding=True,
        max_length=128  # 显式设置最大长度
    )
 
 
train_loader = DataLoader(tokenized_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(tokenized_test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
 # 定义损失函数
def distillation_loss_function(teacher_logits, student_logits, 
                               labels, 
                               alpha=0.5, temperature=2.0):
    """
    teacher_logits, student_logits: (batch_size, seq_len, vocab_size)
    labels: (batch_size, seq_len)
    alpha: 权重，平衡真实任务损失 与 蒸馏损失
    temperature: 蒸馏温度
    
    返回: total_loss
    """
    # 1) LM 真实标签交叉熵
    #    让学生在真实标签上也保持一定的准确度
    #    -100 表示填充位置不计算loss
    lm_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),  #（batch_size*seq_length,vacab_size），使每个词的位置但对于进行交叉熵计算
        labels.view(-1),
        ignore_index=-100
    )
 
    # 2) 蒸馏损失 (KL 散度)
    #    对 teacher / student 的 logits 做 softmax with temperature
    #    p(t) = softmax(teacher_logits / T)
    #    q(s) = softmax(student_logits / T)
    #    dim=-1是
    teacher_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
     
    distill_loss = F.kl_div(
        student_probs, 
        teacher_probs.exp(),  # kl_div 需要 target 是概率分布 (非 log),所以要取exp(),实际上可以直接teacher_
        reduction='batchmean'
    ) * (temperature**2)
 
    total_loss = alpha * lm_loss + (1 - alpha) * distill_loss
    return total_loss, lm_loss, distill_loss

def improved_distillation_loss(
    student_logits,     #(b_s,seq_len,vocab_size)
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
    # def decay_temp(init_temp, final_temp=1.0):
    #     if step is None or max_steps <= 0:
    #         return init_temp
    #     progress = min(step / max_steps, 1.0)
    #     return max(init_temp * (1 - progress) + final_temp * progress, min_temp)
    
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
 
def temp_decay(init_temp,final_temp,max_steps,step)
     if step is None or max_steps <= 0:
        return init_temp
    progress = min(step / max_steps, 1.0)
    return max(init_temp * (1 - progress) + final_temp * progress, min_temp)
# 冻结教师模型，不参与训练
for param in teacher_model.parameters():
    param.requires_grad = False
 
optimizer = optim.AdamW(student_model.parameters(), lr=1e-5)
 
num_epochs = 1  # 简单跑1轮演示
alpha = 0.5
temperature = 2.0
 
student_model.train()


parser = argparse.ArgumentParser()
parser.add_argument("--save_path",default = './save')
parser.add_argument("--save_name",default = 'dl_test')
# parser.add_argument("--plot_path",default = '/root/autodl-tmp',help="损失曲线保存路径")

parser.add_argument("--plot_path",default = './save/img',help="损失曲线保存路径")
parser.add_argument("--plot_name",default = 'dl_test_first.png',help = "损失曲线文件名")

parser.add_argument("--temp_path",default = './save/temp',help = '温度保存路径')
parser.add_argument('--temp_name',default = 'dl_test_qwen')
#解析参数
args = parser.parse_args()

train_loss_history = []
#计算模型参数
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    train_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params,train_total_params
# teacher_total,teacher_train=count_parameters(teacher_model)
# student_total,student_train=count_parameters(student_model)
# print(f"teacher_total:{teacher_total},teacher_train:{teacher_train}")
# print(f"student_total:{student_total},student_train:{student_train}")


#测评模型
def evaluate(model, dataloader, tokenizer):
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # 标签处理（填充部分设为-100）
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item() * input_ids.size(0)  # 考虑batch_size差异
            
    avg_loss = total_loss / len(dataloader.dataset)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, perplexity

#  训练

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)


# student_model.load_state_dict(torch.load("/mnt/workspace/kl/hand_write/save/dl_test.pth"))
# student_loss,student_perplexity=evaluate(student_model, test_loader, student_tokenizer)
# print(f"student Model 测试结果:{student_loss}")
# print(f"平均损失: {student_loss:.4f}, 困惑度: {student_perplexity:.2f}")

# teacher_loss, teacher_perplexity = evaluate(teacher_model, test_loader, teacher_tokenizer)
# print(f"Teacher Model 测试结果:")
# print(f"平均损失: {teacher_loss:.4f}, 困惑度: {teacher_perplexity:.2f}")

# for epoch in range(num_epochs):
#     total_loss_val = 0.0
#     for step, batch in enumerate(train_loader):
      
#         ## 在训练循环中将输入数据移动到GPU修改，batch["inputs_ids"]对应当前批次所有样本的词元索引矩阵
#         # 赋值后的inputs_ids为(batch_size,seq_length),代表当前批次所有的词元的索引
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         assert input_ids.shape[-1] > 0, "输入维度异常"
#         with torch.no_grad():
#             # (batch__size,seq_length,vocab_size)
#             teacher_logits = teacher_model(input_ids, attention_mask=attention_mask).logits
            
 
#         student_logits = student_model(input_ids, attention_mask=attention_mask).logits
        
 
#         # labels 用来计算学生的 LM 任务损失
#         labels = input_ids.clone()
#         # 也可以把 padding位置设为 -100
#         # 标签处理将填充部分设为-100，避免计算这些位置的损失。
#         labels[labels==teacher_tokenizer.pad_token_id] = -100
 
#         loss = improved_distillation_loss(
#             student_logits,teacher_logits,
#             teacher_temp=5.0,
#             student_temp=2.0,
#             step=step,
#             max_steps=10000,
#             reduction='batchmean',
#             min_temp=0.1,
#             temp_penalty=0.01
#         )
#         #将每次的损失添加到train_loss_history列表中,便于后续画图
#         if step > 100:
#             train_loss_history.append(loss.item())
 
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         #增加损失值累加
#         total_loss_val += loss.item()
 
#         if step % 100 == 0:
#             print(f"Epoch {epoch}, Step {step}, Loss {loss.item():.4f}")
#             # torch.save(student_model.state_dict(),f"{args.save_path}/{args.save_name}{step//100}.pth")
#     avg_loss = total_loss_val / (step+1)
#     print(f"Epoch {epoch} finished, avg loss = {avg_loss:.4f}")

# torch.save(student_model.state_dict(),f"{args.save_path}/{args.save_name}.pth")
# plt.figure(figsize=(12, 6))

# plt.plot(train_loss_history,label='step loss')
# plt.title('Training Loss Curve')
# plt.xlabel('Training Steps')
# plt.ylabel('Loss Value')
# plt.legend()
# plt.grid(True)


# os.makedirs(args.plot_path,exist_ok=True)
# plot_save_path = os.path.join(args.plot_path, args.plot_name)
# plt.savefig(plot_save_path)
# print(f"Loss Curve saved to {plot_save_path}")
# plt.show()


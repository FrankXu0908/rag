import time
from transformers import AutoModelForCausalLM, AutoTokenizer
# 1. 初始化模型（推荐加载时配置优化参数）
model_path = "./Baichuan2-13B-Chat-v2.0.1"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",  # 自动选择最优精度
    trust_remote_code=True
).eval()

# 2. 构建符合模型要求的对话格式
prompt = """你是一位资深质量工程师，请详细介绍：
1. PFMEA的定义
2. 核心三要素
3. 典型实施步骤
4. 汽车行业的应用案例"""

# 3. 生成优化配置
generate_kwargs = {
    "max_new_tokens": 512,
    "do_sample": False,    # 关闭随机采样保证确定性结果
    "temperature": 0.7,     # 可调低至0.3-0.5提升专业性
    "top_p": 0.9,
    "repetition_penalty": 1.2,  # 避免重复
    "pad_token_id": tokenizer.eos_token_id
}

# 4. 执行推理（首次运行会有编译延迟）
start = time.time()
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, **generate_kwargs)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"生成内容：\n{response}")
print(f"耗时：{time.time() - start:.2f}秒")
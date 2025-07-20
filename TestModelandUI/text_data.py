from transformers import AutoTokenizer, AutoModelForCausalLM

# 使用清华MiniChat模型（仅1.5亿参数！）
model_name = "thunlp/MiniChat-1.5B-INT4"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print("launch(输入'退出'结束)")

while True:
    user_input = input("你：")
    if user_input.lower() == "退出":
        break
    
    # 构建输入
    inputs = tokenizer.encode(f"用户：{user_input}\nAI：", return_tensors="pt")
    
    # 生成回复（限制长度避免卡死）
    outputs = model.generate(
        inputs, 
        max_length=200,
        do_sample=True,
        temperature=0.7
    )
    
    # 打印结果（跳过用户输入部分）
    response = tokenizer.decode(outputs[0])[len(tokenizer.decode(inputs[0])):]
    print(f"AI：{response.strip()}")
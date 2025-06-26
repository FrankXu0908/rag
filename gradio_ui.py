import gradio as gr
import requests

API_URL = "http://localhost:8000/chat"
def query_fastapi(prompt):
    response = requests.post(API_URL, json={"prompt": prompt})
    print(response.text)
    if response.status_code == 200:
        return response.json().get("response", "❌ No response found.")
    return f"❌ Error: {response.status_code}"

gr.Interface(
    fn=query_fastapi,
    inputs=gr.Textbox(label="请输入 PMEA 分析内容", lines=6),
    outputs=gr.Textbox(label="生成的 PMEA 报告", lines=12),
    title="PMEA报告生成助手",
    description="输入质量问题描述，模型将自动生成专业 PMEA 报告"
).launch()
from transformers import pipeline
import gradio as gr


model = pipeline(
    "summarization",
    "captain-awesome/naveed-ggml-model-gpt4all-falcon-q4_0"
)

def predict(prompt):
    summary = model(prompt)[0]["summary_text"]
    return summary


# create an interface for the model
with gr.Interface(predict, "textbox", "text") as interface:
    interface.launch()

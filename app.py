 from transformers import pipeline
import gradio as gr

classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

def classify_text(text):
    result = classifier(text)
    label = result[0]['label']
    if label == 'LABEL_0':
        label='Negative'
    if label == 'LABEL_1':
        label='Neutral'
    if label == 'LABEL_2':
        label='Positive'
    return f"{label}"

interface = gr.Interface(
    fn=classify_text,
    inputs=gr.Textbox(label="Write anything (*-_-)"),
    outputs="text",
    title="Sentiment Analysis",
    description="Enter text to check the sentiment (Positive or Negative)."
)

interface.launch(debug=True, share=True)

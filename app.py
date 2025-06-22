import gradio as gr
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import torch
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import openai
import faiss
import numpy as np
import speech_recognition as sr
import pyttsx3
import gtts
import io
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables for OpenAI or other APIs
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY", "")

# Download NLTK data if not already present
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
model.to(device)

# Contextual memory for multi-turn conversation
conversation_history = []

# Example knowledge base (can be replaced with a real one)
knowledge_base = [
    ("What is Gradio?", "Gradio is a Python library for building machine learning web apps easily."),
    ("What is LLaVA?", "LLaVA is a large vision-language model for image and text understanding.")
]
vectorizer = TfidfVectorizer().fit([q for q, a in knowledge_base])
kb_vectors = vectorizer.transform([q for q, a in knowledge_base])

def get_kb_answer(question):
    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, kb_vectors)
    idx = np.argmax(sims)
    if sims[0, idx] > 0.5:
        return knowledge_base[idx][1]
    return None

def get_suggestions(user_input):
    # Simple suggestions based on input
    if 'hello' in user_input.lower():
        return ["How can I help you today?", "Tell me about yourself."]
    elif 'image' in user_input.lower():
        return ["Describe this image.", "What objects are in the image?"]
    return ["Can you explain more?", "Show me an example."]

def analyze_sentiment(text):
    score = sia.polarity_scores(text)
    if score['compound'] > 0.2:
        return 'positive'
    elif score['compound'] < -0.2:
        return 'negative'
    return 'neutral'

def ocr_image(image):
    try:
        import pytesseract
        return pytesseract.image_to_string(image)
    except Exception:
        return "OCR not available. Please install pytesseract."

def chat_with_image(image, question, history, user_name):
    try:
        # Personalization
        prefix = f"{user_name}, " if user_name else ""
        # Contextual memory
        full_prompt = "\n".join([f"User: {h[0]}\nBot: {h[1]}" for h in history[-3:]])
        prompt = f"[INST] <image>\n{full_prompt}\nUser: {question} [/INST]"
        inputs = processor(prompt, image, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=60)
        response = processor.decode(output[0], skip_special_tokens=True)
        # Knowledge base
        kb_ans = get_kb_answer(question)
        if kb_ans:
            response += f"\n\n[KB] {kb_ans}"
        # Sentiment
        sentiment = analyze_sentiment(question)
        if sentiment == 'negative':
            response += "\n\nI sense you might be upset. Let me know how I can help."
        # Suggestions
        suggestions = get_suggestions(question)
        # Update history
        history.append((question, response))
        return response, history, suggestions
    except Exception as e:
        return f"Sorry, an error occurred: {e}", history, []

def voice_to_text(audio):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)
        except Exception:
            return "Could not recognize speech."

def text_to_voice(text):
    tts = gtts.gTTS(text)
    fp = io.BytesIO()
    tts.save(fp)
    fp.seek(0)
    return fp

def reset_history():
    return []

with gr.Blocks() as iface:
    gr.Markdown("""
    # Image+Text Chatbot
    **Upload an image and ask a question.**
    - Type your question or use voice input.
    - Click a suggestion to quickly ask a follow-up.
    """)
    user_name = gr.Textbox(label="Your Name (optional)", visible=False)
    chatbot = gr.Chatbot(type="messages")
    image_input = gr.Image(type="pil", label="Upload Image")
    input_mode = gr.Radio(["Text", "Voice"], value="Text", label="Input Mode", interactive=True)
    question = gr.Textbox(label="Type your question and press Enter", visible=True)
    voice_input = gr.Audio(sources=["upload"], type="filepath", label="Upload voice question", visible=False)
    suggestions = gr.Row([gr.Button(value="", visible=False, elem_id=f"sugg{i}") for i in range(3)])
    state = gr.State([])

    def toggle_input(mode):
        return gr.update(visible=(mode=="Text")), gr.update(visible=(mode=="Voice"))

    def handle_input(image, question, state, user_name, voice_input, input_mode):
        if input_mode == "Voice" and voice_input:
            question = voice_to_text(voice_input)
        response, state, sugg = chat_with_image(image, question, state, user_name)
        # Show up to 3 suggestions as buttons
        sugg = sugg[:3] if sugg else ["", "", ""]
        return response, state, [gr.update(value=s, visible=bool(s)) for s in sugg]

    input_mode.change(toggle_input, [input_mode], [question, voice_input])
    question.submit(handle_input, [image_input, question, state, user_name, voice_input, input_mode], [chatbot, state] + [s for s in suggestions.children])
    for i, btn in enumerate(suggestions.children):
        btn.click(lambda img, st, un, idx=i: handle_input(img, btn.value, st, un, None, "Text"), [image_input, state, user_name], [chatbot, state] + [s for s in suggestions.children])

    with gr.Accordion("More Options", open=False):
        gr.Button("Extract Text from Image (OCR)").click(lambda img: ocr_image(img), inputs=image_input, outputs=question)
        gr.Button("Reset Conversation").click(reset_history, None, state)

if __name__ == "__main__":
    iface.launch(share=True)

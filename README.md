# Image-based Chatbot (LLaVA)

A conversational AI assistant that can understand and discuss images with users. Built using state-of-the-art vision-language models (LLaVA) and a user-friendly Gradio interface.

---

## Features
- Upload an image and ask questions about it
- Receives accurate, context-aware responses
- Powered by LLaVA (CLIP image encoder + Vicuna/LLaMA text decoder)
- Simple web interface (Gradio)
- Public sharing link for easy collaboration

## New Features (2025 Update)
- **Contextual Memory:** Remembers previous messages for more natural, multi-turn conversations.
- **Suggestions:** Clickable quick-reply buttons help guide the conversation.
- **Image Handling:** Supports both image Q&A and image text extraction (OCR).
- **Sentiment Analysis:** Detects user sentiment and adapts responses.
- **Voice Input/Output:** Ask questions by voice and get spoken answers.
- **Knowledge Base Integration:** Answers common questions from a built-in knowledge base.
- **Personalization:** Optionally address users by name.
- **Error Handling:** Friendly error messages for unrecognized input or issues.
- **Simple Interface:** Clean, single-question box with easy toggling between text and voice. Advanced options are hidden for a clutter-free experience.

## Architecture
```
[User Image] → [Image Encoder (CLIP)] → 
                                  ↘
                               [Multimodal Projector] → [Text Decoder (Vicuna/LLaMA)] → Response
                    ↑
[User Text Prompt] →
```

## System Architecture Explained

This project uses a modern vision-language model architecture inspired by LLaVA and similar systems. Here’s how the components interact:

1. **User Image Upload**: The user uploads an image through the web interface.
2. **Image Encoder (CLIP)**: The image is processed by a pre-trained image encoder (such as CLIP), which converts the image into a dense feature representation (embedding).
3. **User Text Prompt**: The user enters a question or prompt related to the image.
4. **Multimodal Projector**: The image embedding and the text prompt are projected into a shared feature space, aligning their representations so they can be understood together.
5. **Text Decoder (Vicuna/LLaMA)**: The combined representation is passed to a large language model decoder, which generates a natural language response based on both the image and the text prompt.
6. **Response**: The system returns a detailed, context-aware answer to the user via the web interface.

This architecture allows the chatbot to understand and reason about both visual and textual information, enabling rich, multimodal conversations.

## Setup Instructions
1. **Clone the repository or download the project files.**
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Run the app:**
   ```sh
   python app.py
   ```
4. **Access the web interface:**
   - Open the local or public link provided in your terminal.

## Usage
- Upload an image (JPG, PNG, etc.)
- Type a question (e.g., "What is the person doing in this image?")
- Receive a detailed, AI-generated response

## How to Use (Quick Guide)
1. **Upload an image.**
2. **Choose input mode:** Text or Voice.
3. **Ask your question** (type or upload voice).
4. **Click suggestions** for quick follow-ups.
5. **(Optional)** Use 'More Options' for OCR or to reset the chat.

## Notes
- Large models may take time to respond, especially on CPU. For best performance, use a machine with a GPU.
- For faster responses, reduce the number of generated tokens or use a smaller model.
- Informational messages about input order and token settings are normal.

## Credits
- Built with [Hugging Face Transformers](https://huggingface.co/docs/transformers/index), [Gradio](https://gradio.app/), and [LLaVA](https://llava-vl.github.io/)
- Made with ❤️ by shindeeas

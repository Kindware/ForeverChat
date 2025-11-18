# ForeverChat  
### Smarter Local AI Conversations — Topic-Aware Memory, Auto-Juggling Context, and Lightweight RAG

ForeverChat is a fully local, Ollama-powered conversational engine with long-term memory, automatic topic management, context compression, and a lightweight RAG system backed by ChromaDB.  

It works as both a **Flask web API** and a full **CLI chat interface**, giving you ChatGPT-style persistence with zero cloud dependency.

You need ollama (make sure to 'ollama serve' if you need to) and gemma3:12b to run it out of the box , you can change to whatever model you want though.

'This software works
but it's not done
rip it apart
and have some fun'

---

## 🚀 Features

### 🧠 Topic-Aware Context Manager
Automatically tracks conversation topics, archives old ones, and retrieves them when mentioned again.  
Built on `SmartContextManager`, which dynamically manages active context up to 50% of the model's window.

### 🧹 Conversational Auto-Cleaning
Removes filler (“um”, “lol”, “you know”), sentiment-only chatter (“cool!”, “thanks!”), and repeated fluff before it ever hits the model.  
Keeps context lean and meaningful.

### 🗂️ Dynamic Topic Detection
- Recognizes predefined topics (AI, programming, hardware, Linux, homelab, etc.)  
- Automatically creates brand-new topics based on nouns and keyword patterns  
- Ensures relevant memory gets surfaced during conversation

### 📦 Lightweight RAG System
Powered by:
- **ChromaDB** for long-term embedding storage  
- **SentenceTransformers** for embeddings  
- Topic-aware retrieval (only retrieves notes relevant to the current subject)

### 🎛️ Smart Context Juggling
- Active exchanges limited to ~50% of max context  
- Old topics get archived when cold  
- Auto-retrieval when a topic is mentioned again  
- Ensures you *never* blow past the model's token window

### 🧰 Fully Local Ollama Integration
- Works with any model you have locally (Gemma, Llama, Mistral, etc.)  
- Automatically sizes context windows using configurable model profiles  
- Simple HTTP API wrapper inside `OllamaClient`

### 🌐 Flask Web API Included
Endpoints for:
- `/api/chat`
- `/api/command`
- `/api/models`
- `/api/health`

### 🖥️ CLI Chat App
A clean terminal interface with commands:

### Come by www.kindware.ca and leave a comment in the forums if you create something awesome,  I would love to see it!

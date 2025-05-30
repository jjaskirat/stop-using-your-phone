# stop-using-your-phone!!

This project is an **AI-powered, real-time video surveillance assistant** that detects phone usage from webcam input using a multimodal local LLM. It provides voice feedback, visual alerts via smart lighting (Home Assistant), and runs on a Gradio interface.

## Video Demo:


https://github.com/user-attachments/assets/f6624169-3915-46a6-9e4e-071c0f12045b



## 🧠 Features

- 🖼️ Processes real-time webcam video frames
- 🤖 Uses a **local multimodal LLM** (OpenAI-compatible) for frame analysis
- 🔊 **Text-to-speech** alerts using Piper TTS
- 🚨 **Smart light flashing** via Home Assistant when phone usage is detected
- 📦 Built with `Gradio`, `LangChain`, `Llama CPP`, `Piper TTS` and `Home Assistant`

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/multimodal-surveillance-assistant.git
cd multimodal-surveillance-assistant
```
### 2. Install the Requirements

Create a Virtual Envirronment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

```bash
pip install requests langchain-openai gradio pillow numpy
```
#### Other Requirements:
- Llama CPP server (https://github.com/ggml-org/llama.cpp)
- Piper TTS server (https://github.com/rhasspy/piper)
- Home Assistant server (https://www.home-assistant.io/installation/)

**PS: It is recommended to install the docker containers for all the above**

### 3. Start Servers

- Multimodal Qwen2.5-VL for image processing on port 8000 `docker run --gpus all -p 8000:8000 <container-id> --port 8000 --host 0.0.0.0 -hf ggml-org/Qwen2.5-VL-3B-Instruct-GGUF --jinja`
- Llama 3.2 for TTS text generation on port 8001 `docker run --gpus all -p 8001:8001 <container-id> --port 8001 --host 0.0.0.0 -hf bartowski/Llama-3.2-3B-Instruct-GGUF --jinja`
- Piper TTS on port 5000 `docker run -it -p 5000:5000 -p 10200:10200 -v /path/to/local/data:/data rhasspy/wyoming-piper --voice en_US-lessac-medium`
- Home Assistant Server on port 8213 `docker run -d --name homeassistant --privileged --restart=unless-stopped -e TZ=MY_TIME_ZONE -v /PATH_TO_YOUR_CONFIG:/config -v /run/dbus:/run/dbus:ro --network=host ghcr.io/home-assistant/home-assistant:stable`
- Then add your lights in hassio

### 4: Run the App
`python app.py`

### 📸 How It Works

1. A webcam frame is streamed into the Gradio UI.
2. The frame is base64-encoded and sent with a prompt to a multimodal LLM.
3. The LLM replies with a structured JSON indicating:
```json
{
  using_phone: whether the subject is using a phone.
  consistently_using_phone: if usage is frequent over time.
  doing_what: a brief action summary.
}
```
4. If phone usage is detected:
  - A firm voice warning is generated with Llama 3.2.
  - The voice is played on the speaker using Piper TTS
  - A red strobe effect is triggered via Home Assistant lights.

---
**Cheers**
---


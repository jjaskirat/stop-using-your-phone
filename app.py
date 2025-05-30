# === Standard Library Imports ===
import asyncio
import base64
from copy import copy
import io
import json
import subprocess
import time
from typing import Any, List, Optional, Dict, Union

# === Third-Party Library Imports ===
import requests
import gradio as gr
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# === Pydantic Schema for Structured LLM Output ===
class UsingPhone(BaseModel):
    """
    Structured output for detecting phone usage from the LLM response.
    You can use this for stricter output validation.
    """
    using_phone: bool = Field(description="Is the user using their phone?")
    doing_what: str = Field(description="Briefly describe the user's actions.")

# === Helper Class for Multimodal Interaction ===
class MultimodalHelper:
    def __init__(self, config: Optional[Dict[str, Any]]) -> None:
        self.config = config

        # Initialize LLM (e.g., LLaMA.cpp server)
        self.llm = ChatOpenAI(**config)

        # Structured output helpers
        self.structured_llm = self.llm.with_structured_output(UsingPhone)
        self.structured_llm_json = self.llm.with_structured_output(None, method="json_mode")

    def encode_numpy_array(self, array: np.ndarray, image_format: str = 'jpeg') -> str:
        """
        Encode a NumPy array as a base64 image string.
        """
        image = Image.fromarray(array)
        buffer = io.BytesIO()
        image.save(buffer, format=image_format)
        base64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/{image_format};base64,{base64_string}"

    def decode_numpy_array(self, base_64_url: str) -> np.ndarray:
        """
        Decode a base64 URL string back to a NumPy array.
        """
        image_data = base_64_url.split(',')[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        return np.array(image)

    def get_message(self, image: np.ndarray, query: str) -> List[Union[SystemMessage, HumanMessage]]:
        """
        Compose a prompt with an image and text for basic LLM processing.
        """
        image_base64 = self.encode_numpy_array(image)
        return [
            SystemMessage(content="You are a video surveillance inspector. Use action verbs only."),
            HumanMessage(content=[
                {"type": "image_url", "image_url": {"url": image_base64}},
                {"type": "text", "text": query}
            ])
        ]

    def get_message_json(
        self, image: np.ndarray, user_history: List[bool]
    ) -> List[Union[SystemMessage, HumanMessage]]:
        """
        Compose a prompt asking for structured JSON output, including user history.
        """
        image_base64 = self.encode_numpy_array(image)
        messages = [
            SystemMessage(
                content="""You are a video surveillence inspector..
                        You will only use action verbs and be very precise"""
            ),
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_base64
                        },
                    },
                    {
                        "type": "text",
                        "text": """Respond in json
Key: `using_phone`: bool = Is the user is using ther phone? Or do they have in in their hand?
Key: `consistently_using_phone`: bool = Majority of the User History. Later elements hold more weight
Key: `doing_what`: str = Briefly describe what the user is doing"""
                    },
                    {
                        "type": "text",
                        "text": f"User History: {user_history}"
                    }
                ]
            )
        ]
        return messages

    def predict(
        self, image: np.ndarray, user_history: List[bool]
    ) -> tuple[np.ndarray, Dict[str, Any], List[bool]]:
        """
        Main prediction function integrating LLM output and alert logic.
        """
        message = self.get_message_json(image, user_history if len(user_history) >= 10 else [])
        response = self.structured_llm_json.invoke(message)

        # Update user history
        if len(user_history) > 10:
            user_history = user_history[1:]
        user_history.append(response['using_phone'])

        return image, response, user_history

# === Text-to-Speech Helper Class ===
class TTSHelper:
    def __init__(self, config: Optional[Dict[str, Any]]) -> None:
        self.config = copy(config)
        config.pop('piper_url')
        self.llm = ChatOpenAI(**config)

    def get_message(self) -> List[Union[SystemMessage, HumanMessage]]:
        """
        Prompt to command the user to stop using their phone.
        """
        return [
            SystemMessage(content="Be firm, direct, and under 10 words."),
            HumanMessage(content=[{"type": "text", "text": "Command the user to stop using their phone"}])
        ]

    def predict(self) -> None:
        """
        Generate and speak the command.
        """
        message = self.get_message()
        command = self.llm.invoke(message).content
        self.speak_with_piper(command)

    def speak_with_piper(self, message: str) -> None:
        """
        Send the message to the Piper TTS server.
        """

        response = requests.post(self.config["piper_url"], json=message)
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return

        self.speak_response(response)

    def speak_response(self, response: requests.Response) -> None:
        """
        Play the response audio.
        """
        with open("output.wav", "wb") as f:
            f.write(response.content)
        subprocess.Popen(["aplay", "output.wav"])

# === Home Assistant Helper Class ===
class HassIOHelper:
    def __init__(self, config: Optional[Dict[str, Any]]) -> None:
        # Hardcoded for now — ideally load securely
        self.config = config

    def call_service(self, service: str, data: Dict[str, Any]) -> None:
        """
        Call a Home Assistant service with given data.
        """
        headers = {
            "Authorization": f"Bearer {self.config['access_token']}",
            "Content-Type": "application/json"
        }
        url = f"{self.config['home_assistant_url']}/api/services/{service}"
        r = requests.post(url, headers=headers, json=data, timeout=5)
        r.raise_for_status()

    def red_strobe(self, flashes: int = 5, on_ms: float = 0.3, off_ms: float = 0.3) -> None:
        """
        Flash red light multiple times for visual alert.
        """
        for _ in range(flashes):
            self.call_service("light/turn_on", {
                "entity_id": self.config['entity_id'],
                "rgb_color": [255, 0, 0],
                "brightness": 255
            })
            time.sleep(on_ms)
            self.call_service("light/turn_off", {"entity_id": self.config['entity_id']})
            time.sleep(off_ms)
            
class PutYourPhoneAway:
    def __init__(self, config: Optional[Dict[str, Any]]) -> None:
        self.multimodal_llm_helper = MultimodalHelper(config=config['multimodal_llm_helper'])
        self.tts_helper = TTSHelper(config=config['tts_helper'])
        self.hassio_helper = HassIOHelper(config=config['hassio_helper'])
        self.stream_now = True
        
    def predict(
        self, image: np.ndarray, user_history: List[bool]
    ) -> tuple[np.ndarray, Dict[str, Any], List[bool]]:
        self.stream_now = False
        image, response, user_history = self.multimodal_llm_helper.predict(
            image = image,
            user_history = user_history
        )
        self.stream_now = True
        if response.get('consistently_using_phone') and response.get('using_phone'):
            self.tts_helper.predict()
            self.hassio_helper.red_strobe(flashes=5)
        return image, response, user_history
    
    def get_streaming_info(self) -> bool:
        """
        Return whether streaming is enabled.
        """
        return self.stream_now
        
        
multimodal_config = {
    "openai_api_base": "http://127.0.0.1:8000/v1",
    "openai_api_key": "sk-xxx",  # Dummy key for local server
    "temperature": 0,
}
tts_config = {
    "openai_api_base": "http://127.0.0.1:8001/v1",
    "openai_api_key": "sk-xxx",
    "temperature": 3,
    "piper_url": "http://localhost:5000/api/text-to-speech",
}            
hassio_config = {
    "home_assistant_url": "http://192.168.122.165:8123",
    "access_token": "your-access-token-here",
    "entity_id": "light.lamp",
    "tts_entity_id": "tts.piper",
    "media_player": "media_player.bedroom_speaker"
}

put_your_phone_away_config = {
    "multimodal_llm_helper": multimodal_config,
    "tts_helper": tts_config,
    "hassio_helper": hassio_config,
}

# === Gradio Interface ===
with gr.Blocks(theme=gr.themes.Glass()) as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(streaming=True, label='Input Video')
        with gr.Column():
            output_img = gr.Image(streaming=True, label='Current Frame')
    user_history = gr.State(value=[])
    output_content = gr.Textbox(label='LLM Output')

    # Instantiate the helper
    put_your_phone_away = PutYourPhoneAway(config=put_your_phone_away_config)

    # Stream input image and get prediction
    input_image.stream(
        put_your_phone_away.predict,
        [input_image, user_history],
        [output_img, output_content, user_history],
        time_limit=600,
        stream_every=put_your_phone_away.get_streaming_info(),
        concurrency_limit=None
    )

if __name__ == '__main__':
    demo.launch()
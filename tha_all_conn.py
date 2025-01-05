import sys
import os
import json
import requests
import logging
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QComboBox, QLineEdit
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QTextCursor, QPalette

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("multi_service_client.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

###############################################################################
# Service & Model Configuration
###############################################################################
# For services that don't provide an API for listing models, we can specify them manually.
# For Ollama, we’ll show how to fetch the model list.
SERVICE_CONFIG = {
    "Ollama": {
        "enable_model_fetch": True,
        "fetch_models_url": "http://localhost:11434/api/tags",
        "generate_url": "http://localhost:11434/api/generate",
        "api_key_needed": False,
        "env_var": None,
    },
    "Groq": {
        "enable_model_fetch": False,
        "fetch_models_url": None,
        "generate_url": "https://api.groq.com/openai/v1/chat/completions",
        "hardcoded_models": [
            "llama3-8b-8192",
            "gemma2-9b-it",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "llama-3.2-3b-preview",
        ],
        "api_key_needed": True,
        "env_var": "GROQ_API_KEY",
    },
    "Google": {
        "enable_model_fetch": True,  # Fetch from Google's API
        "fetch_models_url": "https://generativelanguage.googleapis.com/v1beta/models",
        "generate_url": "https://generativelanguage.googleapis.com/v1beta/{model}:generateContent",
        "hardcoded_models": [],  # Fetch dynamically
        "api_key_needed": True,
        "env_var": "GOOGLE_API_KEY",
    },
    "Deepseek": {
        "enable_model_fetch": False,
        "fetch_models_url": None,
        "generate_url": "https://api.deepseek.com/chat/completions",
        "hardcoded_models": ["deepseek-chat"],
        "api_key_needed": True,
        "env_var": "DS_API_KEY",
    },
    "OpenAI": {
        "enable_model_fetch": False,
        "fetch_models_url": None,
        "generate_url": "https://api.openai.com/v1/chat/completions",
        "hardcoded_models": [
            "gpt-4o-mini",
            "chatgpt-4o-latest",
            "o1",
            "o1-mini"
        ],
        "api_key_needed": True,
        "env_var": "OPENAI_API_KEY",
    },
    "SambaNova": {
        "enable_model_fetch": False,
        "fetch_models_url": None,
        "generate_url": "https://api.sambanova.ai/v1/chat/completions",
        "hardcoded_models": [
            "Meta-Llama-3.1-405B-Instruct",
            "Meta-Llama-3.1-8B-Instruct",
            "Meta-Llama-3.2-3B-Instruct",
            "Qwen2.5-Coder-32B-Instruct",
            "QwQ-32B-Preview",
            "Qwen2.5-72B-Instruct"
        ],
        "api_key_needed": True,
        "env_var": "SN_API_KEY",
    },
}

###############################################################################
# Worker Thread for Fetching Models (Ollama and Google)
###############################################################################
class FetchModelsWorker(QThread):
    models_fetched = Signal(list)
    error_occurred = Signal(str)

    def __init__(self, fetch_url, api_key=None, parent=None):
        super().__init__(parent)
        self.fetch_url = fetch_url
        self.api_key = api_key

    def run(self):
        try:
            if not self.fetch_url:
                logging.debug("No fetch URL provided. Emitting empty list.")
                self.models_fetched.emit([])
                return

            logging.debug(f"Fetching available models from {self.fetch_url}")
            headers = {}
            if self.api_key:
                if "generativelanguage" in self.fetch_url:
                    # Google needs the key in the URL
                    self.fetch_url = f"{self.fetch_url}?key={self.api_key}"
                else:
                    # Assume others want it in the header
                    headers["Authorization"] = f"Bearer {self.api_key}"

            response = requests.get(self.fetch_url, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            if 'models' in data:  # Ollama
                models = [model['name'] for model in data['models']]
            elif 'candidates' in data:  # Google Generative AI:
                models = [model["content"]["parts"][0]["text"] for model in data["candidates"]]
            elif 'model' in data:
                models = [data['model']]
            else:
                raise ValueError("Unknown response format. Cannot find model list.")

            logging.debug(f"Models fetched successfully: {models}")
            self.models_fetched.emit(models)
        except requests.RequestException as e:
            error_message = f"Network error while fetching models: {e}"
            logging.error(error_message)
            self.error_occurred.emit(error_message)
        except json.JSONDecodeError as e:
            error_message = f"JSON decode error while fetching models: {e}"
            logging.error(error_message)
            self.error_occurred.emit(error_message)
        except ValueError as e:
            error_message = f"Error parsing model list: {e}"
            logging.error(error_message)
            self.error_occurred.emit(error_message)
        except Exception as e:
            error_message = f"Unexpected error while fetching models: {e}"
            logging.error(error_message)
            self.error_occurred.emit(error_message)

###############################################################################
# Worker Thread for Generating Responses
###############################################################################
class GenerateWorker(QThread):
    response_received = Signal(str)
    generation_complete = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, service_name, model, prompt, context=None, parent=None):
        super().__init__(parent)
        self.service_name = service_name
        self.model = model
        self.prompt = prompt
        self.context = context

        # Pull the service config
        service_cfg = SERVICE_CONFIG.get(self.service_name, {})
        env_var = service_cfg.get("env_var")

        # If the service requires an API key, we try to retrieve it from env
        self.api_key = None
        if service_cfg.get("api_key_needed", False) and env_var:
            self.api_key = os.environ.get(env_var)
            if not self.api_key:
                logging.error(f"API key not found in environment variable: {env_var}")

    def run(self):
        """
        Handle the request based on the selected service.
        """
        service_cfg = SERVICE_CONFIG.get(self.service_name, {})
        if not service_cfg:
            msg = f"No configuration found for service '{self.service_name}'"
            logging.error(msg)
            self.error_occurred.emit(msg)
            return

        try:
            if self.service_name == "Ollama":
                self.run_ollama(service_cfg)
            elif self.service_name == "Groq":
                self.run_groq(service_cfg)
            elif self.service_name == "Google":
                self.run_google(service_cfg)
            elif self.service_name == "Deepseek":
                self.run_deepseek(service_cfg)
            elif self.service_name == "OpenAI":
                self.run_openai(service_cfg)
            elif self.service_name == "SambaNova":
                self.run_sambanova(service_cfg)
            else:
                msg = f"Unsupported service: {self.service_name}"
                logging.error(msg)
                self.error_occurred.emit(msg)
        except requests.RequestException as e:
            error_message = f"Network error during generation: {e}"
            logging.error(error_message)
            self.error_occurred.emit(error_message)
        except Exception as e:
            error_message = f"Unexpected error during generation: {e}"
            logging.error(error_message)
            self.error_occurred.emit(error_message)

    def run_ollama(self, service_cfg):
        """
        Ollama uses a streaming endpoint at /api/generate.
        """
        url = service_cfg["generate_url"]
        headers = {'Content-Type': 'application/json'}
        data = {
            "model": self.model,
            "prompt": self.prompt,
            "stream": True
        }
        if self.context:
            data["context"] = self.context

        logging.debug(f"[Ollama] Sending request to {url} with data={data}")
        with requests.post(url, headers=headers, json=data, stream=True, timeout=60) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    try:
                        json_obj = json.loads(line.decode('utf-8'))
                        logging.debug(f"[Ollama] Received JSON object: {json_obj}")
                        if 'response' in json_obj and json_obj['response']:
                            self.response_received.emit(json_obj['response'])
                        if json_obj.get('done', False):
                            self.generation_complete.emit(json_obj)
                            break
                    except json.JSONDecodeError as e:
                        error_message = f"JSON decode error in streaming response: {e}"
                        logging.error(error_message)
                        self.error_occurred.emit(error_message)

    def run_groq(self, service_cfg):
        """
        Groq expects an OpenAI-compatible chat request.
        """
        url = service_cfg["generate_url"]
        if not self.api_key:
            raise ValueError("Groq API key is required but not provided in environment variable.")

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": self.prompt
                }
            ]
        }
        logging.debug(f"[Groq] Sending request to {url} with data={data}")
        resp = requests.post(url, headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        json_obj = resp.json()
        logging.debug(f"[Groq] Received JSON object: {json_obj}")

        try:
            content = json_obj["choices"][0]["message"]["content"]
            self.response_received.emit(content)
        except KeyError as e:
            self.error_occurred.emit(f"Error parsing Groq response: Missing key {e}")
            return

        usage_info = {
            "prompt_tokens": json_obj.get("usage", {}).get("prompt_tokens", 0),
            "completion_tokens": json_obj.get("usage", {}).get("completion_tokens", 0),
            "total_tokens": json_obj.get("usage", {}).get("total_tokens", 0),
        }
        self.generation_complete.emit(usage_info)

    def run_google(self, service_cfg):
        """
        Google expects a different JSON body and returns a 'candidates' structure.
        """
        if not self.api_key:
            raise ValueError("Google API key is required but not provided in environment variable.")

        # Model is in the URL
        url = service_cfg["generate_url"].format(model=self.model)
        full_url = f"{url}?key={self.api_key}"
        headers = {'Content-Type': 'application/json'}

        data = {
            "contents": [
                {
                    "parts": [
                        {"text": self.prompt}
                    ]
                }
            ]
        }
        logging.debug(f"[Google] Sending request to {full_url} with data={data}")
        resp = requests.post(full_url, headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        json_obj = resp.json()
        logging.debug(f"[Google] Received JSON object: {json_obj}")

        try:
            candidates = json_obj["candidates"]
            parts = candidates[0]["content"]["parts"]
            content = "".join(part["text"] for part in parts)
            self.response_received.emit(content)
        except KeyError as e:
            self.error_occurred.emit(f"Error parsing Google response: Missing key {e}")
            return

        usage_info = {
            "prompt_tokens": json_obj.get("usageMetadata", {}).get("promptTokenCount", 0),
            "candidates_tokens": json_obj.get("usageMetadata", {}).get("candidatesTokenCount", 0),
            "total_tokens": json_obj.get("usageMetadata", {}).get("totalTokenCount", 0),
        }
        self.generation_complete.emit(usage_info)

    def run_deepseek(self, service_cfg):
        """
        Deepseek - similar to Groq/OpenAI
        """
        url = service_cfg["generate_url"]
        if not self.api_key:
            raise ValueError(
                "Deepseek API key is required but not provided in environment variable."
            )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": self.prompt}],
            "stream": False
        }
        logging.debug(f"[Deepseek] Sending request to {url} with data={data}")
        resp = requests.post(url, headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        json_obj = resp.json()
        logging.debug(f"[Deepseek] Received JSON object: {json_obj}")

        try:
            content = json_obj["choices"][0]["message"]["content"]
            self.response_received.emit(content)
        except KeyError as e:
            self.error_occurred.emit(f"Error parsing Deepseek response: Missing key {e}")
            return

        usage_info = {}
        self.generation_complete.emit(usage_info)

    def run_openai(self, service_cfg):
        """
        OpenAI
        """
        url = service_cfg["generate_url"]
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required but not provided in environment variable."
            )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": self.prompt}],
        }
        logging.debug(f"[OpenAI] Sending request to {url} with data={data}")
        resp = requests.post(url, headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        json_obj = resp.json()
        logging.debug(f"[OpenAI] Received JSON object: {json_obj}")

        try:
            content = json_obj["choices"][0]["message"]["content"]
            self.response_received.emit(content)
        except KeyError as e:
            self.error_occurred.emit(f"Error parsing OpenAI response: Missing key {e}")
            return

        usage_info = {
            "prompt_tokens": json_obj.get("usage", {}).get("prompt_tokens", 0),
            "completion_tokens": json_obj.get("usage", {}).get("completion_tokens", 0),
            "total_tokens": json_obj.get("usage", {}).get("total_tokens", 0),
        }
        self.generation_complete.emit(usage_info)

    def run_sambanova(self, service_cfg):
        """
        SambaNova
        """
        url = service_cfg["generate_url"]
        if not self.api_key:
            raise ValueError(
                "SambaNova API key is required but not provided in environment variable."
            )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": self.prompt}],
        }
        logging.debug(f"[SambaNova] Sending request to {url} with data={data}")
        resp = requests.post(url, headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        json_obj = resp.json()
        logging.debug(f"[SambaNova] Received JSON object: {json_obj}")

        try:
            content = json_obj["choices"][0]["message"]["content"]
            self.response_received.emit(content)
        except KeyError as e:
            self.error_occurred.emit(f"Error parsing SambaNova response: Missing key {e}")
            return

        usage_info = {}
        self.generation_complete.emit(usage_info)

###############################################################################
# Main Application Window
###############################################################################
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Service Client")
        self.setGeometry(100, 100, 800, 600)
        self.setup_ui()

        # If you store context for conversational memory, do so here
        self.context = None

    def setup_ui(self):
        # Apply dark theme
        self.apply_dark_theme()

        # Main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # SERVICE SELECTION
        service_layout = QHBoxLayout()
        service_label = QLabel("Select Service:")
        self.service_combo = QComboBox()
        # Populate the service combo box
        self.service_combo.addItems(list(SERVICE_CONFIG.keys()))
        self.service_combo.currentTextChanged.connect(self.on_service_changed)
        service_layout.addWidget(service_label)
        service_layout.addWidget(self.service_combo)
        main_layout.addLayout(service_layout)

        # MODEL SELECTION
        model_layout = QHBoxLayout()
        model_label = QLabel("Select Model:")
        self.model_combo = QComboBox()
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        main_layout.addLayout(model_layout)

        # FETCH MODELS BUTTON (for Ollama or any other service supporting fetch)
        self.fetch_models_button = QPushButton("Fetch Models")
        self.fetch_models_button.clicked.connect(self.fetch_models)
        main_layout.addWidget(self.fetch_models_button)

        # RESPONSE DISPLAY AREA
        self.response_display = QTextEdit()
        self.response_display.setReadOnly(True)
        main_layout.addWidget(self.response_display)

        # INPUT AREA LAYOUT
        input_layout = QHBoxLayout()
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Enter your prompt here...")
        self.send_button = QPushButton("Send")
        input_layout.addWidget(self.prompt_input)
        input_layout.addWidget(self.send_button)
        main_layout.addLayout(input_layout)

        # Connect send button
        self.send_button.clicked.connect(self.send_prompt)

        # Initialize the UI based on the default selected service
        self.on_service_changed(self.service_combo.currentText())

    def apply_dark_theme(self):
        try:
            palette = QPalette()

            # Window colors
            palette.setColor(QPalette.Window, Qt.black)
            palette.setColor(QPalette.WindowText, Qt.black)
            palette.setColor(QPalette.Base, Qt.white)
            palette.setColor(QPalette.AlternateBase, Qt.white)
            palette.setColor(QPalette.ToolTipBase, Qt.black)
            palette.setColor(QPalette.ToolTipText, Qt.black)
            palette.setColor(QPalette.Text, Qt.black)
            palette.setColor(QPalette.Button, Qt.blue)
            palette.setColor(QPalette.ButtonText, Qt.black)
            palette.setColor(QPalette.BrightText, Qt.red)

            # Highlight colors
            palette.setColor(QPalette.Highlight, Qt.blue)
            palette.setColor(QPalette.HighlightedText, Qt.white)

            self.setPalette(palette)
            logging.debug("Dark theme applied successfully.")
        except Exception as e:
            logging.error(f"Error applying dark theme: {e}")

    def on_service_changed(self, service_name):
        """
        Called when the user picks a different service.
        We update the model combo box accordingly.
        Also enable/disable the fetch button if the service supports it.
        """
        self.model_combo.clear()
        service_cfg = SERVICE_CONFIG.get(service_name, {})
        if not service_cfg:
            return

        if service_cfg["enable_model_fetch"]:
            # We might want to let user click "Fetch Models" for Ollama
            self.fetch_models_button.setEnabled(True)
            self.model_combo.setEnabled(False)
        else:
            # Hardcode model list if provided
            self.fetch_models_button.setEnabled(False)
            models = service_cfg.get("hardcoded_models", [])
            self.model_combo.addItems(models)
            self.model_combo.setEnabled(True)

    def fetch_models(self):
        """
        For Ollama (or any service that has an endpoint for model listing).
        This is disabled for other services.
        """
        service_name = self.service_combo.currentText()
        service_cfg = SERVICE_CONFIG.get(service_name, {})
        if not service_cfg or not service_cfg.get("enable_model_fetch"):
            self.response_display.append("No model fetch supported for this service.")
            return

        fetch_url = service_cfg["fetch_models_url"]
        api_key = os.environ.get(service_cfg.get("env_var")) if service_cfg.get("api_key_needed") else None

        self.model_combo.setEnabled(False)
        self.send_button.setEnabled(False)
        self.response_display.append(f"Fetching available models from {service_name}...")
        self.fetch_worker = FetchModelsWorker(fetch_url=fetch_url, api_key=api_key)
        self.fetch_worker.models_fetched.connect(self.populate_models)
        self.fetch_worker.error_occurred.connect(self.display_error)
        self.fetch_worker.start()

    def populate_models(self, models):
        """
        Called when the FetchModelsWorker has retrieved a list of models.
        """
        self.model_combo.clear()
        if models:
            self.model_combo.addItems(models)
            self.response_display.append("Models loaded successfully.")
            logging.debug(f"Populated models: {models}")
        else:
            self.response_display.append("No models available.")
            logging.warning("No models were fetched from the server.")
        self.model_combo.setEnabled(True)
        self.send_button.setEnabled(True)

    def display_error(self, message):
        self.response_display.append(f"Error: {message}")
        logging.error(f"Displayed error to user: {message}")
        self.model_combo.setEnabled(True)
        self.send_button.setEnabled(True)

    def send_prompt(self):
        prompt = self.prompt_input.text().strip()
        if not prompt:
            self.response_display.append("Please enter a prompt.")
            logging.warning("User attempted to send an empty prompt.")
            return

        selected_service = self.service_combo.currentText()
        selected_model = self.model_combo.currentText()

        if not selected_model:
            self.response_display.append("Please select a model.")
            logging.warning("User attempted to send a prompt without selecting a model.")
            return

        # Disable UI elements to prevent multiple requests
        self.send_button.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.response_display.append(f"\n> [{selected_service}/{selected_model}] {prompt}\n")
        self.prompt_input.clear()

        # Start the GenerateWorker thread
        self.generate_worker = GenerateWorker(
            service_name=selected_service,
            model=selected_model,
            prompt=prompt,
            context=self.context
        )
        self.generate_worker.response_received.connect(self.update_response)
        self.generate_worker.generation_complete.connect(self.generation_done)
        self.generate_worker.error_occurred.connect(self.display_error)
        self.generate_worker.start()
        logging.debug(f"Started GenerateWorker for service '{selected_service}', model '{selected_model}'.")

    def update_response(self, text):
        try:
            self.response_display.moveCursor(QTextCursor.End)
            self.response_display.insertPlainText(text)
            self.response_display.moveCursor(QTextCursor.End)
            logging.debug(f"Updated response display with text: {text}")
        except Exception as e:
            error_message = f"Error updating response display: {e}"
            self.display_error(error_message)

    def generation_done(self, data):
        """
        Once generation is complete, you can store new context,
        display usage stats, etc. The content of 'data' depends on the service.
        """
        try:
            # For Ollama, the response might contain a context key you can store for conversation
            new_context = data.get('context')
            if new_context:
                self.context = new_context
                logging.debug(f"Updated conversation context: {self.context}")

            # If your service returns usage stats
            prompt_tokens = data.get('prompt_tokens', None)
            completion_tokens = data.get('completion_tokens', None)
            total_tokens = data.get('total_tokens', None)
            if total_tokens is not None:
                self.response_display.append(
                    f"\nPrompt tokens: {prompt_tokens}, "
                    f"Completion tokens: {completion_tokens}, "
                    f"Total tokens: {total_tokens}"
                )

            # (Ollama-specific) If it returns eval_count/eval_duration, you can compute tokens per second
            if 'eval_count' in data and 'eval_duration' in data:
                eval_count = data['eval_count']
                eval_duration_ns = data['eval_duration'] or 1
                tokens_per_second = eval_count / eval_duration_ns * 1e9
                self.response_display.append(
                    f"\nTokens per second: {tokens_per_second:.2f} tokens/s"
                )

            self.response_display.append("\nGeneration completed.\n")
            logging.info(f"Generation completed with data: {data}")

            # Re-enable UI elements
            self.send_button.setEnabled(True)
            self.model_combo.setEnabled(True)
        except Exception as e:
            error_message = f"Error processing generation completion: {e}"
            self.display_error(error_message)

###############################################################################
# Entry point of the application
###############################################################################
def main():
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        logging.info("Multi-Service Client application started successfully.")
        sys.exit(app.exec())
    except Exception as e:
        logging.critical(f"Application failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

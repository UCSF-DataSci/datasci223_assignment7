# Part 2: LLM API Usage (Command Line Chat Tool)

## Introduction

In this part, you'll create a command line tool that interacts with a Large Language Model (LLM) API. This tool will allow users to have conversations with an LLM, focusing on healthcare-related queries about gout. You'll learn how to connect to LLM APIs, handle responses, and create a user-friendly interface using the Gout Emergency Department Chief Complaint Corpora.

## Learning Objectives

- Connect to an LLM API (e.g., Hugging Face)
- Implement a command line interface for chat interactions
- Handle API errors and rate limits gracefully
- Process and format LLM responses
- Create a reusable utility for LLM interactions

## Setup and Installation

```python
# Install required packages
%pip install -r requirements.txt

# Additional packages for LLM API interaction
%pip install requests transformers huggingface_hub

# Import necessary libraries
import os
import sys
import json
import requests
import argparse
from typing import List, Dict, Any, Optional
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create directories
os.makedirs('utils', exist_ok=True)
os.makedirs('results/part_2', exist_ok=True)
```

## 1. Dataset and LLM API Options

```python
# First, let's set up the Gout Emergency Department dataset
import os
import pandas as pd
import json

# Create data directory if it doesn't exist
os.makedirs('data/gout_emergency', exist_ok=True)

# Function to download the dataset
def download_gout_dataset():
    """
    Download the Gout Emergency Department Chief Complaint Corpora
    
    Note: You need to manually download this dataset from PhysioNet:
    https://physionet.org/content/emer-complaint-gout/
    
    After downloading, place the files in the data/gout_emergency directory
    """
    gout_data_path = 'data/gout_emergency/chief_complaints.csv'
    
    if os.path.exists(gout_data_path):
        print(f"Loading Gout Emergency dataset from {gout_data_path}")
        return pd.read_csv(gout_data_path)
    else:
        print(f"Gout Emergency dataset not found at {gout_data_path}")
        print("Please download the dataset from PhysioNet:")
        print("https://physionet.org/content/emer-complaint-gout/")
        print("After downloading, place the files in the data/gout_emergency directory")
        return None

# Try to load the dataset
gout_df = download_gout_dataset()

# If the dataset is loaded successfully, display some information
if gout_df is not None:
    print(f"Dataset shape: {gout_df.shape}")
    print("\nSample complaints:")
    for i, complaint in enumerate(gout_df['chief_complaint'].head(5)):
        print(f"{i+1}. {complaint}")

# Now let's explore different LLM API options that are free or low-cost

# Option 1: Hugging Face Inference API
# Pros: Free tier available, many models to choose from
# Cons: Limited rate, smaller models than commercial options

# Option 2: Local models with Hugging Face Transformers
# Pros: No API costs, full control
# Cons: Requires more computational resources, limited to smaller models

# Option 3: OpenAI API (not free, but included for comparison)
# Pros: State-of-the-art models
# Cons: Requires payment

# For this assignment, we'll focus on the Hugging Face Inference API
# as it provides a good balance of accessibility and capability

# Let's explore some suitable models on Hugging Face
huggingface_models = [
    "google/flan-t5-small",       # 80M parameters, good for simple Q&A
    "google/flan-t5-base",        # 250M parameters, better quality
    "google/flan-t5-large",       # 780M parameters, good quality but slower
    "facebook/opt-350m",          # 350M parameters, general purpose
    "facebook/opt-1.3b",          # 1.3B parameters, better quality
    "EleutherAI/gpt-neo-125m",    # 125M parameters, GPT-style model
    "bigscience/bloom-560m",      # 560M parameters, multilingual
]

print("\nAvailable models for free use with Hugging Face:")
for i, model in enumerate(huggingface_models):
    print(f"{i+1}. {model}")
```

## 2. Implementing the LLM API Client

```python
# Let's implement a client for the Hugging Face Inference API

class LLMClient:
    """Client for interacting with LLM APIs"""
    
    def __init__(
        self, 
        model_name: str = "google/flan-t5-base",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 30
    ):
        """
        Initialize the LLM client
        
        Args:
            model_name: Name of the model to use
            api_key: API key for authentication (optional for some models)
            max_retries: Maximum number of retries on failure
            timeout: Timeout for API requests in seconds
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("HUGGINGFACE_API_KEY")
        self.max_retries = max_retries
        self.timeout = timeout
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        
        logger.info(f"Initialized LLM client with model: {model_name}")
    
    def generate_text(self, prompt: str) -> str:
        """
        Generate text from the LLM based on the prompt
        
        Args:
            prompt: The input prompt for the LLM
            
        Returns:
            The generated text response
        """
        payload = {"inputs": prompt}
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return response.json()[0]["generated_text"]
                elif response.status_code == 429:
                    # Rate limit exceeded
                    wait_time = int(response.headers.get("Retry-After", 30))
                    logger.warning(f"Rate limit exceeded. Waiting {wait_time} seconds.")
                    time.sleep(wait_time)
                else:
                    logger.error(f"API request failed with status code {response.status_code}: {response.text}")
                    time.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                logger.error(f"Error during API request: {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return "I'm sorry, I couldn't generate a response due to technical difficulties."
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a response based on a conversation history
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            The generated response
        """
        # Format the conversation history into a prompt
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
        
        prompt += "Assistant: "
        
        # Generate response
        response = self.generate_text(prompt)
        
        return response

# Test the LLM client with a simple prompt
def test_llm_client():
    # Create client (no API key needed for testing)
    client = LLMClient(model_name="google/flan-t5-small")
    
    # Test with a simple healthcare question
    messages = [
        {"role": "system", "content": "You are a helpful healthcare assistant."},
        {"role": "user", "content": "What are the symptoms of diabetes?"}
    ]
    
    print("Testing LLM client with a healthcare question...")
    response = client.chat(messages)
    print(f"Response: {response}")
    
    return client

# Uncomment to test
# test_client = test_llm_client()
```

## 3. Building the Command Line Interface

```python
# Now let's build a command line interface for our LLM chat tool

class LLMChatTool:
    """Command line tool for chatting with an LLM"""
    
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        api_key: Optional[str] = None,
        system_prompt: str = "You are a helpful healthcare assistant that provides accurate, evidence-based information."
    ):
        """
        Initialize the chat tool
        
        Args:
            model_name: Name of the model to use
            api_key: API key for authentication
            system_prompt: System prompt to guide the LLM's behavior
        """
        self.client = LLMClient(model_name=model_name, api_key=api_key)
        self.conversation = [{"role": "system", "content": system_prompt}]
        self.system_prompt = system_prompt
        
    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation = [{"role": "system", "content": self.system_prompt}]
        print("Conversation has been reset.")
    
    def add_message(self, content: str, role: str = "user"):
        """Add a message to the conversation history"""
        self.conversation.append({"role": role, "content": content})
    
    def get_response(self) -> str:
        """Get a response from the LLM based on the conversation history"""
        response = self.client.chat(self.conversation)
        self.add_message(response, role="assistant")
        return response
    
    def run_interactive(self):
        """Run the chat tool in interactive mode"""
        print(f"LLM Chat Tool - Model: {self.client.model_name}")
        print("Type 'exit' to quit or 'reset' to start a new conversation.")
        print("=" * 50)
        
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            
            if user_input.lower() == "reset":
                self.reset_conversation()
                continue
            
            self.add_message(user_input)
            
            print("\nAssistant: ", end="")
            sys.stdout.flush()  # Ensure the prompt is displayed immediately
            
            response = self.get_response()
            print(response)

# Create a function to save the tool as a Python script
def save_chat_tool():
    """Save the LLM chat tool as a Python script"""
    script_content = '''#!/usr/bin/env python3
# llm_chat.py - A command line tool for chatting with an LLM

import os
import sys
import json
import requests
import argparse
from typing import List, Dict, Any, Optional
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLMClient:
    """Client for interacting with LLM APIs"""
    
    def __init__(
        self, 
        model_name: str = "google/flan-t5-base",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 30
    ):
        """
        Initialize the LLM client
        
        Args:
            model_name: Name of the model to use
            api_key: API key for authentication (optional for some models)
            max_retries: Maximum number of retries on failure
            timeout: Timeout for API requests in seconds
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("HUGGINGFACE_API_KEY")
        self.max_retries = max_retries
        self.timeout = timeout
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        
        logger.info(f"Initialized LLM client with model: {model_name}")
    
    def generate_text(self, prompt: str) -> str:
        """
        Generate text from the LLM based on the prompt
        
        Args:
            prompt: The input prompt for the LLM
            
        Returns:
            The generated text response
        """
        payload = {"inputs": prompt}
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return response.json()[0]["generated_text"]
                elif response.status_code == 429:
                    # Rate limit exceeded
                    wait_time = int(response.headers.get("Retry-After", 30))
                    logger.warning(f"Rate limit exceeded. Waiting {wait_time} seconds.")
                    time.sleep(wait_time)
                else:
                    logger.error(f"API request failed with status code {response.status_code}: {response.text}")
                    time.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                logger.error(f"Error during API request: {str(e)}")
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return "I'm sorry, I couldn't generate a response due to technical difficulties."
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a response based on a conversation history
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            The generated response
        """
        # Format the conversation history into a prompt
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                prompt += f"System: {content}\\n\\n"
            elif role == "user":
                prompt += f"User: {content}\\n\\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\\n\\n"
        
        prompt += "Assistant: "
        
        # Generate response
        response = self.generate_text(prompt)
        
        return response

class LLMChatTool:
    """Command line tool for chatting with an LLM"""
    
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        api_key: Optional[str] = None,
        system_prompt: str = "You are a helpful healthcare assistant that provides accurate, evidence-based information."
    ):
        """
        Initialize the chat tool
        
        Args:
            model_name: Name of the model to use
            api_key: API key for authentication
            system_prompt: System prompt to guide the LLM's behavior
        """
        self.client = LLMClient(model_name=model_name, api_key=api_key)
        self.conversation = [{"role": "system", "content": system_prompt}]
        self.system_prompt = system_prompt
        
    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation = [{"role": "system", "content": self.system_prompt}]
        print("Conversation has been reset.")
    
    def add_message(self, content: str, role: str = "user"):
        """Add a message to the conversation history"""
        self.conversation.append({"role": role, "content": content})
    
    def get_response(self) -> str:
        """Get a response from the LLM based on the conversation history"""
        response = self.client.chat(self.conversation)
        self.add_message(response, role="assistant")
        return response
    
    def run_interactive(self):
        """Run the chat tool in interactive mode"""
        print(f"LLM Chat Tool - Model: {self.client.model_name}")
        print("Type 'exit' to quit or 'reset' to start a new conversation.")
        print("=" * 50)
        
        while True:
            user_input = input("\\nYou: ")
            
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            
            if user_input.lower() == "reset":
                self.reset_conversation()
                continue
            
            self.add_message(user_input)
            
            print("\\nAssistant: ", end="")
            sys.stdout.flush()  # Ensure the prompt is displayed immediately
            
            response = self.get_response()
            print(response)

def main():
    """Main function to run the chat tool"""
    parser = argparse.ArgumentParser(description="LLM Chat Tool - A command line interface for chatting with an LLM")
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="google/flan-t5-base",
        help="Name of the Hugging Face model to use"
    )
    
    parser.add_argument(
        "--api-key", 
        type=str, 
        default=None,
        help="API key for Hugging Face (optional, can also use HUGGINGFACE_API_KEY env var)"
    )
    
    parser.add_argument(
        "--system-prompt", 
        type=str, 
        default="You are a helpful healthcare assistant that provides accurate, evidence-based information.",
        help="System prompt to guide the LLM's behavior"
    )
    
    args = parser.parse_args()
    
    chat_tool = LLMChatTool(
        model_name=args.model,
        api_key=args.api_key,
        system_prompt=args.system_prompt
    )
    
    try:
        chat_tool.run_interactive()
    except KeyboardInterrupt:
        print("\\nGoodbye!")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    # Save the script
    with open('utils/llm_chat.py', 'w') as f:
        f.write(script_content)
    
    # Make the script executable
    os.chmod('utils/llm_chat.py', 0o755)
    
    print("Chat tool saved to utils/llm_chat.py")

# Save the chat tool
save_chat_tool()
```

## 4. Testing and Usage Examples

```python
# Let's create some usage examples for our chat tool

def create_usage_examples():
    """Create usage examples for the LLM chat tool"""
    
    examples = '''# LLM Chat Tool Usage Examples

## Basic Usage

Run the chat tool with default settings:

```bash
python utils/llm_chat.py
```

## Specifying a Different Model

Use a smaller model for faster responses:

```bash
python utils/llm_chat.py --model google/flan-t5-small
```

Use a larger model for better quality:

```bash
python utils/llm_chat.py --model google/flan-t5-large
```

## Custom System Prompt

Specify a custom system prompt to guide the LLM's behavior:

```bash
python utils/llm_chat.py --system-prompt "You are a medical expert specializing in cardiology. Provide detailed, technical answers to questions about heart health."
```

## Using an API Key

If you have a Hugging Face API key, you can provide it:

```bash
python utils/llm_chat.py --api-key YOUR_API_KEY
```

Or set it as an environment variable:

```bash
export HUGGINGFACE_API_KEY=YOUR_API_KEY
python utils/llm_chat.py
```

## Example Conversations

Here are some example conversations you can have with the tool:

### General Health Questions

```
You: What are the symptoms of type 2 diabetes?
Assistant: Common symptoms of type 2 diabetes include increased thirst, frequent urination, hunger, fatigue, blurred vision, slow-healing sores, and frequent infections. Many people with type 2 diabetes may not experience symptoms initially.

You: How is it diagnosed?
Assistant: Type 2 diabetes is diagnosed through blood tests that measure blood glucose levels. The main tests include fasting blood glucose test, A1C test (which shows average blood sugar over 2-3 months), and oral glucose tolerance test. A diagnosis is typically confirmed when fasting blood glucose is ≥126 mg/dL, A1C is ≥6.5%, or glucose tolerance test shows blood glucose ≥200 mg/dL after 2 hours.
```

### Medication Information

```
You: What are the side effects of metformin?
Assistant: Common side effects of metformin include gastrointestinal issues like nausea, vomiting, diarrhea, stomach pain, and loss of appetite. These often improve over time. Less common side effects include metallic taste, vitamin B12 deficiency, and lactic acidosis (rare but serious). Taking metformin with meals can help reduce gastrointestinal side effects.

You: Can it interact with other medications?
Assistant: Yes, metformin can interact with several medications. Notable interactions include with contrast dyes used in some imaging procedures, which may increase risk of kidney problems. It can also interact with certain diabetes medications, potentially causing low blood sugar. Some heart medications, diuretics, corticosteroids, and certain antibiotics may also interact with metformin. Always inform your healthcare provider about all medications you're taking.
```

### Preventive Health

```
You: What are the recommended cancer screenings for someone in their 50s?
Assistant: For people in their 50s, recommended cancer screenings typically include:

1. Colorectal cancer: Regular screening starting at age 45-50 with colonoscopy every 10 years or alternatives like stool-based tests annually
2. Breast cancer (women): Mammogram every 1-2 years
3. Cervical cancer (women): Pap test every 3 years or HPV test every 5 years
4. Prostate cancer (men): Discuss PSA testing with your doctor
5. Lung cancer: Annual low-dose CT scan if you have a significant smoking history
6. Skin cancer: Regular skin examinations

These recommendations may vary based on personal and family medical history. Consult with your healthcare provider for personalized screening recommendations.
```

## Interactive Mode

In interactive mode, you can have a continuous conversation with the LLM. Type 'reset' to start a new conversation or 'exit' to quit.

```
You: I've been experiencing headaches and fatigue lately. What could be causing this?
Assistant: Headaches and fatigue can have many causes. Common ones include stress, dehydration, lack of sleep, poor posture, eye strain, anemia, or caffeine withdrawal. More serious causes might include infections, medication side effects, or chronic conditions like fibromyalgia. If symptoms persist or are severe, please consult a healthcare provider for proper evaluation.

You: reset
Conversation has been reset.

You: What's the recommended daily water intake?
Assistant: The recommended daily water intake is generally about 8 cups (64 ounces) of water per day, but this can vary based on factors like your size, activity level, climate, and overall health. The Institute of Medicine suggests about 3.7 liters (125 ounces) daily for men and 2.7 liters (91 ounces) for women, including water from all beverages and foods.
```
'''
    
    # Save the examples
    with open('results/part_2/usage_examples.txt', 'w') as f:
        f.write(examples)
    
    print("Usage examples saved to results/part_2/usage_examples.txt")

# Create usage examples
create_usage_examples()
```

## 5. Optimizing for Healthcare Queries

```python
# Let's create a specialized version of our chat tool for healthcare queries

def create_healthcare_specialized_tool():
    """Create a specialized version of the chat tool for healthcare queries"""
    
    healthcare_prompt = '''You are a helpful healthcare assistant that provides accurate, evidence-based information. Follow these guidelines:

1. Provide factual, scientifically accurate information based on current medical knowledge
2. Clearly indicate when information is general advice versus medical guidance
3. Remind users to consult healthcare professionals for personal medical advice
4. Avoid making definitive diagnoses or treatment recommendations
5. Use plain language and explain medical terminology
6. Cite sources or mention the general consensus when appropriate
7. Be honest about limitations of knowledge
8. Focus on providing educational information about health conditions, treatments, and preventive care

Remember that your purpose is to inform and educate, not to replace professional medical care.'''
    
    script_content = '''#!/usr/bin/env python3
# healthcare_chat.py - A specialized chat tool for healthcare queries

import os
import sys
from llm_chat import LLMClient, LLMChatTool
import argparse

HEALTHCARE_PROMPT = """You are a helpful healthcare assistant that provides accurate, evidence-based information. Follow these guidelines:

1. Provide factual, scientifically accurate information based on current medical knowledge
2. Clearly indicate when information is general advice versus medical guidance
3. Remind users to consult healthcare professionals for personal medical advice
4. Avoid making definitive diagnoses or treatment recommendations
5. Use plain language and explain medical terminology
6. Cite sources or mention the general consensus when appropriate
7. Be honest about limitations of knowledge
8. Focus on providing educational information about health conditions, treatments, and preventive care

Remember that your purpose is to inform and educate, not to replace professional medical care."""

def main():
    """Main function to run the healthcare chat tool"""
    parser = argparse.ArgumentParser(description="Healthcare Chat Tool - A specialized chat interface for healthcare queries")
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="google/flan-t5-large",  # Using a larger model for better healthcare knowledge
        help="Name of the Hugging Face model to use"
    )
    
    parser.add_argument(
        "--api-key", 
        type=str, 
        default=None,
        help="API key for Hugging Face (optional, can also use HUGGINGFACE_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    print("Healthcare Chat Tool - Specialized for medical and health information")
    print("NOTE: This tool provides general information only and is not a substitute for professional medical advice.")
    print("=" * 80)
    
    chat_tool = LLMChatTool(
        model_name=args.model,
        api_key=args.api_key,
        system_prompt=HEALTHCARE_PROMPT
    )
    
    try:
        chat_tool.run_interactive()
    except KeyboardInterrupt:
        print("\\nGoodbye!")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    # Save the script
    with open('utils/healthcare_chat.py', 'w') as f:
        f.write(script_content)
    
    # Make the script executable
    os.chmod('utils/healthcare_chat.py', 0o755)
    
    print("Healthcare specialized chat tool saved to utils/healthcare_chat.py")

# Create the healthcare specialized tool
create_healthcare_specialized_tool()
```

## Progress Checkpoints

1. **API Exploration**:
   - [ ] Research available LLM API options
   - [ ] Identify free or low-cost options
   - [ ] Select appropriate model for healthcare queries

2. **Client Implementation**:
   - [ ] Implement API client with error handling
   - [ ] Test basic text generation
   - [ ] Implement conversation formatting

3. **Command Line Interface**:
   - [ ] Create interactive chat interface
   - [ ] Implement conversation management
   - [ ] Add command line arguments

4. **Tool Creation**:
   - [ ] Save implementation as executable script
   - [ ] Document usage examples
   - [ ] Create healthcare-specialized version

5. **Testing and Optimization**:
   - [ ] Test with various healthcare queries
   - [ ] Optimize response formatting
   - [ ] Ensure error handling works properly

## Common Issues and Solutions

1. **API Access Issues**:
   - Problem: Rate limiting
   - Solution: Implement exponential backoff and retry logic
   - Problem: Authentication errors
   - Solution: Verify API key and environment variables

2. **Response Quality Issues**:
   - Problem: Irrelevant or generic responses
   - Solution: Improve system prompt and use larger models
   - Problem: Inconsistent formatting
   - Solution: Post-process responses or use structured output prompts

3. **Performance Issues**:
   - Problem: Slow response times
   - Solution: Use smaller models or implement caching
   - Problem: High API costs
   - Solution: Optimize prompt length and use free tier models

4. **Ethical Considerations**:
   - Problem: Medical advice liability
   - Solution: Add clear disclaimers and focus on educational content
   - Problem: Privacy concerns
   - Solution: Don't store or log sensitive user queries
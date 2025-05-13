# Part 3: Structured Response (Few-shot Learning with LLMs)

## Introduction

In this part, you'll explore how to use 0-shot, 1-shot, and few-shot learning techniques with Large Language Models (LLMs) to generate structured responses for healthcare applications. You'll implement different prompting strategies using the Gout Emergency Department Chief Complaint Corpora, compare their performance, and analyze how prompt design affects response quality and structure.

## Learning Objectives

- Understand 0-shot, 1-shot, and few-shot learning approaches
- Design effective prompts for structured outputs
- Implement different prompting strategies
- Evaluate and compare prompt performance
- Create a reusable utility for structured LLM responses

## Setup and Installation

```python
# Install required packages
%pip install -r requirements.txt

# Additional packages for LLM API interaction and evaluation
%pip install requests transformers huggingface_hub pandas matplotlib seaborn

# Import necessary libraries
import os
import sys
import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Union, Tuple
import time
import logging
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create directories
os.makedirs('utils', exist_ok=True)
os.makedirs('results/part_3', exist_ok=True)
```

## 1. Loading the Gout Emergency Dataset

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
    
    # Check if the dataset has the expected columns
    print("\nDataset columns:")
    print(gout_df.columns.tolist())
```

## 2. Understanding Few-shot Learning

```python
# Let's explore the concepts of 0-shot, 1-shot, and few-shot learning

# 0-shot learning: The model performs a task without any examples
# Example: "Classify this text as positive or negative: 'I love this product!'"

# 1-shot learning: The model is given one example before performing the task
# Example: "Classify the sentiment. 'I love this product!' is positive. 'I hate this product!' is ?"

# Few-shot learning: The model is given multiple examples before performing the task
# Example: "Classify the sentiment. 'I love this product!' is positive. 'This is terrible.' is negative. 'I hate this product!' is ?"

# Let's define a healthcare task for our few-shot learning experiments
# Task: Classify medical symptoms into body systems (cardiovascular, respiratory, digestive, etc.)

# Define our task description
task_description = """
Task: Classify medical symptoms into the appropriate body system category.
Categories: Cardiovascular, Respiratory, Digestive, Neurological, Musculoskeletal, Dermatological, Endocrine, Urinary

Output format: 
{
  "symptom": "the symptom text",
  "category": "the body system category",
  "confidence": "high/medium/low",
  "explanation": "brief explanation of the classification"
}
"""

print("Task Description:")
print(task_description)
## 3. Implementing the Structured Response Utility

```python
# Let's implement a utility for generating structured responses using different prompting strategies

class StructuredResponseGenerator:
    """Utility for generating structured responses using different prompting strategies"""
    
    def __init__(
        self, 
        model_name: str = "google/flan-t5-base",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 30
    ):
        """
        Initialize the structured response generator
        
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
        
        logger.info(f"Initialized structured response generator with model: {model_name}")
    
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
        
        return "Error: Could not generate a response due to technical difficulties."
    
    def zero_shot_prompt(self, task_description: str, input_text: str) -> str:
        """
        Create a 0-shot prompt for the given task and input
        
        Args:
            task_description: Description of the task
            input_text: Input text to process
            
        Returns:
            Formatted 0-shot prompt
        """
        return f"{task_description}\n\nInput: {input_text}\n\nOutput:"
    
    def one_shot_prompt(self, task_description: str, example_input: str, 
                        example_output: str, input_text: str) -> str:
        """
        Create a 1-shot prompt with one example
        
        Args:
            task_description: Description of the task
            example_input: Input text for the example
            example_output: Expected output for the example
            input_text: Input text to process
            
        Returns:
            Formatted 1-shot prompt
        """
        return f"{task_description}\n\nExample:\nInput: {example_input}\nOutput: {example_output}\n\nInput: {input_text}\n\nOutput:"
    
    def few_shot_prompt(self, task_description: str, examples: List[Tuple[str, str]], 
                        input_text: str) -> str:
        """
        Create a few-shot prompt with multiple examples
        
        Args:
            task_description: Description of the task
            examples: List of (input, output) tuples for examples
            input_text: Input text to process
            
        Returns:
            Formatted few-shot prompt
        """
        examples_text = ""
        for i, (example_input, example_output) in enumerate(examples):
            examples_text += f"Example {i+1}:\nInput: {example_input}\nOutput: {example_output}\n\n"
        
        return f"{task_description}\n\n{examples_text}Input: {input_text}\n\nOutput:"
    
    def generate_structured_response(
        self, 
        task_description: str,
        input_text: str,
        prompt_type: str = "zero-shot",
        examples: Optional[List[Tuple[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate a structured response using the specified prompting strategy
        
        Args:
            task_description: Description of the task
            input_text: Input text to process
            prompt_type: Type of prompt to use ("zero-shot", "one-shot", or "few-shot")
            examples: List of (input, output) tuples for examples (required for one-shot and few-shot)
            
        Returns:
            Parsed structured response as a dictionary
        """
        if prompt_type == "zero-shot":
            prompt = self.zero_shot_prompt(task_description, input_text)
        elif prompt_type == "one-shot":
            if not examples or len(examples) < 1:
                raise ValueError("At least one example is required for one-shot prompting")
            prompt = self.one_shot_prompt(task_description, examples[0][0], examples[0][1], input_text)
        elif prompt_type == "few-shot":
            if not examples or len(examples) < 2:
                raise ValueError("At least two examples are required for few-shot prompting")
            prompt = self.few_shot_prompt(task_description, examples, input_text)
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        
        # Generate response
        response_text = self.generate_text(prompt)
        
        # Try to parse the response as JSON
        try:
            # First, try to find JSON-like structure in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                # If no JSON structure found, return as raw text
                return {"raw_response": response_text}
        except json.JSONDecodeError:
            # If JSON parsing fails, return as raw text
            return {"raw_response": response_text}

# Create an instance of the structured response generator
generator = StructuredResponseGenerator(model_name="google/flan-t5-base")
```
## 4. Defining Examples for Few-shot Learning

```python
# Let's define some examples for our medical symptom classification task

# Define the task description
symptom_classification_task = """
Task: Classify medical symptoms into the appropriate body system category.
Categories: Cardiovascular, Respiratory, Digestive, Neurological, Musculoskeletal, Dermatological, Endocrine, Urinary

Output format: 
{
  "symptom": "the symptom text",
  "category": "the body system category",
  "confidence": "high/medium/low",
  "explanation": "brief explanation of the classification"
}
"""

# Define examples for few-shot learning
symptom_examples = [
    # Example 1
    (
        "chest pain radiating to the left arm",
        """
{
  "symptom": "chest pain radiating to the left arm",
  "category": "Cardiovascular",
  "confidence": "high",
  "explanation": "Pain radiating from the chest to the left arm is a classic symptom of cardiac issues, particularly heart attacks."
}
        """
    ),
    # Example 2
    (
        "shortness of breath and wheezing",
        """
{
  "symptom": "shortness of breath and wheezing",
  "category": "Respiratory",
  "confidence": "high",
  "explanation": "Wheezing and difficulty breathing are typical respiratory symptoms, often seen in asthma, COPD, or respiratory infections."
}
        """
    ),
    # Example 3
    (
        "severe abdominal pain and nausea",
        """
{
  "symptom": "severe abdominal pain and nausea",
  "category": "Digestive",
  "confidence": "high",
  "explanation": "Abdominal pain and nausea are common digestive system symptoms that can indicate various gastrointestinal issues."
}
        """
    ),
    # Example 4
    (
        "persistent headache and dizziness",
        """
{
  "symptom": "persistent headache and dizziness",
  "category": "Neurological",
  "confidence": "medium",
  "explanation": "Headaches and dizziness are primarily neurological symptoms, though they can also be caused by other systems."
}
        """
    ),
    # Example 5
    (
        "joint pain and stiffness",
        """
{
  "symptom": "joint pain and stiffness",
  "category": "Musculoskeletal",
  "confidence": "high",
  "explanation": "Joint pain and stiffness are classic musculoskeletal symptoms, often seen in arthritis and other joint disorders."
}
        """
    )
]

# Print the first example
print("Example 1:")
print(f"Input: {symptom_examples[0][0]}")
print(f"Output: {symptom_examples[0][1]}")
```
## 5. Comparing Prompting Strategies

```python
# Let's compare different prompting strategies on a set of test symptoms

# Define test symptoms
test_symptoms = [
    "rapid heartbeat and palpitations",
    "persistent cough with yellow sputum",
    "acid reflux and heartburn",
    "numbness and tingling in extremities",
    "lower back pain when bending",
    "itchy rash with red bumps",
    "excessive thirst and frequent urination",
    "painful urination and urgency"
]

# Define expected categories for evaluation
expected_categories = [
    "Cardiovascular",
    "Respiratory",
    "Digestive",
    "Neurological",
    "Musculoskeletal",
    "Dermatological",
    "Endocrine",
    "Urinary"
]

# Function to evaluate responses
def evaluate_responses(responses, expected_categories):
    """Evaluate the quality of responses against expected categories"""
    results = {
        "accuracy": 0,
        "structured_format": 0,
        "complete_fields": 0
    }
    
    correct_count = 0
    structured_count = 0
    complete_count = 0
    
    for i, (response, expected) in enumerate(zip(responses, expected_categories)):
        # Check if response is structured (has category field)
        if isinstance(response, dict) and "category" in response:
            structured_count += 1
            
            # Check if all required fields are present
            required_fields = ["symptom", "category", "confidence", "explanation"]
            if all(field in response for field in required_fields):
                complete_count += 1
            
            # Check if category matches expected
            if response["category"].lower() == expected.lower():
                correct_count += 1
    
    total = len(responses)
    results["accuracy"] = correct_count / total if total > 0 else 0
    results["structured_format"] = structured_count / total if total > 0 else 0
    results["complete_fields"] = complete_count / total if total > 0 else 0
    
    return results

# Test different prompting strategies
def test_prompting_strategies():
    """Test and compare different prompting strategies"""
    strategies = ["zero-shot", "one-shot", "few-shot"]
    results = {}
    
    for strategy in strategies:
        print(f"\nTesting {strategy} prompting strategy...")
        responses = []
        
        for symptom in test_symptoms:
            if strategy == "zero-shot":
                response = generator.generate_structured_response(
                    symptom_classification_task,
                    symptom,
                    prompt_type="zero-shot"
                )
            elif strategy == "one-shot":
                response = generator.generate_structured_response(
                    symptom_classification_task,
                    symptom,
                    prompt_type="one-shot",
                    examples=[symptom_examples[0]]
                )
            else:  # few-shot
                response = generator.generate_structured_response(
                    symptom_classification_task,
                    symptom,
                    prompt_type="few-shot",
                    examples=symptom_examples
                )
            
            responses.append(response)
            print(f"Symptom: {symptom}")
            print(f"Response: {json.dumps(response, indent=2)}")
            print("-" * 40)
        
        # Evaluate responses
        evaluation = evaluate_responses(responses, expected_categories)
        results[strategy] = evaluation
        print(f"{strategy} evaluation: {json.dumps(evaluation, indent=2)}")
    
    return results

# Run the comparison
comparison_results = test_prompting_strategies()

# Visualize the results
def plot_comparison(results):
    """Plot the comparison of different prompting strategies"""
    strategies = list(results.keys())
    metrics = ["accuracy", "structured_format", "complete_fields"]
    
    data = []
    for strategy in strategies:
        for metric in metrics:
            data.append({
                "Strategy": strategy,
                "Metric": metric,
                "Value": results[strategy][metric]
            })
    
## 6. Creating a Reusable Utility

```python
# Let's create a reusable utility for structured responses

def save_structured_response_utility():
    """Save the structured response utility as a Python script"""
    
    script_content = '''#!/usr/bin/env python3
# structured_response.py - A utility for generating structured responses from LLMs

import os
import sys
import json
import requests
import re
import time
import logging
from typing import List, Dict, Any, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StructuredResponseGenerator:
    """Utility for generating structured responses using different prompting strategies"""
    
    def __init__(
        self, 
        model_name: str = "google/flan-t5-base",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 30
    ):
        """
        Initialize the structured response generator
        
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
        
        logger.info(f"Initialized structured response generator with model: {model_name}")
    
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
        
        return "Error: Could not generate a response due to technical difficulties."
    
    def zero_shot_prompt(self, task_description: str, input_text: str) -> str:
        """
        Create a 0-shot prompt for the given task and input
        
        Args:
            task_description: Description of the task
            input_text: Input text to process
            
        Returns:
            Formatted 0-shot prompt
        """
        return f"{task_description}\\n\\nInput: {input_text}\\n\\nOutput:"
    
    def one_shot_prompt(self, task_description: str, example_input: str, 
                        example_output: str, input_text: str) -> str:
        """
        Create a 1-shot prompt with one example
        
        Args:
            task_description: Description of the task
            example_input: Input text for the example
            example_output: Expected output for the example
            input_text: Input text to process
            
        Returns:
            Formatted 1-shot prompt
        """
        return f"{task_description}\\n\\nExample:\\nInput: {example_input}\\nOutput: {example_output}\\n\\nInput: {input_text}\\n\\nOutput:"
    
    def few_shot_prompt(self, task_description: str, examples: List[Tuple[str, str]], 
                        input_text: str) -> str:
        """
        Create a few-shot prompt with multiple examples
        
        Args:
            task_description: Description of the task
            examples: List of (input, output) tuples for examples
            input_text: Input text to process
            
        Returns:
            Formatted few-shot prompt
        """
        examples_text = ""
        for i, (example_input, example_output) in enumerate(examples):
            examples_text += f"Example {i+1}:\\nInput: {example_input}\\nOutput: {example_output}\\n\\n"
        
        return f"{task_description}\\n\\n{examples_text}Input: {input_text}\\n\\nOutput:"
    
    def generate_structured_response(
        self, 
        task_description: str,
        input_text: str,
        prompt_type: str = "zero-shot",
        examples: Optional[List[Tuple[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate a structured response using the specified prompting strategy
        
        Args:
            task_description: Description of the task
            input_text: Input text to process
            prompt_type: Type of prompt to use ("zero-shot", "one-shot", or "few-shot")
            examples: List of (input, output) tuples for examples (required for one-shot and few-shot)
            
        Returns:
            Parsed structured response as a dictionary
        """
        if prompt_type == "zero-shot":
            prompt = self.zero_shot_prompt(task_description, input_text)
        elif prompt_type == "one-shot":
            if not examples or len(examples) < 1:
                raise ValueError("At least one example is required for one-shot prompting")
            prompt = self.one_shot_prompt(task_description, examples[0][0], examples[0][1], input_text)
        elif prompt_type == "few-shot":
            if not examples or len(examples) < 2:
                raise ValueError("At least two examples are required for few-shot prompting")
            prompt = self.few_shot_prompt(task_description, examples, input_text)
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        
        # Generate response
        response_text = self.generate_text(prompt)
        
        # Try to parse the response as JSON
        try:
            # First, try to find JSON-like structure in the response
            json_match = re.search(r'\\{.*\\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                # If no JSON structure found, return as raw text
                return {"raw_response": response_text}
        except json.JSONDecodeError:
            # If JSON parsing fails, return as raw text
            return {"raw_response": response_text}

def main():
    """Command line interface for the structured response generator"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate structured responses from LLMs using different prompting strategies")
    
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
        "--task-file", 
        type=str, 
        required=True,
        help="Path to a JSON file containing the task description and examples"
    )
    
    parser.add_argument(
        "--input", 
        type=str, 
        required=True,
        help="Input text to process"
    )
    
    parser.add_argument(
        "--prompt-type", 
        type=str, 
        default="few-shot",
        choices=["zero-shot", "one-shot", "few-shot"],
        help="Type of prompting strategy to use"
    )
    
    parser.add_argument(
        "--output-file", 
        type=str, 
        default=None,
        help="Path to save the output (if not provided, prints to stdout)"
    )
    
    args = parser.parse_args()
    
    # Load task description and examples
    with open(args.task_file, 'r') as f:
        task_data = json.load(f)
    
    task_description = task_data.get("task_description", "")
    examples = task_data.get("examples", [])
    
    # Create generator
    generator = StructuredResponseGenerator(
        model_name=args.model,
        api_key=args.api_key
    )
    
    # Generate response
    response = generator.generate_structured_response(
        task_description=task_description,
        input_text=args.input,
        prompt_type=args.prompt_type,
        examples=examples if examples else None
    )
    
    # Output response
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(response, f, indent=2)
        print(f"Response saved to {args.output_file}")
    else:
        print(json.dumps(response, indent=2))

if __name__ == "__main__":
    main()
'''
    
    # Save the script
    with open('utils/structured_response.py', 'w') as f:
        f.write(script_content)
    
    # Make the script executable
    os.chmod('utils/structured_response.py', 0o755)
    
    print("Structured response utility saved to utils/structured_response.py")

# Save the utility
save_structured_response_utility()

# Create a sample task file for the utility
def create_sample_task_file():
    """Create a sample task file for the structured response utility"""
    
    task_data = {
        "task_description": symptom_classification_task,
        "examples": [
            {
                "input": example[0],
                "output": example[1]
            } for example in symptom_examples
        ]
    }
    
    # Save the task file
    with open('utils/symptom_classification_task.json', 'w') as f:
        json.dump(task_data, f, indent=2)
    
    print("Sample task file saved to utils/symptom_classification_task.json")

# Create the sample task file
create_sample_task_file()

# Create usage examples
def create_usage_examples():
    """Create usage examples for the structured response utility"""
    
    examples = '''# Structured Response Utility Usage Examples

## Basic Usage

Generate a structured response using few-shot learning:

```bash
python utils/structured_response.py --task-file utils/symptom_classification_task.json --input "severe headache with sensitivity to light" --prompt-type few-shot
```

## Different Prompting Strategies

### Zero-shot (no examples):

```bash
python utils/structured_response.py --task-file utils/symptom_classification_task.json --input "severe headache with sensitivity to light" --prompt-type zero-shot
```

### One-shot (one example):

```bash
python utils/structured_response.py --task-file utils/symptom_classification_task.json --input "severe headache with sensitivity to light" --prompt-type one-shot
```

## Using Different Models

Use a larger model for better quality:

```bash
python utils/structured_response.py --task-file utils/symptom_classification_task.json --input "severe headache with sensitivity to light" --model google/flan-t5-large
```

## Saving Output to a File

```bash
python utils/structured_response.py --task-file utils/symptom_classification_task.json --input "severe headache with sensitivity to light" --output-file results/headache_classification.json
```

## Creating Your Own Task

1. Create a JSON file with your task description and examples:

```json
{
  "task_description": "Task: Classify diseases into acute or chronic categories...",
  "examples": [
    {
      "input": "common cold",
      "output": "{ \"disease\": \"common cold\", \"category\": \"acute\", ... }"
    },
    {
      "input": "diabetes type 2",
      "output": "{ \"disease\": \"diabetes type 2\", \"category\": \"chronic\", ... }"
    }
  ]
}
```

2. Run the utility with your custom task:

```bash
python utils/structured_response.py --task-file your_custom_task.json --input "influenza" --prompt-type few-shot
```
'''
    
    # Save the examples
    with open('results/part_3/usage_examples.txt', 'w') as f:
        f.write(examples)
    
    print("Usage examples saved to results/part_3/usage_examples.txt")

# Create usage examples
create_usage_examples()
```

## Progress Checkpoints

1. **Dataset and Few-shot Learning Concepts**:
   - [ ] Load and explore the Gout Emergency dataset
   - [ ] Understand 0-shot, 1-shot, and few-shot learning
   - [ ] Define a gout-related classification task
   - [ ] Create a structured output format

2. **Utility Implementation**:
   - [ ] Implement the structured response generator
   - [ ] Create different prompting strategies
   - [ ] Handle API errors and response parsing

3. **Example Creation**:
   - [ ] Create diverse examples for few-shot learning
   - [ ] Cover different body systems
   - [ ] Format examples with proper structure

4. **Strategy Comparison**:
   - [ ] Test different prompting strategies
   - [ ] Evaluate response quality
   - [ ] Analyze performance differences

5. **Reusable Utility**:
   - [ ] Create a command-line utility
   - [ ] Document usage examples
   - [ ] Support different models and tasks

## Common Issues and Solutions

1. **API Issues**:
   - Problem: Rate limiting
   - Solution: Implement exponential backoff and retry logic
   - Problem: Authentication errors
   - Solution: Verify API key and environment variables

2. **Response Parsing Issues**:
   - Problem: Inconsistent JSON formatting
   - Solution: Use regex to extract JSON-like structures
   - Problem: Missing fields in response
   - Solution: Validate response structure and provide defaults

3. **Prompting Issues**:
   - Problem: Poor performance with zero-shot
   - Solution: Use few-shot with diverse examples
   - Problem: Examples not representative of gout complaints
   - Solution: Ensure examples cover different types of gout presentations

4. **Evaluation Issues**:
   - Problem: Metrics not capturing quality
   - Solution: Use multiple metrics (accuracy, format, completeness)
   - Problem: Subjective quality assessment
   - Solution: Include confidence levels and explanations
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 6))
    chart = sns.barplot(x="Strategy", y="Value", hue="Metric", data=df)
    chart.set_title("Comparison of Prompting Strategies")
    chart.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("results/part_3/strategy_comparison.png")
    plt.show()

# Plot the comparison
plot_comparison(comparison_results)

# Save the results
with open("results/part_3/prompt_comparison.txt", "w") as f:
    f.write("# Prompt Strategy Comparison Results\n\n")
    
    for strategy, metrics in comparison_results.items():
        f.write(f"## {strategy.capitalize()} Prompting\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        f.write("\n")
    
    f.write("# Analysis\n\n")
    f.write("Based on the comparison, the most effective prompting strategy is ")
    
    # Determine best strategy based on accuracy
    best_strategy = max(comparison_results.items(), key=lambda x: x[1]["accuracy"])[0]
    f.write(f"{best_strategy} prompting, which achieved the highest accuracy.\n\n")
    
    f.write("## Observations\n\n")
    f.write("- Zero-shot prompting provides the baseline performance without examples\n")
    f.write("- One-shot prompting shows how a single example can guide the model\n")
    f.write("- Few-shot prompting demonstrates the benefit of multiple diverse examples\n\n")
    
    f.write("## Recommendations\n\n")
    f.write("For medical symptom classification tasks:\n")
    f.write("1. Use few-shot prompting with diverse, high-quality examples\n")
    f.write("2. Ensure examples cover different body systems\n")
    f.write("3. Provide clear formatting instructions for structured outputs\n")
    f.write("4. Include confidence levels to acknowledge uncertainty\n")

print("Results saved to results/part_3/prompt_comparison.txt")
```
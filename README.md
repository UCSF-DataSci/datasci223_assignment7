# Assignment 7: Deep Learning and LLM Applications in Healthcare

## Overview

This assignment explores deep learning techniques and Large Language Model (LLM) applications in healthcare data analysis. You'll work through four parts:

1. **Model Run-off**: Implement systematic model selection for healthcare data
2. **LLM API Usage**: Create a basic command line tool for LLM interaction
3. **Structured Response**: Use 0/1/few-shot learning with an LLM via API
4. **Optional: nanoGPT Training**: Train a small GPT model on healthcare text data

## Learning Objectives

- Implement systematic model selection techniques
- Interact with LLM APIs for healthcare applications
- Apply few-shot learning techniques with LLMs
- Evaluate model performance using appropriate metrics
- Interpret results in a healthcare context
- (Optional) Train a small language model on domain-specific data

## Setup

1. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Directory Structure**:

   ```
   datasci223_assignment7/
   ├── models/              # Saved models
   ├── results/             # Evaluation metrics
   │   ├── part_1/          # Part 1 results
   │   ├── part_2/          # Part 2 results
   │   ├── part_3/          # Part 3 results
   │   └── part_4/          # Part 4 results (optional)
   ├── logs/                # Training logs
   ├── data/                # Downloaded datasets
   ├── utils/               # Utility functions
   ├── 01_model_runoff.md   # Part 1 notebook
   ├── 02_llm_api_chat.md   # Part 2 notebook
   ├── 03_structured_response.md # Part 3 notebook
   ├── 04_nanogpt_training.md # Part 4 notebook (optional)
   └── requirements.txt
   ```

## Part 1: Model Run-off (Systematic Model Selection)

- Implement a systematic approach to model selection
- Compare multiple model architectures on the same dataset
- Save best model as `models/best_model.keras` or `models/best_model.pt`
- Save metrics in `results/part_1/model_comparison.txt`

Goals:
- Compare at least 3 different model architectures
- Implement proper cross-validation
- Analyze performance trade-offs (accuracy vs. complexity)
- Justify final model selection

## Part 2: LLM API Usage (Command Line Chat Tool)

- Create a command line tool to interact with an LLM API
- Implement a chat interface for healthcare-related queries
- Save the tool as `utils/llm_chat.py`
- Document usage in `results/part_2/usage_examples.txt`

Goals:
- Successfully connect to an LLM API (e.g., Hugging Face)
- Implement a user-friendly command line interface
- Handle API errors gracefully
- Optimize for low-latency responses

## Part 3: Structured Response (Few-shot Learning)

- Implement 0/1/few-shot learning with an LLM
- Compare performance across different prompting strategies
- Save implementation as `utils/structured_response.py`
- Save metrics in `results/part_3/prompt_comparison.txt`

Goals:
- Implement at least 3 different prompting strategies
- Evaluate performance on a healthcare-related task
- Analyze the impact of prompt design on response quality
- Optimize for both accuracy and response structure

## Part 4: Optional - nanoGPT Training

- Train a small GPT model on healthcare text data
- Use the provided open database text or find an appropriate corpus
- Save model as `models/nanogpt.pt`
- Save training metrics in `results/part_4/training_metrics.txt`

Goals:
- Successfully train a small language model
- Analyze training dynamics and convergence
- Evaluate generated text quality
- Compare performance to larger pre-trained models

## API Options

1. **Hugging Face**:
   - Free tier available
   - Wide range of models
   - Simple Python API

2. **OpenAI**:
   - Requires API key and payment
   - State-of-the-art models
   - Comprehensive documentation

3. **Local Models**:
   - No API costs
   - Limited to smaller models
   - Higher computational requirements

## Common Issues and Solutions

1. **API Access**:
   - Problem: API key issues or rate limiting
   - Solution: Use Hugging Face free tier or implement retries

2. **Model Training**:
   - Problem: Training instability
   - Solution: Use gradient clipping, reduce learning rate
   - Problem: Memory limitations
   - Solution: Reduce batch size, use gradient accumulation

3. **Evaluation**:
   - Problem: Metrics format incorrect
   - Solution: Follow the exact format specified
   - Problem: Inconsistent LLM responses
   - Solution: Use temperature=0 for deterministic outputs

## Resources

1. **Documentation**:
   - [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
   - [OpenAI API](https://platform.openai.com/docs/api-reference)
   - [nanoGPT](https://github.com/karpathy/nanoGPT)
   - [Few-shot Learning Guide](https://www.promptingguide.ai/techniques/fewshot)
   - [Model Selection Best Practices](https://scikit-learn.org/stable/model_selection.html)
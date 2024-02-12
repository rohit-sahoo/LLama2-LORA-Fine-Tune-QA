# LLM Llama 2 LORA: Fine-Tuned for Enhanced Question Answering


This project enhances the question-answering capabilities of the 7B-parameter Llama 2 Large Language Model (LLM) through Low-Rank Adaptation (LORA) and incorporates Retrieval Augmented Generation (RAG). It focuses on a specialized dataset from Databricks, consisting of high-quality, AI-generated prompt/response pairs across various instruction categories. RAG dynamically enriches the LLM's input with relevant information, significantly improving the model's ability to generate accurate and contextually rich responses.
**Table of Contents**

- Features
- Dataset
- Requirements
- Usage
  - Training
  - Inference
- Project Structure

**Features**
1. **Llama 2 LLM Integration:** Utilizes the powerful Llama 2 model with 7 billion parameters.

2. **LORA Fine-Tuning:** Applies Low-Rank Adaptation for efficient and effective fine-tuning of the LLM.

3.  **RAG Enhancement:** Integrates Retrieval Augmented Generation to dynamically enrich the model's input with relevant information from the databricks-dolly-15k dataset, improving answer accuracy and contextuality.

4. **Diverse Dataset:** Leverages a specialized dataset from Databricks for training and validation.

5. **Advanced NLP Techniques:** Employs state-of-the-art NLP methodologies for enhanced question answering.


**Dataset:**
```databricks-dolly-15k``` is a corpus of more than 15,000 records generated by thousands of Databricks employees to enable large language models to exhibit the magical interactivity of ChatGPT. Databricks employees were invited to create prompt / response pairs in each of eight different instruction categories, including the seven outlined in the InstructGPT paper, as well as an open-ended free-form category. The contributors were instructed to avoid using information from any source on the web with the exception of Wikipedia (for particular subsets of instruction categories), and explicitly instructed to avoid using generative AI in formulating instructions or responses. Examples of each behavior were provided to motivate the types of questions and instructions appropriate to each category.

Halfway through the data generation process, contributors were given the option of answering questions posed by other contributors. They were asked to rephrase the original question and only select questions they could be reasonably expected to answer correctly.

databricks-dolly-15k is an open source dataset of instruction-following records generated by thousands of Databricks employees in several of the behavioral categories outlined in the InstructGPT paper, including brainstorming, classification, closed QA, generation, information extraction, open QA, and summarization.

Dataset link: [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)

**Retrieval Augmented Generation**

RAG enhances the project by dynamically incorporating relevant external knowledge from the databricks-dolly-15k dataset during the question-answering process. This approach significantly enriches the model's responses, allowing for more accurate, context-aware answers that draw from a broad knowledge base. RAG's integration facilitates a deeper understanding and more nuanced responses, leveraging the vast information embedded in the dataset to address complex queries across various instruction categories.

**Requirements**

Before using this project, make sure you have the following dependencies installed:

- Python 3.7+
- PyTorch
- Hugging Face Transformers library
- bitsandbytes
- Additional dependencies as required by your specific environment.

You can install Python packages using ```pip```:

```
pip install -r requirements.txt
```

**Usage**

1. Training

To train the Llama model with your own dataset or configuration, follow these steps:

1. Ensure that you have access through Llama is freely available.
2. Modify the configuration in the training script as needed, specifying the dataset location and hyperparameters.
3. Run the training script:

```
python main.py
```

2. Inference

To use the trained Llama model for question answering, you can utilize the inference script. Here's how:

1. Load the pretrained Llama model (or train your own as described above).
2. Create a question prompt using the provided utility functions.
3. Generate responses to your questions using the Llama model.
4. Analyze and evaluate the responses as needed.

```python
# Load the pretrained Llama model
llama = Llama.load_pretrained_model('llama_adapter')

# Create a question prompt
question = create_question(question_instance)

# Generate a response
response = llama.ask_model(question)
```

**Note**: You can also run ```python eval.py``` to ask the model with a predefined question, and compare the performance from the fine-tuned Llama and the vanilla version.

**Project Structure**

The project structure is organized as follows:

```
project-root/
  ├── README.md
  ├── main.py               # Training script (customize for your dataset)
  ├── eval.py               # Inference script for question answering
  ├── model.py              # Llama model definition and utilities
  ├── dataset.py            # Dataset loading and preprocessing
  └── requirements.txt      # List of Python dependencies
```

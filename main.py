import os
import torch
from model import Llama
from dataset import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

if __name__ == '__main__':
    # Define the output directory for saving the trained model.
    output_dir = 'llama_adapter'

    # Initialize the Llama model.
    llama = Llama()

    # Initialize the dataset.
    dataset = Dataset('databricks-dolly-15k-context-rag.json', llama.max_length)

    # Preprocess the dataset using the Llama tokenizer.
    data = dataset.preprocess_dataset(llama.tokenizer)

    # Initialize the Trainer for model training.
    trainer = Trainer(
        model=llama.model,
        train_dataset=data,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            max_steps=20,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir='outputs',
            optim='paged_adamw_8bit'
        ),
        data_collator=DataCollatorForLanguageModeling(
            llama.tokenizer, mlm=False
        )
    )

    print('Training ...')

    # Train the model.
    train_result = trainer.train()
    metrics = train_result.metrics

    # Log and save training metrics.
    trainer.log_metrics('train', metrics)
    trainer.save_metrics('train', metrics)
    trainer.save_state()
    print(metrics)

    print('Saving last checkpoint of the model ...')

    # Create the output directory if it doesn't exist.
    os.makedirs(output_dir, exist_ok=True)

    # Save the last checkpoint of the trained model to the output directory.
    trainer.model.save_pretrained(output_dir)

    # Clean up resources.
    del llama
    del trainer
    torch.cuda.empty_cache()

from datasets import load_dataset
from functools import partial


class Dataset:
    """
    A class for loading and preprocessing a text dataset for a given model.

    Args:
        dataset (str): The name of the dataset to load.
        max_length (int): The maximum length of input sequences.

    Attributes:
        dataset (datasets.Dataset): The loaded dataset.
        max_length (int): The maximum length of input sequences.

    Methods:
        format_prompt(sample: dict) -> dict:
            Formats a dataset sample into a model input prompt.

        preprocess_batch(batch: dict, tokenizer) -> dict:
            Preprocesses a batch of samples using the tokenizer.

        preprocess_dataset(tokenizer) -> datasets.Dataset:
            Preprocesses the entire dataset using the provided tokenizer.

    """

    def __init__(self, dataset: str, max_length: int):
        self.dataset = load_dataset(dataset, split='train')
        self.max_length = max_length

    def format_prompt(self, sample: dict) -> dict:
        """
        Formats a dataset sample into a model input prompt.

        Args:
            sample (dict): A dictionary containing dataset sample information.

        Returns:
            dict: The formatted sample with a 'text' field containing the prompt.

        """
        # Constants for different sections in the prompt
        INTRO_BLURB = 'Below is an instruction that describes a task. Write a response that appropriately completes the request.'
        INSTRUCTION_KEY = '### Instruction:'
        INPUT_KEY = 'Input:'
        RESPONSE_KEY = '### Response:'
        END_KEY = '### End'

        blurb = INTRO_BLURB
        instruction = f'{INSTRUCTION_KEY}\n{sample["instruction"]}'
        input_context = f'{INPUT_KEY}\n{sample["context"]}' if sample['context'] else None
        response = f'{RESPONSE_KEY}\n{sample["response"]}'
        end = f'{END_KEY}'

        parts = [part for part in [blurb, instruction,
                                   input_context, response, end] if part]

        formatted_prompt = '\n\n'.join(parts)

        sample['text'] = formatted_prompt

        return sample

    def preprocess_batch(self, batch: dict, tokenizer) -> dict:
        """
        Preprocesses a batch of samples using the tokenizer.

        Args:
            batch (dict): A batch of samples.
            tokenizer: The tokenizer to use for preprocessing.

        Returns:
            dict: The preprocessed batch.

        """
        return tokenizer(
            batch['text'],
            max_length=self.max_length,
            truncation=True
        )

    def preprocess_dataset(self, tokenizer):
        """
        SOURCE https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
        Preprocesses the entire dataset using the provided tokenizer.

        Args:
            tokenizer: The tokenizer to use for preprocessing.

        Returns:
            datasets.Dataset: The preprocessed dataset.

        """
        _preprocessing_function = partial(
            self.preprocess_batch, tokenizer=tokenizer)
        dataset = self.dataset.map(self.format_prompt)

        dataset = dataset.map(
            _preprocessing_function,
            batched=True,
            remove_columns=['instruction', 'context',
                            'response', 'text', 'category']
        )

        dataset = dataset.filter(lambda sample: len(
            sample['input_ids']) < self.max_length)

        return dataset



if __name__ == '__main__':
    from model import Llama

    model = Llama()
    dataset = Dataset('databricks/databricks-dolly-15k', model.max_length)

    print('\nDataset Sample:')
    print(dataset.dataset[0])

    data = dataset.preprocess_dataset(model.tokenizer)
    print('\nPre-processed by Llama 2 Tokenizer')
    print(data[0])

# -*- coding: utf-8 -*-

# This script contains all data transformation and reading

import random
from torch.utils.data import Dataset

def read_line_examples_from_file(data_path, lowercase, silence=True):
    """
    Read data from file, each line is: sent####labels
    Return List[List[word]], List[Tuple]
    """
    sents, labels = [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        words, labels = [], []
        for line in fp:
            line = line.strip()
            if lowercase:
                line = line.lower()
            if line != '':
                words, tuples = line.split('####')
                sents.append(words.split())
                labels.append(tuples.split())
    if silence:
        print(f"Total examples = {len(sents)}")
    return sents, labels


def get_transformed_io(data_path, data_type, args):
    """
    The main function to transform input & target according to the task
    """
    sents, labels = read_line_examples_from_file(data_path, args.lowercase)

    # the input is just the raw sentence
    inputs = [s.copy() for s in sents]
    targets = [s.copy() for s in labels]
    

    return inputs, targets


class ABSADataset(Dataset):
    def __init__(self,
                 tokenizer,
                 task_name,
                 data_type,
                 args,
                 max_len=200):
        # './data/rest16/train.txt'
        self.data_path = f'{args.data_path}/{data_type}.txt'
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.data_type = data_type
        self.args = args

        self.inputs = []
        self.targets = []

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, 
                "target_ids": target_ids, "target_mask": target_mask}

    def _build_examples(self):

        inputs, targets = get_transformed_io(self.data_path, self.task_name, self.args)

        for i in range(len(inputs)):
            # change input and target to two strings
            input = ' '.join(inputs[i])
            target = ' '.join(targets[i])

            tokenized_input = self.tokenizer.batch_encode_plus(
              [input], max_length=self.max_len, padding="max_length",
              truncation=True, return_tensors="pt"
            )
            tokenized_target = self.tokenizer.batch_encode_plus(
              [target], max_length=self.max_len, padding="max_length",
              truncation=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)

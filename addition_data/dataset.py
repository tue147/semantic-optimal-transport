import torch
from torch.utils.data import Dataset, DataLoader
from transformer_lens import HookedTransformer

class MathAdditionDataset(Dataset):
    def __init__(self, model: HookedTransformer, min_val=0, max_val=99):
        """
        Generates a dataset of string addition problems.
        Format: "Calc:p1+p2=" -> "answer"
        """
        self.samples = []
        self.model = model
        # Generate all combinations of p1 and p2
        all_strings = []
        self.all_answers = []
        for p1 in range(min_val, max_val + 1):
            for p2 in range(min_val, max_val + 1):
                query = f"Calc:{p1}+{p2}="
                answer = str(p1 + p2)
                all_strings.append(query)
                self.all_answers.append(answer)
        
        self.samples.append((self.model.to_tokens(all_strings, padding_side="left")))

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        query = self.samples[idx]
        answer = self.all_answers[idx]
        return query, answer
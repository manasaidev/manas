from torch.utils.data import Dataset
from .text_processing import word_tokenize

class TextDataset(Dataset):
    def __init__(self, texts, vocab, max_seq_length=512):
        self.texts = texts
        self.max_seq_length = max_seq_length
        self.vocab = vocab
        self.idx2word = {idx: word for word, idx in vocab.items()}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = word_tokenize(self.texts[idx].lower())
        token_ids = [self.vocab.get(token, self.vocab.get('<unk>')) for token in tokens]
        if len(token_ids) > self.max_seq_length:
            token_ids = token_ids[:self.max_seq_length]
        else:
            token_ids += [self.vocab.get('<pad>')] * (self.max_seq_length - len(token_ids))
        return torch.tensor(token_ids, dtype=torch.long)
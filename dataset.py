from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, seq_len=50):
        self.seq_len = seq_len
        self.text = self.get_text()
        self.tokens = self.tokenize()


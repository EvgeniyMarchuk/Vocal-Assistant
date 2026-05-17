from pathlib import Path

import torch
from Tokenizers import Tokenizers, TokenizersConfig

# load the pre-trained checkpoints
checkpoint_path = Path(__file__).with_name("Tokenizer_iter2.pt")
if not checkpoint_path.exists():
    raise FileNotFoundError(
        "Missing Tokenizer_iter2.pt. Download it from the links in README.md "
        "and place it in this directory before running the example."
    )
checkpoint = torch.load(checkpoint_path)

cfg = TokenizersConfig(checkpoint["cfg"])
BEATs_tokenizer = Tokenizers(cfg)
BEATs_tokenizer.load_state_dict(checkpoint["model"])
BEATs_tokenizer.eval()

# tokenize the audio and generate the labels
audio_input_16khz = torch.randn(1, 10000)
padding_mask = torch.zeros(1, 10000).bool()

labels = BEATs_tokenizer.extract_labels(audio_input_16khz, padding_mask=padding_mask)
print(labels.shape)

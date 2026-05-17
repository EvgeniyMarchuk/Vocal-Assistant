from pathlib import Path

import torch
from BEATs import BEATs, BEATsConfig

# load the pre-trained checkpoints
checkpoint_path = Path(__file__).with_name("BEATs_iter1.pt")
if not checkpoint_path.exists():
    raise FileNotFoundError(
        "Missing BEATs_iter1.pt. Download it from the links in README.md "
        "and place it in this directory before running the example."
    )
checkpoint = torch.load(checkpoint_path)

cfg = BEATsConfig(checkpoint["cfg"])
BEATs_model = BEATs(cfg)
BEATs_model.load_state_dict(checkpoint["model"])
BEATs_model.eval()

# extract the the audio representation
audio_input_16khz = torch.randn(1, 10000)
padding_mask = torch.zeros(1, 10000).bool()

representation = BEATs_model.extract_features(
    audio_input_16khz, padding_mask=padding_mask
)[0]
print(representation.shape, type(representation))

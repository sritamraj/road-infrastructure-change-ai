import random, numpy as np, torch, yaml

def load_yaml(path):
    with open(path, encoding='utf-8') as f: return yaml.safe_load(f)

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class AssessmentDataset(Dataset):
    def __init__(self, root_dir, subset_fraction=1.0):
        self.root_dir = root_dir
        self.samples = []
        
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

        for label, shape in enumerate(["cubes", "spheres"]):
            shape_dir = os.path.join(root_dir, shape)
            rgb_dir = os.path.join(shape_dir, "rgb")

            if not os.path.exists(rgb_dir):
                print(f"Warning: {rgb_dir} not found. Skipping.")
                continue

            try:
                az = np.load(os.path.join(shape_dir, "azimuth.npy"))
                ze = np.load(os.path.join(shape_dir, "zenith.npy"))
            except FileNotFoundError:
                print(f"Warning: Metadata not found in {shape_dir}. Skipping.")
                continue

            max_idx = len(az)

            files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".png")])

            valid_files = []
            for f in files:
                try:
                    idx = int(f.split('.')[0])
                    if idx < max_idx:  
                        valid_files.append(f)
                except ValueError:
                    continue
            
            files = valid_files

            if subset_fraction < 1.0:
                count = int(len(files) * subset_fraction)
                count = max(count, 1) if len(files) > 0 else 0
                files = files[:count]
                
            for fname in files:
                idx = int(fname.split('.')[0])
                self.samples.append({
                    "rgb": os.path.join(rgb_dir, fname),
                    "lidar": os.path.join(shape_dir, "lidar", f"{idx:04d}.npy"),
                    "az": az[idx], "ze": ze[idx], "label": label
                })
        
        print(f"Loaded {len(self.samples)} valid samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        try:

            rgb = Image.open(item["rgb"]).convert("RGB")
            rgb_t = self.transform(rgb)
            rgb_in = torch.cat([rgb_t, torch.zeros(1, 64, 64)], dim=0)
            
            depth = torch.tensor(np.load(item["lidar"]), dtype=torch.float32)
            lidar_in = self.depth_to_xyza(depth, item["az"], item["ze"])
            
            return rgb_in, lidar_in, torch.tensor(item["label"], dtype=torch.long)
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return torch.zeros(4, 64, 64), torch.zeros(4, 64, 64), torch.tensor(0)

    def depth_to_xyza(self, d, az, ze):

        x = d * np.sin(-az) * np.cos(-ze)
        y = d * np.cos(-az) * np.cos(-ze)
        z = d * np.sin(-ze)
        mask = (d < 50.0).float() 
        return torch.stack([x, y, z, mask], dim=0)

def get_loaders(root, batch_size=32, fraction=0.1):
    ds = AssessmentDataset(root, subset_fraction=fraction)
    
    if len(ds) == 0:
        raise ValueError("Dataset is empty! Check your data paths and metadata files.")

    train_len = int(0.8 * len(ds))
    val_len = len(ds) - train_len
    train, val = torch.utils.data.random_split(ds, [train_len, val_len])
    
    return DataLoader(train, batch_size, shuffle=True), DataLoader(val, batch_size)
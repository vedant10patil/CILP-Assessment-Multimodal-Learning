import torch
import wandb
from tqdm import tqdm
import os

def run_training(model, train_loader, val_loader, config, name):
    # Initialize W&B

    wandb.init(project="cilp-extended-assessment", name=name, config=config, reinit=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optim = torch.optim.Adam(model.parameters(), lr=config["lr"])
    crit = torch.nn.CrossEntropyLoss()
    
    best_acc = 0.0

    # Training Loop
    for epoch in range(config["epochs"]):
        model.train()

        for rgb, lidar, y in tqdm(train_loader, desc=f"Ep {epoch}", leave=False):
            rgb, lidar, y = rgb.to(device), lidar.to(device), y.to(device)
            
            optim.zero_grad()
            outputs = model(rgb, lidar)
            loss = crit(outputs, y)
            loss.backward()
            optim.step()
            
        # Validation Loop
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for rgb, lidar, y in val_loader:
                rgb, lidar, y = rgb.to(device), lidar.to(device), y.to(device)
                
                preds = model(rgb, lidar).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        
        # Calculate Accuracy
        acc = 100 * correct / total

        wandb.log({"epoch": epoch, "accuracy": acc, "loss": loss.item()})

        if acc > best_acc:
            best_acc = acc
            
            try:

                save_dir = os.path.join(os.getcwd(), "checkpoints")
                os.makedirs(save_dir, exist_ok=True)
                
                save_path = os.path.join(save_dir, f"{name}.pth")
                torch.save(model.state_dict(), save_path)
                
            except Exception as e:
                print(f"Warning: Could not save checkpoint: {e}")
            
    wandb.finish()
    return best_acc
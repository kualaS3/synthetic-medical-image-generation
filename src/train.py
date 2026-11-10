"""Training loops: GAN training and downstream classifier training.

Both functions are verbatim from the notebook (cells 19 and 30).
"""
import os

import torch
import torch.optim as optim
from tqdm import tqdm

from .models import DRClassifier


def train_gan(CONFIG, device, dataloader, G, D, opt_G, opt_D, criterion):
    G.train()
    D.train()

    g_losses, d_losses = [], []
    best_g_loss = float("inf")
    checkpoint_dir = os.path.join(CONFIG["output_dir"], "checkpoints")

    print("Testing dataloader...")
    for batch_idx, (imgs, labels) in enumerate(dataloader):
        print(f"Got batch {batch_idx}")
        break

    print("Starting GAN training...")
    print(f"Epochs: {CONFIG['epochs']} | Batches per epoch: {len(dataloader)}")

    for epoch in range(1, CONFIG["epochs"] + 1):
        print(f"\n--- Epoch {epoch}/{CONFIG['epochs']} START ---")
        epoch_g = epoch_d = 0.0

        with tqdm(dataloader, desc=f"Epoch {epoch}/{CONFIG['epochs']}") as pbar:
            for batch_idx, (imgs, labels) in enumerate(pbar):
                if batch_idx == 0:
                    print(f"  First batch loaded (epoch {epoch})")

                imgs = imgs.to(device)
                labels = labels.to(device)
                bs = imgs.size(0)

                real_targets = torch.full((bs,), 0.9, device=device)
                fake_targets = torch.zeros(bs, device=device)

                # Dstep
                opt_D.zero_grad()
                d_real = D(imgs, labels)
                loss_d_real = criterion(d_real, real_targets)

                z = torch.randn(bs, CONFIG["latent_dim"], device=device)
                fake_imgs = G(z, labels).detach()
                d_fake = D(fake_imgs, labels)
                loss_d_fake = criterion(d_fake, fake_targets)

                loss_D = loss_d_real + loss_d_fake
                loss_D.backward()
                opt_D.step()

                # Gstep
                opt_G.zero_grad()
                z = torch.randn(bs, CONFIG["latent_dim"], device=device)
                fake_imgs = G(z, labels)
                d_fake = D(fake_imgs, labels)
                loss_G = criterion(d_fake, torch.ones(bs, device=device))
                loss_G.backward()
                opt_G.step()

                epoch_g += loss_G.item()
                epoch_d += loss_D.item()

                if batch_idx % 10 == 0:
                    print(
                        f"    batch {batch_idx}/{len(dataloader)} | D: {loss_D.item():.3f} | G: {loss_G.item():.3f}"
                    )

                pbar.set_postfix({"D": f"{loss_D.item():.3f}", "G": f"{loss_G.item():.3f}"})

        print(f"Epoch {epoch} done.")
        g_losses.append(epoch_g / len(dataloader))
        d_losses.append(epoch_d / len(dataloader))
        current_g_loss = g_losses[-1]
        print(f"[Epoch {epoch}] avg D: {d_losses[-1]:.4f} | avg G: {current_g_loss:.4f}")

        # checkpoint
        if epoch % CONFIG["save_every"] == 0 or epoch == CONFIG["epochs"]:
            ckpt_path = os.path.join(checkpoint_dir, f"ckpt_epoch_{epoch:04d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "G_state": G.state_dict(),
                    "D_state": D.state_dict(),
                    "opt_G_state": opt_G.state_dict(),
                    "opt_D_state": opt_D.state_dict(),
                    "g_losses": g_losses,
                    "d_losses": d_losses,
                    "CONFIG": CONFIG,
                },
                ckpt_path,
            )
            print(f"  ✓ Checkpoint saved → {ckpt_path}")

        # Bestmodel
        if current_g_loss < best_g_loss:
            best_g_loss = current_g_loss
            best_path = os.path.join(checkpoint_dir, "best_generator.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "G_state": G.state_dict(),
                    "g_loss": best_g_loss,
                    "CONFIG": CONFIG,
                },
                best_path,
            )
            print(f"  ✓ Best model updated (G loss: {best_g_loss:.4f}) → {best_path}")

    return g_losses, d_losses


def train_classifier(name, dataloader, val_loader, device, n_classes=5, epochs=15):
    model = DRClassifier(n_classes=n_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    history = []
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        for imgs, labels in tqdm(dataloader, desc=f"[{name}] Epoch {epoch}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        history.append(acc)
        best_acc = max(best_acc, acc)
        print(f"  [{name}] Epoch {epoch:>3}  Val Acc: {acc:.4f}")

    return best_acc, history

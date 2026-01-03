def train_model(model, train_loader, val_loader, cfg):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import wandb
    from sklearn.metrics import mean_absolute_error

    device = next(model.parameters()).device
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # ✅ 출력층 초기 가중치/바이어스 확인
    print("[학습 시작 전] 출력층 weight:")
    print(model.head.weight.data[:2])
    print("[학습 시작 전] 출력층 bias:")
    print(model.head.bias.data)

    best_val_loss = float("inf")
    early_stop_counter = 0
    patience = cfg.get("early_stopping", {}).get("patience",10)
    use_early_stop = cfg.get("early_stopping", {}).get("use", True)

    train_losses, val_losses = [], []
    val_maes = []

    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0

        for batch_idx, (pat, rocket, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")):
            pat, rocket, y = pat.to(device), rocket.to(device), y.to(device)

            preds = model(pat, rocket)
            loss = criterion(preds,y)

            # ✅ 예측값 & 정답 일부 출력 (처음 1~2번 에폭만 출력해도 충분)
            if epoch < 2 and batch_idx == 0:
                print("[예측값]", preds[0,:10].detach().cpu().numpy())
                print("[정답]", y[0,:10].detach().cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        total_loss /= len(train_loader)
        train_losses.append(total_loss)

        # validation
        model.eval()
        val_loss = 0
        val_preds, val_trues = [],[]

        with torch.no_grad():
            for pat, rocket, y in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False):
                pat, rocket, y = pat.to(device), rocket.to(device), y.to(device)
                preds = model(pat, rocket)
                loss = criterion(preds,y)
                val_loss += loss.item()

                val_preds.append(preds.cpu())
                val_trues.append(y.cpu())

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        val_preds = torch.cat(val_preds, dim=0).numpy()
        val_trues = torch.cat(val_trues, dim=0).numpy()
        val_mae = mean_absolute_error(val_trues.flatten(), val_preds.flatten())
        val_maes.append(val_mae)

        print(f"Epoch {epoch+1} | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}")

        # W&B 로그 기록
        if cfg.get("wandb",{}).get("use",False):
            wandb.log({
                "epoch": epoch+1,
                "train_loss": total_loss,
                "val_loss": val_loss,
                "val_mae_abp": val_mae,
                "lr": scheduler.optimizer.param_groups[0]['lr']
            })

        # 학습률 스케줄러 적용
        scheduler.step(val_loss)

        # 최적 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
            print("Saved best model.")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"Validation loss 개선 없음 (연속 {early_stop_counter}회)")

        if use_early_stop and early_stop_counter >= patience:
            print(f"⏹ Early stopping triggered! (patience={patience})")
            break

    # Loss 그래프 저장
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_plot.png")
    plt.show()

    # 최종 로그
    if cfg.get("wandb", {}).get("use", False):
        wandb.log({
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "final_val_mae_abp": val_maes[-1],
            "Loss Curve": wandb.Image("loss_plot.png")
        })
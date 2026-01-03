def evaluate_model(model, test_loader, cfg):
    import torch
    from sklearn.metrics import mean_absolute_error
    import matplotlib.pyplot as plt
    import numpy as np
    import wandb
    from tqdm import tqdm

    device = next(model.parameters()).device
    model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for i, (pat, rocket, y) in enumerate(tqdm(test_loader, desc="Testing")):
            pat, rocket = pat.to(device), rocket.to(device)
            pred = model(pat, rocket).cpu().numpy()
            y = y.numpy()
            
            if i == 0:
                print("[EVAL] 예측 ABP 파형 (첫 샘플):", pred[0, :10])
                print("[EVAL] 정답 ABP 파형 (첫 샘플):", y[0, :10])
        
            preds.extend(pred)
            trues.extend(y)

    preds = np.array(preds)
    trues = np.array(trues)

    # MAE 계산 (전체 파형 기준)
    mae_abp = mean_absolute_error(trues.flatten(), preds.flatten())
    print(f"[Test MAE] ABP waveform: {mae_abp:.4f}")

    # wandb 로깅
    if cfg.get("wandb", {}).get("use",False):
        wandb.log({
            "Test MAE/ABP": mae_abp
        })

    # ✅ 시각화 1: 파형 비교 (첫 샘플)
    plt.figure(figsize=(12, 4))
    plt.plot(trues[0], label="True ABP")
    plt.plot(preds[0], label="Predicted ABP", linestyle='--')
    plt.title("ABP Waveform Prediction (Sample 0)")
    plt.xlabel("Time Step")
    plt.ylabel("Pressure")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("abp_waveform_prediction.png")
    plt.show()

    if cfg.get("wandb", {}).get("use", False):
        wandb.log({"ABP Prediction Sample": wandb.Image("abp_waveform_prediction.png")})

    # ✅ 시각화 2: 에러 분포 히스토그램
    errors = preds - trues
    plt.figure(figsize=(6, 4))
    plt.hist(errors.flatten(), bins=50, alpha=0.7, color='gray', edgecolor='black')
    plt.title("ABP Prediction Error Histogram")
    plt.xlabel("Error (Pred - True)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("abp_error_hist.png")
    plt.show()

    if cfg.get("wandb", {}).get("use", False):
        wandb.log({"ABP Error Histogram": wandb.Image("abp_error_hist.png")})
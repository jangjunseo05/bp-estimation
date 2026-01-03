import argparse
from train import train_model
from eval import evaluate_model
from data_loader import get_dataloaders
from model import TSTModel
from utils import load_yaml, set_seed
import torch
import wandb
import multiprocessing

def main():
    # 명령줄 인자 설정
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')            # 설정 파일 경로
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train', help='Run mode')    # 실행 모드 선택
    args = parser.parse_args()

    # 설정 로드 및 시드 고정
    config = load_yaml(args.config)     # YAML 설정 불러오기
    set_seed(config['seed'])            # 랜덤 시드 고정

    # wandb 초기화 (train/eval 공통)
    if config.get("wandb", {}).get("use", False):
        wandb.init(
            project=config["wandb"]["project"],
            entity=config["wandb"].get("entity", None),
            config=config,
            reinit=True,
            name=f"{args.mode}_run"
        )

    # 데이터 로더 준비
    train_loader, val_loader, test_loader, input_size, output_length = get_dataloaders(config)


    # ✅ CUDA 사용 가능 여부 확인
    if not torch.cuda.is_available():
        print("⚠️ CUDA를 사용할 수 없어 CPU로 전환됩니다.")
        config['device'] = 'cpu'
    else:
        print("✅ CUDA 사용 가능. GPU로 모델 이동합니다.")

    # 모델 정의
    model = TSTModel(
        input_dim=input_size,
        d_model=int(config['model']['d_model']),
        n_heads=int(config['model']['n_heads']),
        d_ff=int(config['model']['d_ff']),
        num_layers=int(config['model']['num_layers']),
        dropout=float(config['model']['dropout']),
        output_length=output_length
    ) 
    model = model.to(config['device'])                          # CPU 또는 GPU 할당

    # 실행 모드에 따라 학습 또는 평가
    if args.mode == 'train':
        train_model(model, train_loader, val_loader, config)
        model.load_state_dict(torch.load("best_model.pt", map_location=config['device']))
        evaluate_model(model,test_loader,config)

    elif args.mode == 'eval':
        model.load_state_dict(torch.load("best_model.pt", map_location=config['device']))
        evaluate_model(model, test_loader, config)

    # 세션 종료는 main에서 일괄 처리
    if config.get("wandb", {}).get("use", False):
        wandb.finish()

# Python 실행 시 main 함수 실행
if __name__ == '__main__':
    multiprocessing.set_start_method("spawn", force=True)
    multiprocessing.freeze_support()
    main()

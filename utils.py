import random
import numpy as np
import torch
import yaml
import os

# 랜덤 시드를 고정하여 재현 가능한 결과를 얻도록 설정하는 함수
def set_seed(seed):
    random.seed(seed)                   # Python의 random 모듈 시드 설정 
    np.random.seed(seed)                # Numpy 시드 설정 !!!!
    torch.manual_seed(seed)             # Python CPU 시드 설정
    torch.cuda.manual_seed_all(seed)    # Python GPU 시드 설정 

# YAML 파일을 읽어 파이썬 dict 형태로 반환하는 함수
def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) 

# 파이썬 객체를 YAML 형식으로 저장하는 함수
def save_yaml(data, path):
    with open(path, "w") as f:
        yaml.dump(data, f)  

# 지정한 경로가 존재하지 않으면 새로 생성하는 함수 (디렉토리 생성)
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

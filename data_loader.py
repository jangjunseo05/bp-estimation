import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sktime.transformations.panel.rocket import MiniRocket
import joblib
import heartpy as hp
from multiprocessing import Pool, cpu_count, set_start_method
import gc

# 데이터셋 클래스 정의
class BPRegressionDataset(Dataset):
    
    def __init__(self, pats, rockets, targets):
        self.pats = pats                  # PAT 시계열 (B, 1, L)
        self.rockets = rockets            # MiniRocket features (B,D)
        self.targets = targets            # 정답: ABP 파형 (L_abp)

    def __len__(self):
        return len(self.targets)

    # feature와 target을 torch.tensor로 변환하여 반환
    def __getitem__(self, idx):
        return (
            torch.tensor(self.pats[idx], dtype=torch.float32),
            torch.tensor(self.rockets[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )


# PAT 시퀀스 추출
def extract_pat_sequence(ecg_batch, ppg_batch, fs=100, max_length=50):

    """
    ECG와 PPG 피크 간 간격 기반 쌍을 구성하여 PAT 시퀀스를 계산하는 함수.
    - PPG가 먼저 시작되는 경우 해당 피크는 무시됨.
    - 첫 쌍은 전후 간격이 일치하는 경우로 찾고, 이후에는 순차적 매칭.
    """
    pat_list = []
    excluded_count = 0

    for ecg, ppg in zip(ecg_batch, ppg_batch):
        try:
            # heartpy로 peak 탐지
            ecg_wd, _ = hp.process(ecg, sample_rate=fs, report_time=False)
            ppg_wd, _ = hp.process(ppg, sample_rate=fs, report_time=False)

            ecg_peaks = [i for i in ecg_wd['peaklist'] if i != -1]
            ppg_peaks = [i for i in ppg_wd['peaklist'] if i != -1]

            # peak가 충분하지 않으면 제외
            if len(ecg_peaks) < 2 or len(ppg_peaks) == 0:
                excluded_count += 1
                continue
            
            # 첫 쌍 찾기
            i, j = 0, 0
            found = False
            while i < len(ecg_peaks) - 1 and len(ppg_peaks) > 1:
                ecg_gap = ecg_peaks[i+1] - ecg_peaks[i]
                ppg_gap = ppg_peaks[j+1] - ppg_peaks[j]
                # 간격 차이가 n 샘플 이내일 때 동일 간격으로 판단(해당 코드는 부정맥 환자에 대한 모델의 성능을 결정지을 수 있는 중요한 부분임;여러 수치로 실험 필요)
                if abs(ecg_gap - ppg_gap) <= 9 and ppg_peaks[j] > ecg_peaks[i]:
                    found = True
                    break
                # 먼저 도달하는 쪽 인덱스 증가
                if ecg_peaks[i] < ppg_peaks[j]:
                    i += 1
                else:
                    j += 1

            if not found:
                excluded_count += 1
                continue

            # 첫 쌍 이후는 순서대로 매칭
            ecg_valid = ecg_peaks[i:]
            ppg_valid = ppg_peaks[j:]
            pair_len = min(len(ecg_valid), len(ppg_valid))
            current_pat = []

            # PPG가 ECG보다 먼저 나오는 경우 제외
            for k in range(pair_len):
                if ppg_valid[k] < ecg_valid[k]:
                    continue  
                pat = (ppg_valid[k] - ecg_valid[k]) / fs
                current_pat.append(pat)

            # 유효 쌍 없음
            if len(current_pat) == 0:
                excluded_count += 1
                continue
            
            # 길이 보정
            if len(current_pat) < max_length:
                current_pat = np.pad(current_pat, (0, max_length - len(current_pat)))
            else:
                current_pat = current_pat[:max_length]

            pat_list.append(current_pat)

        except Exception as e:
            excluded_count += 1
            continue
    
    if len(pat_list) == 0:
        return np.zeros((1,max_length), dtype=np.float32),excluded_count

    return np.array(pat_list, dtype=np.float32), excluded_count

global_transformer = None

def init_worker(trained_transformer):
    global global_transformer
    global_transformer = trained_transformer

# 병렬 처리 단위 함수
def process_file(file_path):
    global global_transformer
    pid = os.path.basename(file_path).split('.')[0]

    try:
        data = np.load(file_path, mmap_mode='r').astype(np.float32)
        features = global_transformer.transform(data).astype(np.float32)

        # 시그널 분리
        ecg = data[:, 0, :]
        ppg = data[:, 1, :]
        abp = data[:, 2, :]

        # PAT 시퀀스 추출
        pat_seq, _ = extract_pat_sequence(ecg, ppg)

        # 유효성 보정
        if pat_seq.ndim == 1:
            pat_seq = pat_seq[np.newaxis, :]
        elif pat_seq.ndim != 2:
            pat_seq = np.zeros((1, 50), dtype=np.float32)

        M = pat_seq.shape[0]           # PAT 시퀀스 개수
        N = features.shape[0]          # MiniRocket feature 개수
        K = abp.shape[0]               # ABP 파형 개수
        common = min(M, N, K)
        pat_seq = pat_seq[:common]
        features = features[:common]
        abp = abp[:common]

        return pid, pat_seq[:,np.newaxis,:], features, abp
    
    except Exception as e:
        print(f"[ERROR] {pid}: {e}")
        return pid, None,None,None
    
    finally:
        # 불필요한 메모리 수거
        del data
        gc.collect()

# 전체 DataLoader 구성 함수
def get_dataloaders(cfg):
    root = cfg['data']['root']
    cache_dir = cfg['data']['feature_cache_dir']
    os.makedirs(cache_dir, exist_ok=True)

    all_files = sorted([os.path.join(root, f) for f in os.listdir(root) if f.endswith('.npy')])
    all_ids = [os.path.basename(f).split('.')[0] for f in all_files]

    # Train/Val/Test 분할
    train_ids, test_ids = train_test_split(all_ids, test_size=cfg["data"]["test_ratio"], random_state=42)
    train_ids, val_ids = train_test_split(
        train_ids,
        test_size=cfg["data"]["val_ratio"] / (cfg["data"]["val_ratio"] + cfg["data"]["train_ratio"]),
        random_state=42
    )

    # MiniRocket 훈련
    transformer = MiniRocket()
    train_data = [np.load(os.path.join(root, f"{pid}.npy"), mmap_mode='r').astype(np.float32) for pid in train_ids]
    transformer.fit(np.concatenate(train_data, axis=0))
    del train_data
    gc.collect()

    # 병렬 처리 시작
    with Pool(processes=min(10, cpu_count()), initializer=init_worker, initargs=(transformer,)) as pool:
        results = pool.imap(process_file, all_files, chunksize=2)

        # 결과 통합
        all_pats, all_features, all_targets, all_labels = [], [], [], []
        for pid, pats, feats, tgts in results:
            if pats is None:
                continue
            all_pats.append(pats)
            all_features.append(feats)
            all_targets.append(tgts)
            all_labels.extend([pid] * len(feats))

    all_pats = np.concatenate(all_pats, axis=0)
    all_features = np.concatenate(all_features, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_labels = np.array(all_labels)

    # 정규화
    # --- Split 인덱스 계산 (ID → 행 인덱스) ---
    def idx_by_ids(ids):
        return np.array([i for i, pid in enumerate(all_labels) if pid in ids], dtype=int)

    train_idx = idx_by_ids(train_ids)
    val_idx   = idx_by_ids(val_ids)
    test_idx  = idx_by_ids(test_ids)

    # --- 스케일러: train만 fit, 나머지 transform ---
    scaler = RobustScaler()
    scaler.fit(all_features[train_idx])  # ✅ 데이터 누수 방지
    all_features[train_idx] = scaler.transform(all_features[train_idx])
    all_features[val_idx]   = scaler.transform(all_features[val_idx])
    all_features[test_idx]  = scaler.transform(all_features[test_idx])
    joblib.dump(scaler, os.path.join(cache_dir, "scaler.pkl"))

    # --- 행 인덱스로 split ---
    train_pats, train_feats, train_targets = all_pats[train_idx], all_features[train_idx], all_targets[train_idx]
    val_pats,   val_feats,   val_targets   = all_pats[val_idx],   all_features[val_idx],   all_targets[val_idx]
    test_pats,  test_feats,  test_targets  = all_pats[test_idx],  all_features[test_idx],  all_targets[test_idx]


    print("✅ PAT 시퀀스 예시 (train):", train_pats[:2, 0, :5])
    print("✅ MiniRocket feature 예시 (train):", train_feats[:2, :5])
    print("✅ ABP waveform shape (train):", train_targets.shape)


    # num_workers 설정 (병렬 데이터 로딩)
    num_workers = cfg.get("data",{}).get("num_workers", 3)

    train_loader = DataLoader(BPRegressionDataset(train_pats, train_feats, train_targets), batch_size=cfg["batch_size"], shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(BPRegressionDataset(val_pats, val_feats, val_targets), batch_size=cfg["batch_size"], num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(BPRegressionDataset(test_pats, test_feats, test_targets), batch_size=cfg["batch_size"], num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, all_features.shape[1], all_targets.shape[1]
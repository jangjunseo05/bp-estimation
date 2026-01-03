import numpy as np
import os
import re
from tqdm import tqdm
import heartpy as hp
from multiprocessing import Pool, cpu_count

def get_all_case_ids(folder_path):
    case_ids = []
    for fname in os.listdir(folder_path):
        if fname.endswith('.npy'):
            match = re.match(r'case(\d+)\.npy', fname)
            if match:
                case_ids.append(int(match.group(1)))
    return sorted(case_ids)

def hr_intervals(data, fs=100):
    """HeartPy 기반 HR 간격 추출 함수 (예외 처리 추가됨)."""
    try:
        result, _ = hp.process(data, sample_rate=fs, bpmmin=40, bpmmax=180)
        peaks = result['peaklist']
        if len(peaks) < 2:
            raise ValueError("피크 개수가 부족합니다.")
        return peaks, np.diff(peaks) / fs
    except Exception as e:
        print(f"❌ HR 간격 추출 실패: {e}")
        return None  # 예외 발생 시 None 반환

def process_ecg_ppg_matched(ecg, ppg, fs=100):
    """ECG와 PPG를 슬라이딩하며 corr ≥ 0.96인 가장 뒤의 쌍 찾기."""
    ecg_len = len(ecg)
    step = 6000
    results = []
    ecg_idx = 0
    ppg_idx = 0

    while ecg_idx + 6000 <= ecg_len and ppg_idx + 6000 <= len(ppg):
        ecg_seg = ecg[ecg_idx: ecg_idx + 6000]

        ecg_results = hr_intervals(ecg_seg, fs)

        if ecg_results is not None:
            ecg_peak_idx, ecg_rr = ecg_results
        
        else:
            ecg_rr = None
        
        '''
        if hr_intervals(ecg_seg, fs) is not None:
            ecg_peak_idx, ecg_rr = hr_intervals(ecg_seg)
        else:
            ecg_rr = None
        
        '''        
        if ecg_rr is None:
            ecg_idx += step
            ppg_idx += step
            continue

        found=False

        for shift in range(0, 10):  # PPG를 1씩 옮기며 검사
            ppg_start = ppg_idx + shift
            if ppg_start + 6000 > len(ppg):
                break

            ppg_seg = ppg[ppg_start:ppg_start + 6000]

            ppg_results = hr_intervals(ppg_seg, fs)
            
            if ppg_results is not None:
                ppg_peak_idx, ppg_rr = ppg_results
            
            else:
                ppg_rr = None
            
            '''
            if hr_intervals(ppg_seg, fs) is not None:
                ppg_peak_idx, ppg_rr = hr_intervals(ppg_seg, fs)
            else:
                ppg_rr = None
            '''
            
            if ppg_rr is None or len(ecg_rr) != len(ppg_rr):
                continue
            if ecg_peak_idx[0] > ppg_peak_idx[0]:
                continue

            corr = round(np.corrcoef(ecg_rr, ppg_rr)[0, 1], 2)
            if corr >= 0.9:
                print(f"✅ ECG {ecg_idx}-{ecg_idx+6000} ↔ PPG {ppg_start}-{ppg_start+6000} | corr={corr:.4f}")
                results.append((ecg_idx, ppg_start, corr))
                found = True
                break
                
        if not found:
            print(f"❌ ECG {ecg_idx}-{ecg_idx+6000}: corr ≥ 0.9 구간 없음")

        ecg_idx += step
        ppg_idx += step

    return results

'''
        if best_match is not None:
            print(f"✅ ECG {ecg_idx}-{ecg_idx+6000} ↔ PPG {best_match[0]}-{best_match[0]+6000} | corr={best_match[1]:.4f}")
            results.append((ecg_idx, best_match[0], best_match[1]))
        else:
            print(f"❌ ECG {ecg_idx}-{ecg_idx+6000}: corr ≥ 0.96 구간 없음")

        ecg_idx += step
        ppg_idx += step

    return results
'''

def process_case(args):
    caseid, base_path, save_path = args

    try:
        print(f"➡️ case{caseid} 시작")  # 시작 로그

        ecg = np.load(os.path.join(base_path, 'ecg', f'case{caseid}.npy'))
        ppg = np.load(os.path.join(base_path, 'ppg', f'case{caseid}.npy'))
        abp = np.load(os.path.join(base_path, 'abp', f'case{caseid}.npy'))

        matches = process_ecg_ppg_matched(ecg, ppg)

        matched_segments = []
        for i, (ecg_idx, ppg_idx, corr) in enumerate(matches):
            ecg_seg = ecg[ecg_idx:ecg_idx+6000]
            ppg_seg = ppg[ppg_idx:ppg_idx+6000]
            abp_seg = abp[ppg_idx:ppg_idx+6000]

            segment = np.stack([ecg_seg, ppg_seg, abp_seg], axis=0)
            matched_segments.append(segment)

        if matched_segments:
            matched_segments = np.stack(matched_segments)
            np.save(os.path.join(save_path, f'case{caseid}_segments.npy'), matched_segments)
            print(f"✅ case{caseid}: {matched_segments.shape[0]}개 세그먼트 저장 완료 → shape = {matched_segments.shape}")
        else:
            print(f"⚠️ case{caseid}: 저장할 세그먼트 없음")

    except Exception as e:
        print(f"❌ case{caseid}: 오류 발생 - {e}")

if __name__ == '__main__':
    base_path = r"C:\Users\USER\Desktop\새PC\프로젝트\혈압추정\데이터\byCode\imputation"
    save_path = r"C:\Users\USER\Desktop\새PC\프로젝트\혈압추정\데이터\byCode\imputation\ppg0"

    ecg_folder = os.path.join(base_path, 'ecg')
    case_ids = get_all_case_ids(ecg_folder)
    print(f"총 {len(case_ids)}개의 환자 데이터를 병렬 처리합니다.")

    os.makedirs(save_path, exist_ok=True)

    args_list = [(caseid, base_path, save_path) for caseid in case_ids]

    num_workers = min(cpu_count(), 64)
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap(process_case, args_list), total=len(args_list)))

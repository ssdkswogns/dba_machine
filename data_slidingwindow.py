import os
import glob
import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

LABEL_MAP = {'DROWSY': 0, 'AGGRESSIVE': 1, 'NORMAL': 2}


def segment_and_save(root_dir, out_dir, window_sec=200, overlap_sec=50qq):
    os.makedirs(out_dir, exist_ok=True)
    window_size = int(window_sec * 10)       # 10 Hz accelerometer
    stride = window_size - int(overlap_sec * 10)

    drivers = sorted(glob.glob(os.path.join(root_dir, 'D*')))
    meta = []  # list of dicts: { 'file':..., 'label':... }

    for drv in drivers:
        seqs = sorted(glob.glob(os.path.join(drv, '*')))
        for seq in tqdm(seqs, desc=os.path.basename(drv)):
            parts = os.path.basename(seq).split('-')
            if len(parts) < 4: continue
            label_str = parts[3]
            if label_str not in LABEL_MAP:
                continue
            label = LABEL_MAP[label_str]

            # Load accelerometer data (10Hz)
            acc_path = os.path.join(seq, 'RAW_ACCELEROMETERS.txt')
            acc_df = pd.read_csv(acc_path, delim_whitespace=True, header=None,
                                 names=["Timestamp","Boolean of system activated (>50km/h)","Acc_X","Acc_Y","Acc_Z",
                                        "KF_Acc_X","KF_Acc_Y","KF_Acc_Z",
                                        "Roll","Pitch","Yaw"])
            acc_feats = acc_df[["KF_Acc_X","KF_Acc_Y","KF_Acc_Z",
                                "Roll","Pitch","Yaw"]].values.astype(np.float32)
            t_acc = acc_df['Timestamp'].values  # shape: [N_acc] #10Hz
            N_acc = acc_feats.shape[0]

            # Load GPS speed data (1Hz)
            gps_path = os.path.join(seq, 'RAW_GPS.txt')
            gps_df = pd.read_csv(gps_path, delim_whitespace=True, header=None,
                                 names=["Timestamp","Speed (km/h)","Latitude","Longitude","Altitude",
                                        "Vertical accuracy","Horizontal accuracy",
                                        "Course (degrees)","Difcourse: course variation",
                                        "Position state","Lanex dist state","Lanex history"])
            t_gps = gps_df['Timestamp'].values  # shape: [N_gps] #1Hz
            speed = gps_df["Speed (km/h)"].values.astype(np.float32)
            N_gps = speed.shape[0]

            # 디버깅: 원본 데이터 정보 출력
            print(f"\n=== {os.path.basename(seq)} ===")
            print(f"ACC: {N_acc} samples, t_acc[0]:{t_acc[0]:.2f}s ~ {t_acc[-1]:.2f}s (duration: {(t_acc[-1] - t_acc[0]):.2f}s)")
            print(f"GPS: {N_gps} samples, t_gps[0]:{t_gps[0]:.2f}s ~ {t_gps[-1]:.2f}s (duration: {(t_gps[-1] - t_gps[0]):.2f}s)")
            
            # ACC 데이터의 타임스텝 확인 (10Hz여야 함)
            acc_timesteps = np.diff(t_acc)
            print(f"ACC timesteps: mean={acc_timesteps.mean():.3f}, std={acc_timesteps.std():.3f}, min={acc_timesteps.min():.3f}, max={acc_timesteps.max():.3f}s")
            
            # GPS 데이터의 타임스텝 확인 (1Hz여야 함)
            gps_timesteps = np.diff(t_gps)
            print(f"GPS timesteps: mean={gps_timesteps.mean():.3f}, std={gps_timesteps.std():.3f}, min={gps_timesteps.min():.3f}, max={gps_timesteps.max():.3f}s")

            # 동기화: ACC 데이터를 기준으로 공통 구간 찾기
            acc_start, acc_end = t_acc[0], t_acc[-1]
            gps_start, gps_end = t_gps[0], t_gps[-1]
            # 공통 구간의 시작과 끝
            sync_start = max(acc_start, gps_start)
            sync_end = min(acc_end, gps_end)
            
            print(f"Sync period: {sync_start:.2f}s ~ {sync_end:.2f}s (duration: {(sync_end - sync_start):.2f}s)")
            
            if sync_end <= sync_start:
                print(f"WARNING: No overlap between ACC and GPS data! Skipping sequence.")
                continue
            
            # ACC 데이터에서 공통 구간 인덱스 찾기
            acc_mask = (t_acc >= sync_start) & (t_acc <= sync_end)
            acc_indices = np.where(acc_mask)[0]
            
            # GPS 데이터에서 공통 구간 인덱스 찾기
            gps_mask = (t_gps >= sync_start) & (t_gps <= sync_end)
            gps_indices = np.where(gps_mask)[0]
            
            # 동기화된 데이터 추출
            t_acc_sync = t_acc[acc_indices]
            acc_feats_sync = acc_feats[acc_indices]
            t_gps_sync = t_gps[gps_indices]
            speed_sync = speed[gps_indices]
            
            N_acc_sync = len(t_acc_sync)
            N_gps_sync = len(t_gps_sync)
            
            print(f"After sync: ACC={N_acc_sync} samples, GPS={N_gps_sync} samples")
            print(f"t_acc_sync: {t_acc_sync[0]} ~ {t_acc_sync[-1]}")
            print(f"t_gps_sync: {t_gps_sync[0]} ~ {t_gps_sync[-1]}")
            
            # ACC 타임스탬프에 맞춰 GPS 속도 보간
            speed_interp = np.interp(t_acc_sync, t_gps_sync, speed_sync)
            
            print(f"Interpolated speed shape: {speed_interp.shape}")
            
            # 최종 동기화된 데이터로 sliding windows 생성
            for start in range(0, N_acc_sync - window_size + 1, stride):
                feats_win = acc_feats_sync[start:start+window_size]
                speed_win = speed_interp[start:start+window_size].reshape(-1,1)
                window_feats = np.concatenate([feats_win, speed_win], axis=1)

                fname = f"{os.path.basename(seq)}_{start}.npz"
                out_path = os.path.join(out_dir, fname)
                np.savez_compressed(out_path, feats=window_feats, label=label)
                meta.append({'file': fname, 'label': label})

    # Save metadata
    meta_path = os.path.join(out_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {len(meta)} windows under '{out_dir}', metadata at {meta_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Segment UAH-DRIVESET into sliding windows and save")
    parser.add_argument('--data-root', type=str, default='/mnt/storage2/UAH-DRIVESET-v1/', help='UAH root')
    parser.add_argument('--out-dir', type=str, default='./windows', help='Output directory for windows')
    parser.add_argument('--window-sec', type=float, default=200.0, help='Window length in seconds')
    parser.add_argument('--overlap-sec', type=float, default=50.0, help='Overlap in seconds')
    args = parser.parse_args()
    segment_and_save(args.data_root, args.out_dir, args.window_sec, args.overlap_sec)

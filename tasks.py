# tasks.py（進捗率つき、完全版ベース）
from celery import Celery
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from ahrs.filters import Madgwick
from fastdtw import fastdtw
import base64
from io import BytesIO
import matplotlib.pyplot as plt

from celery import Celery

celery = Celery(
    "tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"  # ← ✅ 結果バックエンドを追加
)


@celery.task(bind=True)
def analyze_task(self, acc_data, gyro_data):
    try:
        self.update_state(state='PROGRESS', meta={'progress': 5})
        acc_df = pd.DataFrame(acc_data)
        gyro_df = pd.DataFrame(gyro_data)
        gyro_df['z'] = pd.to_numeric(gyro_df['gz'], errors='coerce')

        dt = (acc_df['t'].iloc[1] - acc_df['t'].iloc[0]) / 1000.0
        fs = 1.0 / dt

        madgwick = Madgwick(frequency=fs, gain=0.33)
        q = [1.0, 0.0, 0.0, 0.0]
        quats = []
        for i in range(len(gyro_df)):
            gyr = gyro_df.loc[i, ['gx', 'gy', 'z']].tolist()
            acc = acc_df.loc[i, ['ax', 'ay', 'az']].tolist()
            q = madgwick.updateIMU(q=q, gyr=gyr, acc=acc)
            quats.append([gyro_df['t'][i]/1000.0, *q])
        quat_df = pd.DataFrame(quats, columns=['time', 'w', 'x', 'y', 'z'])

        self.update_state(state='PROGRESS', meta={'progress': 20})

        y = savgol_filter(gyro_df['gy'], window_length=11, polyorder=3)
        mu, sigma = y.mean(), y.std()
        peaks, _ = find_peaks(y, height=mu + sigma)
        valleys, _ = find_peaks(-y, height=abs(mu - sigma))
        t_sec = (gyro_df['t'] - gyro_df['t'].iloc[0]) / 1000.0

        loops = []
        i = 0
        while i < len(valleys) - 1:
            v1 = valleys[i]
            ps = [p for p in peaks if p > v1 and y[p] > mu + sigma]
            if ps:
                p = ps[0]
                vs2 = [v for v in valleys if v > p and y[v] < mu - sigma]
                if vs2 and (t_sec.iloc[vs2[0]] - t_sec.iloc[v1] <= 1.0):
                    loops.append((v1, p, vs2[0]))
                    i = valleys.tolist().index(vs2[0])
                    continue
            i += 1

        self.update_state(state='PROGRESS', meta={'progress': 35})

        segments = []
        for v1, p, v2 in loops:
            t_start, t_end = t_sec.iloc[v1], t_sec.iloc[v2]
            mask = (quat_df['time'] >= t_start) & (quat_df['time'] <= t_end)
            segments.append(quat_df[mask].reset_index(drop=True))

        n = len(segments)
        dtw_mat = np.zeros((n, n))
        for a in range(n):
            for b in range(n):
                dw, _ = fastdtw(segments[a]['w'], segments[b]['w'], dist=lambda x, y: abs(x - y))
                dx, _ = fastdtw(segments[a]['x'], segments[b]['x'], dist=lambda x, y: abs(x - y))
                dy, _ = fastdtw(segments[a]['y'], segments[b]['y'], dist=lambda x, y: abs(x - y))
                dz, _ = fastdtw(segments[a]['z'], segments[b]['z'], dist=lambda x, y: abs(x - y))
                dtw_mat[a, b] = dw + dx + dy + dz

        self.update_state(state='PROGRESS', meta={'progress': 55})

        vals = dtw_mat[np.triu_indices(n, 1)]
        if vals.size > 0:
            if vals.max() != vals.min():
                norm = (vals - vals.min()) / (vals.max() - vals.min())
                score = float((100 * (1.0 - norm)).mean())
            else:
                score = 100.0
        else:
            score = 0.0

        loop_durations = [float(t_sec.iloc[v2] - t_sec.iloc[v1]) for v1, _, v2 in loops]
        loop_mean = float(np.mean(loop_durations)) if loop_durations else None
        loop_std = float(np.std(loop_durations)) if loop_durations else None

        snap_values = []
        for v1, _, v2 in loops:
            t_start = t_sec.iloc[v1]
            t_end = t_sec.iloc[v2]
            acc_seg = acc_df[(acc_df['t']/1000 >= t_start) & (acc_df['t']/1000 <= t_end)]
            if not acc_seg.empty:
                norm = np.sqrt(acc_seg['ax']**2 + acc_seg['ay']**2 + acc_seg['az']**2)
                snap_values.append(norm.max())

        snap_median = float(np.median(snap_values)) if snap_values else None
        snap_std = float(np.std(snap_values)) if snap_values else None

        self.update_state(state='PROGRESS', meta={'progress': 90})

        loop_duration_list = [
            f"ループ {i+1}: {t_sec.iloc[v2] - t_sec.iloc[v1]:.3f} 秒 / {np.sqrt(acc_df[(acc_df['t']/1000 >= t_sec.iloc[v1]) & (acc_df['t']/1000 <= t_sec.iloc[v2])][['ax','ay','az']].pow(2).sum(axis=1)).max():.2f} m/s²"
            for i, (v1, _, v2) in enumerate(loops)
        ]

        result = {
            'score': score,
            'total_score': score,  # 仮（本来はレーダーの平均）
            'loop_count': n,
            'loop_mean_duration': loop_mean,
            'loop_std_duration': loop_std,
            'snap_median': snap_median,
            'snap_std': snap_std,
            'loop_duration_list': loop_duration_list
        }

        self.update_state(state='PROGRESS', meta={'progress': 100})
        return result

    except Exception as e:
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise

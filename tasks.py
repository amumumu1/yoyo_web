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
def analyze_task(acc_data, gyro_data):
    acc_df = pd.DataFrame(acc_data)
    gyro_df = pd.DataFrame(gyro_data)
    gyro_df['z'] = pd.to_numeric(gyro_df['gz'], errors='coerce')

    dt = (acc_df['t'].iloc[1] - acc_df['t'].iloc[0]) / 1000.0
    fs = 1.0 / dt

    mad = Madgwick(frequency=fs, gain=0.33)
    q = [1.0, 0.0, 0.0, 0.0]
    quats = []
    for i in range(len(gyro_df)):
        gyr = gyro_df.loc[i, ['gx', 'gy', 'z']].tolist()
        acc = acc_df.loc[i, ['ax', 'ay', 'az']].tolist()
        q = mad.updateIMU(q=q, gyr=gyr, acc=acc)
        quats.append([gyro_df['t'][i] / 1000.0, *q])
    quat_df = pd.DataFrame(quats, columns=['time', 'w', 'x', 'y', 'z'])

    y = savgol_filter(gyro_df['gy'], window_length=11, polyorder=3)
    mu, sigma = y.mean(), y.std()
    peaks, _ = find_peaks(y, height=mu + sigma)
    valleys, _ = find_peaks(-y, height=abs(mu - sigma))

    loops = []
    t_sec = (gyro_df['t'] - gyro_df['t'].iloc[0]) / 1000.0
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

    segments = []
    for v1, p, v2 in loops:
        mask = (quat_df['time'] >= t_sec.iloc[v1]) & (quat_df['time'] <= t_sec.iloc[v2])
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
    orig_self_mat = dtw_mat.copy()

    distances = []
    try:
        gyro_pro, quats_pro = load_and_compute_quaternions("3_acc2.csv", "3_gyro2.csv")
        pro_segments = segment_loops(gyro_pro, quats_pro)
        if len(pro_segments) >= 3:
            M = len(pro_segments)
            pro_dtw = np.zeros((M, M))
            for i in range(M):
                for j in range(M):
                    pro_dtw[i, j] = sum(
                        fastdtw(pro_segments[i][k], pro_segments[j][k], dist=lambda a, b: abs(a - b))[0]
                        for k in ['w', 'x', 'y', 'z']
                    )
            valid_indices = np.arange(M)[1:-1]
            row_sums = pro_dtw.sum(axis=1)
            ref_idx = valid_indices[np.argmin(row_sums[1:-1])]
            ref_loop = pro_segments[ref_idx]

            distances = [
                sum(fastdtw(seg[k], ref_loop[k], dist=lambda a, b: abs(a - b))[0] for k in ['w', 'x', 'y', 'z'])
                for seg in segments
            ]
        else:
            distances = [0] * len(segments)
    except Exception as e:
        print("プロ比較エラー:", e)
        distances = [0] * len(segments)

    self_heatmap_b64 = encode_heatmap(orig_self_mat, 'Self Loop Similarity')
    pro_mat = np.full((n, n), np.nan)
    for i, d in enumerate(distances):
        pro_mat[i, i] = d
    pro_heatmap_b64 = encode_heatmap(pro_mat, 'Pro vs Each Loop (Diagonal Only)')

    loop_indices = list(range(1, n + 1))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(loop_indices, distances, color='skyblue', edgecolor='black')
    ax.set_title("プロと各ループの距離比較", fontproperties=font_prop)
    ax.set_xlabel("ループ番号", fontproperties=font_prop)
    ax.set_ylabel("DTW距離", fontproperties=font_prop)
    ax.set_xticks(loop_indices)
    ax.grid(True)
    buf = BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    compare_plot_b64 = base64.b64encode(buf.getvalue()).decode('ascii')

    vals = dtw_mat[np.triu_indices(n, 1)]
    if vals.size > 0:
        if len(vals) == 0 or np.isnan(vals).any():
            norm = np.zeros_like(vals)
        elif vals.max() == vals.min():
            norm = np.ones_like(vals)
        else:
            norm = (vals - vals.min()) / (vals.max() - vals.min())
        score = float((100 * (1.0 - norm)).mean())
    else:
        score = 0.0

    stable_loop = detect_stable_loop_by_tail(dtw_mat)
    loop_durations = [float(t_sec.iloc[v2] - t_sec.iloc[v1]) for v1, _, v2 in loops]
    loop_mean = float(np.mean(loop_durations)) if loop_durations else None
    loop_std = float(np.std(loop_durations)) if loop_durations else None

    pro_distance_mean = float(np.mean(distances)) if distances else None
    radar_b64, total_score = generate_radar_chart(
        score=score,
        loop_mean=loop_mean,
        loop_std=loop_std,
        stable_loop=stable_loop or 7,
        pro_distance=pro_distance_mean
    )

    loop_max_acc_list = []
    for i, (v1, p, v2) in enumerate(loops):
        t_start = t_sec.iloc[v1]
        t_end = t_sec.iloc[v2]
        acc_segment = acc_df[(acc_df['t'] / 1000 >= t_start) & (acc_df['t'] / 1000 <= t_end)]
        if not acc_segment.empty:
            norm = np.sqrt(acc_segment['ax']**2 + acc_segment['ay']**2 + acc_segment['az']**2)
            max_norm = norm.max()
            loop_max_acc_list.append(f"ループ {i+1}: {max_norm:.3f} m/s²")

    result = {
        'score': score,
        'total_score': total_score,
        'stable_loop': stable_loop,
        'pro_distance_mean': pro_distance_mean,
        'self_heatmap': self_heatmap_b64,
        'pro_heatmap': pro_heatmap_b64,
        'compare_plot': compare_plot_b64,
        'radar_chart': radar_b64,
        'loop_max_acc_list': loop_max_acc_list
    }

    self.update_state(state='PROGRESS', meta={'progress': 100})
    return result

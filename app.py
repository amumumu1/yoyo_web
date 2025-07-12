import matplotlib
matplotlib.use('Agg')
from flask_cors import CORS 
from flask import Flask, request, jsonify, send_file
import pandas as pd, numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from fastdtw import fastdtw
from ahrs.filters import Madgwick
import base64
import math  # ← 追加

plt.rcParams['font.family'] = 'Hiragino Sans'

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # JSON 受信
    payload = request.get_json(force=True)
    acc_df  = pd.DataFrame(payload['acc'])
    gyro_df = pd.DataFrame(payload['gyro'])

    # JSON の gz を z 列に変換
    gyro_df['z'] = pd.to_numeric(gyro_df['gz'], errors='coerce')

    # サンプリングレート算出
    dt = (acc_df['t'].iloc[1] - acc_df['t'].iloc[0]) / 1000.0
    fs = 1.0 / dt

    # Madgwick フィルタでクォータニオン計算
    mad = Madgwick(frequency=fs, gain=0.33)
    q = [1.0, 0.0, 0.0, 0.0]
    quats = []
    for i in range(len(gyro_df)):
        gyr = gyro_df.loc[i, ['gx','gy','z']].tolist()
        acc = acc_df.loc[i, ['ax','ay','az']].tolist()
        q = mad.updateIMU(q=q, gyr=gyr, acc=acc)
        quats.append([gyro_df['t'][i]/1000.0, *q])
    quat_df = pd.DataFrame(quats, columns=['time','w','x','y','z'])

    # y軸成分をサヴォルフィルタ
    y = savgol_filter(gyro_df['gy'], window_length=11, polyorder=3)
    mu, sigma = y.mean(), y.std()

    # ピーク・谷検出
    peaks, _   = find_peaks(y,   height=mu+sigma)
    valleys, _ = find_peaks(-y,  height=abs(mu-sigma))

    # ループ区間検出
    loops = []
    t_sec = (gyro_df['t'] - gyro_df['t'].iloc[0]) / 1000.0
    i = 0
    while i < len(valleys) - 1:
        v1 = valleys[i]
        ps = [p for p in peaks if p > v1 and y[p] > mu+sigma]
        if ps:
            p = ps[0]
            vs2 = [v for v in valleys if v > p and y[v] < mu-sigma]
            if vs2 and (t_sec.iloc[vs2[0]] - t_sec.iloc[v1] <= 1.0):
                loops.append((v1, p, vs2[0]))
                i = valleys.tolist().index(vs2[0])
                continue
        i += 1

    # DTW 距離行列計算
    n = len(loops)
    dtw_mat = np.zeros((n, n))
    segments = []
    for v1, p, v2 in loops:
        mask = (quat_df['time'] >= t_sec.iloc[v1]) & (quat_df['time'] <= t_sec.iloc[v2])
        segments.append(quat_df[mask].reset_index(drop=True))
    for a in range(n):
        for b in range(n):
            dw, _ = fastdtw(segments[a]['w'], segments[b]['w'], dist=lambda x,y: abs(x-y))
            dx, _ = fastdtw(segments[a]['x'], segments[b]['x'], dist=lambda x,y: abs(x-y))
            dy, _ = fastdtw(segments[a]['y'], segments[b]['y'], dist=lambda x,y: abs(x-y))
            dz, _ = fastdtw(segments[a]['z'], segments[b]['z'], dist=lambda x,y: abs(x-y))
            dtw_mat[a, b] = dw + dx + dy + dz

    # ヒートマップを PNG 生成
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(dtw_mat, cmap='coolwarm')
    plt.colorbar(cax)
    ax.set_title('Loop Similarity')
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    heatmap_b64 = base64.b64encode(buf.getvalue()).decode('ascii')


    # スコア算出
    vals = dtw_mat[np.triu_indices(n, 1)]
    if vals.size > 0:
        norm = (vals - vals.min()) / (vals.max() - vals.min())
        score = float((100 * (1.0 - norm)).mean())
    else:
        score = 0.0

    def detect_stable_loop_by_tail(dtw_matrix):
        N = dtw_matrix.shape[0]
        if N == 0:
            return None  # ← 追加
        tail_len = N // 2
        vals = dtw_matrix[np.triu_indices(N, k=1)]
        d_min, d_max = vals.min(), vals.max()
        threshold = (d_min + d_max) / 2  
    

        ref_idx = list(range(N - tail_len, N))

        for i in range(N - tail_len):
            mean_dist = dtw_matrix[i, ref_idx].mean()
            if mean_dist <= threshold:
                return i + 1  # 1始まり
        return None

    stable_loop = detect_stable_loop_by_tail(dtw_mat)
   
   # 各ループの所要時間（秒）を計算
    loop_durations = [
    float(t_sec.iloc[v2] - t_sec.iloc[v1])
    for v1, _, v2 in loops
    ]

    if loop_durations:
        loop_mean_duration = float(np.mean(loop_durations))
        loop_std_duration  = float(np.std(loop_durations))
        loop_duration_list = [
            f"ループ {i+1}: {d:.3f} 秒" for i, d in enumerate(loop_durations)
        ]
    else:
        loop_mean_duration = None  # ← 修正
        loop_std_duration  = None  # ← 修正
        loop_duration_list = []    # ← 空リストのままでOK

        # ====== ループ検出グラフの描画 ======
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(t_sec, y, color='orange')

    # 検出されたループ領域を塗る
    for idx, (v1, p, v2) in enumerate(loops):
        ax2.axvspan(t_sec.iloc[v1], t_sec.iloc[v2], color='red', alpha=0.3, label='1周' if idx == 0 else "")

    # ピーク・谷にマーカー
    ax2.plot(t_sec.iloc[peaks], y[peaks], "go", label="ピーク")
    ax2.plot(t_sec.iloc[valleys], y[valleys], "ro", label="谷")

    ax2.set_title("ループ検出")
    ax2.set_xlabel("時間 [秒]")
    ax2.set_ylabel("角速度 gy [rad/s]")
    ax2.legend()
    ax2.grid(True)

    buf2 = BytesIO()
    fig2.savefig(buf2, format='png')
    plt.close(fig2)
    loop_plot_b64 = base64.b64encode(buf2.getvalue()).decode('ascii')




    

    if len(loops) < 2:
        # ----- セグメントグラフ（最低限の描画） -----
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(t_sec, y, color='orange', label='gy (filtered)')
        ax2.set_xlabel("時間 [秒]")
        ax2.set_ylabel("角速度 gy [rad/s]")
        ax2.set_title("ループ検出グラフ")
        ax2.grid(True)
        ax2.legend()

        buf2 = BytesIO()
        fig2.savefig(buf2, format='png')
        plt.close(fig2)
        loop_plot_b64 = base64.b64encode(buf2.getvalue()).decode('ascii')

        return jsonify({
            'score': 0.0,
            'heatmap': heatmap_b64,
            'loop_plot': loop_plot_b64,
            'stable_loop': None,
            'loop_count': len(loops),
            'loop_mean_duration': None,
            'loop_std_duration': None,
            'loop_duration_list': []
        })
    else:
        return jsonify({
            'score': score,
            'heatmap': heatmap_b64,
            'loop_plot': loop_plot_b64,
            'stable_loop': stable_loop,  # ← Noneなら null として返る
            'loop_count': n , 
            'loop_mean_duration': loop_mean_duration,  # 平均
            'loop_std_duration': loop_std_duration,    # 標準偏差
            'loop_duration_list': loop_duration_list   # 各ループの文字列リスト
    })
        

    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

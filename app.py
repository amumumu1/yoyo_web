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

    return jsonify({'score': score, 'heatmap': heatmap_b64})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

from flask import Flask, request, jsonify, send_file
import pandas as pd, numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from fastdtw import fastdtw
from ahrs.filters import Madgwick
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    acc_df  = pd.DataFrame(data['acc'])
    gyro_df = pd.DataFrame(data['gyro'])
    gyro_df['z'] = pd.to_numeric(gyro_df['z'], errors='coerce')

    dt = (acc_df['t'].iloc[1] - acc_df['t'].iloc[0]) / 1000.0
    fs = 1.0 / dt

    mad = Madgwick(frequency=fs, gain=0.33)
    q = [1.0,0.0,0.0,0.0]
    quats = []
    for i in range(len(gyro_df)):
        gyr = gyro_df.loc[i, ['x','y','z']].tolist()
        acc = acc_df.loc[i, ['x','y','z']].tolist()
        q = mad.updateIMU(q=q, gyr=gyr, acc=acc)
        quats.append([gyro_df['t'][i]/1000, *q])
    quat_df = pd.DataFrame(quats, columns=['time','w','x','y','z'])

    # yフィルタ＆ピーク・谷検出
    y = savgol_filter(gyro_df['y'], window_length=11, polyorder=3)
    mu, sigma = y.mean(), y.std()
    peaks, _   = find_peaks(y,   height=mu+sigma)
    valleys, _ = find_peaks(-y,  height=abs(mu-sigma))

    # ループ検出
    loops=[]; i=0; t_sec=(gyro_df['t']-gyro_df['t'][0])/1000
    while i < len(valleys)-1:
        v1=valleys[i]
        ps=[p for p in peaks if p>v1 and y[p]>mu+sigma]
        if ps:
            p=ps[0]; vs2=[v for v in valleys if v>p and y[v]<mu-sigma]
            if vs2 and (t_sec.iloc[vs2[0]]-t_sec.iloc[v1])<=1:
                loops.append((v1,p,vs2[0])); i=valleys.tolist().index(vs2[0]); continue
        i+=1

    # DTW行列
    n=len(loops)
    dtw_mat = np.zeros((n,n))
    segs=[]
    for v1,p,v2 in loops:
        mask=(quat_df['time']>=t_sec.iloc[v1])&(quat_df['time']<=t_sec.iloc[v2])
        segs.append(quat_df[mask].reset_index(drop=True))
    for a in range(n):
        for b in range(n):
            dw=fastdtw(segs[a]['w'], segs[b]['w'], dist=lambda x,y:abs(x-y))[0]
            dx=fastdtw(segs[a]['x'], segs[b]['x'], dist=lambda x,y:abs(x-y))[0]
            dy=fastdtw(segs[a]['y'], segs[b]['y'], dist=lambda x,y:abs(x-y))[0]
            dz=fastdtw(segs[a]['z'], segs[b]['z'], dist=lambda x,y:abs(x-y))[0]
            dtw_mat[a,b] = dw+dx+dy+dz

    # ヒートマップ作成
    fig, ax = plt.subplots(figsize=(6,6))
    cax = ax.matshow(dtw_mat, cmap='coolwarm')
    plt.colorbar(cax)
    plt.title('Loop Similarity')
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    heatmap_b64 = base64.b64encode(buf.getvalue()).decode('ascii')

    # スコア算出
    vals = dtw_mat[np.triu_indices(n,1)]
    if len(vals)>0:
        norm = (vals-vals.min())/(vals.max()-vals.min())
        score = float((100*(1-norm)).mean())
    else:
        score = 0.0

    return jsonify({'score': score, 'heatmap': heatmap_b64})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
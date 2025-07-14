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
from matplotlib import font_manager

font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()


#plt.rcParams['font.family'] = 'Hiragino Sans'

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return send_file('index.html')



def load_and_compute_quaternions(acc_csv, gyro_csv, gain=0.33):
        acc = pd.read_csv(acc_csv)
        gyro = pd.read_csv(gyro_csv)
        gyro["z"] = pd.to_numeric(gyro["z"], errors="coerce")
        dt = (acc['time'].iloc[1] - acc['time'].iloc[0]) / 1000.0
        fs = 1.0 / dt

        q = [1.0, 0.0, 0.0, 0.0]
        madgwick = Madgwick(frequency=fs, gain=gain)
        quats = []
        for i in range(len(gyro)):
            gyr = [gyro.at[i,'x'], gyro.at[i,'y'], gyro.at[i,'z']]
            a   = [acc.at[i,'x'],  acc.at[i,'y'],  acc.at[i,'z']]
            q = madgwick.updateIMU(q=q, gyr=gyr, acc=a)
            quats.append([gyro.at[i,'time']/1000, *q])
        return gyro, pd.DataFrame(quats, columns=["time","w","x","y","z"])

def segment_loops(gyro, quats):
    time_data = (gyro['time'] - gyro['time'].iloc[0]) / 1000.0
    y_raw = gyro['y']
    y_smooth = savgol_filter(y_raw, window_length=11, polyorder=3)
    mean_y = np.mean(y_smooth)
    std_y = np.std(y_smooth)
    peak_th = mean_y + std_y
    valley_th = mean_y - std_y
    peaks, _ = find_peaks(y_smooth, height=peak_th)
    valleys, _ = find_peaks(-y_smooth, height=abs(valley_th))

    valid_loops = []
    i = 0
    while i < len(valleys) - 1:
        v1 = valleys[i]
        possible_peaks = [p for p in peaks if p > v1 and y_smooth[p] > peak_th]
        if not possible_peaks:
            i += 1
            continue
        p = possible_peaks[0]
        possible_valleys2 = [v for v in valleys if v > p and y_smooth[v] < valley_th]
        if not possible_valleys2:
            i += 1
            continue
        v2 = possible_valleys2[0]
        if y_smooth[p] - y_smooth[v1] >= 0 and y_smooth[p] - y_smooth[v2] >= 0 and time_data[v2] - time_data[v1] <= 1:
            valid_loops.append((v1, p, v2))
            i = list(valleys).index(v2)
        else:
            i += 1

    segments = []
    for v1, _, v2 in valid_loops:
        t_start = time_data[v1]
        t_end = time_data[v2]
        mask = (quats["time"] >= t_start) & (quats["time"] <= t_end)
        segments.append(quats[mask].reset_index(drop=True))
    return segments

@app.route('/analyze', methods=['POST'])
def analyze():

    # プロのデータ読み込みとクォータニオン生成
    pro_acc_path = "3_acc2.csv"
    pro_gyro_path = "3_gyro2.csv"
    gyro_pro, quats_pro = load_and_compute_quaternions(pro_acc_path, pro_gyro_path)
    pro_segments = segment_loops(gyro_pro, quats_pro)

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
    tick_labels = [str(i+1) for i in range(dtw_mat.shape[0])]
    ax.set_xticks(range(dtw_mat.shape[0]))
    ax.set_yticks(range(dtw_mat.shape[0]))
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    heatmap_b64 = base64.b64encode(buf.getvalue()).decode('ascii')




    # スコア算出
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

    def detect_stable_loop_by_tail(dtw_matrix):
        N = dtw_matrix.shape[0]
        if N < 2:
            return None  # ループが1個以下なら安定性評価できない

        vals = dtw_matrix[np.triu_indices(N, k=1)]
        if vals.size == 0:
            return None  # これが今回のクラッシュ防止

        d_min, d_max = vals.min(), vals.max()
        threshold = (d_min + d_max) / 2  

        tail_len = N // 2
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
        ax2.axvspan(t_sec.iloc[v1], t_sec.iloc[v2], color='red', alpha=0.3, label='1周' if idx == 0 else "" )

    # ピーク・谷にマーカー
    ax2.plot(t_sec.iloc[peaks], y[peaks], "go", label="ピーク")
    ax2.plot(t_sec.iloc[valleys], y[valleys], "ro", label="谷")

    ax2.set_title("ループ検出" ,fontproperties=font_prop)
    ax2.set_xlabel("時間 [秒]",fontproperties=font_prop)
    ax2.set_ylabel("角速度 gy [rad/s]",fontproperties=font_prop)
    ax2.legend(prop=font_prop)
    ax2.grid(True)

    buf2 = BytesIO()
    fig2.savefig(buf2, format='png')
    plt.close(fig2)
    loop_plot_b64 = base64.b64encode(buf2.getvalue()).decode('ascii')

    


    # ==== プロ代表ループとの比較グラフ生成 ====
    try:
        gyro_pro, quats_pro = load_and_compute_quaternions("3_acc2.csv", "3_gyro2.csv")
        pro_segments = segment_loops(gyro_pro, quats_pro)
        if len(pro_segments) >= 3:
            M = len(pro_segments)
            pro_dtw = np.zeros((M, M))
            for i in range(M):
                for j in range(M):
                    pro_dtw[i,j] = sum(
                        fastdtw(pro_segments[i][k], pro_segments[j][k], dist=lambda a,b:abs(a-b))[0]
                        for k in ['w', 'x', 'y', 'z']
                    )
            valid_indices = np.arange(M)[1:-1]
            row_sums = pro_dtw.sum(axis=1)
            ref_idx = valid_indices[np.argmin(row_sums[1:-1])]
            ref_loop = pro_segments[ref_idx]

            # ユーザーとの比較
            distances = [
                sum(fastdtw(seg[k], ref_loop[k], dist=lambda a,b:abs(a-b))[0] for k in ['w','x','y','z'])
                for seg in segments
            ]
            loop_indices = list(range(1, len(segments)+1))
            fig3, ax3 = plt.subplots(figsize=(8, 4))
            ax3.bar(loop_indices, distances, color='skyblue', edgecolor='black')
            ax3.set_title("プロと各ループの距離比較", fontproperties=font_prop)
            ax3.set_xlabel("あなたのループ番号", fontproperties=font_prop)
            ax3.set_ylabel("DTW距離", fontproperties=font_prop)
            ax3.set_xticks(loop_indices)
            ax3.grid(True)
            buf3 = BytesIO()
            fig3.savefig(buf3, format='png')
            plt.close(fig3)
            compare_plot_b64 = base64.b64encode(buf3.getvalue()).decode('ascii')
        else:
            compare_plot_b64 = None
    except Exception as e:
        print("プロ比較エラー:", e)
        compare_plot_b64 = None







    

    if len(loops) < 2:
        # ----- セグメントグラフ（最低限の描画） -----
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(t_sec, y, color='orange', label='gy (filtered)')
        ax2.set_xlabel("時間 [秒]",fontproperties=font_prop)
        ax2.set_ylabel("角速度 gy [rad/s]",fontproperties=font_prop)
        ax2.set_title("ループ検出グラフ" ,fontproperties=font_prop)
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
            'loop_duration_list': [],
            'compare_plot': compare_plot_b64
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
            'loop_duration_list': loop_duration_list,  # 各ループの文字列リスト
            'compare_plot': compare_plot_b64
    })
        

    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

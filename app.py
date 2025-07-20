import matplotlib
matplotlib.use('Agg')
from flask_cors import CORS 
import pandas as pd, numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from fastdtw import fastdtw
from ahrs.filters import Madgwick
import base64
import math  # ← 追加
from matplotlib import font_manager
import sqlite3
import os
from flask import Flask, request, jsonify, send_file, render_template_string
from datetime import datetime
from flask import Response
from datetime import datetime, timedelta

font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()


#plt.rcParams['font.family'] = 'Hiragino Sans'

app = Flask(__name__)
CORS(app)

DB_PATH = "results.db"

# --- データベース初期化 ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        name TEXT,
        score REAL,
        loop_count INTEGER,
        stable_loop INTEGER,
        loop_mean_duration REAL,
        loop_std_duration REAL,
        loop_plot TEXT,
        self_heatmap TEXT,
        heatmap TEXT,
        pro_heatmap TEXT,
        compare_plot TEXT,
        acc_csv TEXT,
        gyro_csv TEXT
    )
    """)
    conn.commit()
    conn.close()

init_db()

# --- DBに結果を保存 ---
def save_result_to_db(result):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    jst_now = datetime.utcnow() + timedelta(hours=9)
    cur.execute("""
        INSERT INTO results (
            timestamp, name, score, loop_count, stable_loop,
            loop_mean_duration, loop_std_duration,
            loop_plot, self_heatmap, heatmap, pro_heatmap, compare_plot, acc_csv, gyro_csv
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        jst_now.strftime("%Y-%m-%d %H:%M:%S"),  # ← JSTを保存
        result.get("name"),
        result.get("score"),
        result.get("loop_count"),
        result.get("stable_loop"),
        result.get("loop_mean_duration"),
        result.get("loop_std_duration"),
        result.get("loop_plot"),
        result.get("self_heatmap"),
        result.get("heatmap"),
        result.get("pro_heatmap"),
        result.get("compare_plot"),
        result.get("acc_csv"),  # 追加　
        result.get("gyro_csv")  # 追加
    ))
    conn.commit()
    conn.close()

# --- 履歴一覧をJSONで返す ---
@app.route("/results", methods=["GET"])
def get_results():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, timestamp, name, score, loop_count, stable_loop FROM results ORDER BY id DESC LIMIT 20")
    rows = cur.fetchall()
    conn.close()
    return jsonify([
        {
            "id": r[0],
            "timestamp": r[1],
            "name": r[2] or "無題",
            "score": r[3],
            "loop_count": r[4],
            "stable_loop": r[5]
        }
        for r in rows
    ])

# --- 特定の結果を取得（グラフ画像含む） ---
@app.route("/results/<int:result_id>", methods=["GET"])
def get_result_detail(result_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT timestamp, name, score, loop_count, stable_loop,
               loop_mean_duration, loop_std_duration,
               loop_plot, self_heatmap, heatmap, pro_heatmap, compare_plot
        FROM results WHERE id = ?
    """, (result_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "Result not found"}), 404

    return jsonify({
        "timestamp": row[0],
        "name": row[1] or "無題",
        "score": row[2],
        "loop_count": row[3],
        "stable_loop": row[4],
        "loop_mean_duration": row[5],
        "loop_std_duration": row[6],
        "loop_plot": row[7],
        "self_heatmap": row[8],
        "heatmap": row[9],
        "pro_heatmap": row[10],
        "compare_plot": row[11]
    })

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


# --- スケーリング関数（0〜5に変換） ---
def scale_score(value, min_val, max_val, invert=False):
    if value is None or np.isnan(value):
        return 0
    if invert:
        # 値が小さいほどスコアが高い
        if value <= min_val: return 5
        if value >= max_val: return 0
        return 5 * (max_val - value) / (max_val - min_val)
    else:
        # 値が大きいほどスコアが高い
        if value <= min_val: return 0
        if value >= max_val: return 5
        return 5 * (value - min_val) / (max_val - min_val)

# --- レーダーチャート生成 ---
def generate_radar_chart(score, loop_mean, loop_std, stable_loop, pro_distance):
    s_score = scale_score(score, 0, 100) / 20  # 100点→5点（100/20=5）
    s_mean  = scale_score(loop_mean, 0.9, 0.4, invert=True)   # 0.4sで5, 0.9sで0
    s_std   = scale_score(loop_std, 0.2, 0.05, invert=True)   # 0.05で5, 0.2で0
    s_stable = scale_score(stable_loop, 7, 2, invert=True)    # 2周目で5, 7周目で0
    s_pro   = scale_score(pro_distance, 70, 20, invert=True)  # 20で5, 70で0

    labels = ['類似度スコア', '平均ループ時間', '時間の標準偏差', '安定開始ループ', 'プロ距離']
    values = [s_score, s_mean, s_std, s_stable, s_pro]
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels) + 1, endpoint=True)

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(angles[:-1] * 180/np.pi, labels, fontproperties=font_prop)
    ax.plot(angles, values, color='blue', linewidth=2)
    ax.fill(angles, values, color='skyblue', alpha=0.4)
    ax.set_ylim(0, 5)
    ax.grid(True)

    buf = BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('ascii')

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



    # --- ユーザー側ループのsegmentsを生成 ---
    segments = []
    for v1, p, v2 in loops:
        mask = (quat_df['time'] >= t_sec.iloc[v1]) & (quat_df['time'] <= t_sec.iloc[v2])
        segments.append(quat_df[mask].reset_index(drop=True))

     # --- プロ距離を上書きする前に、自分自身のDTW行列を保存 ---
    self_dtw_mat = dtw_mat.copy()


    # --- プロ代表ループとの比較 ---
    distances = []
    try:
        gyro_pro, quats_pro = load_and_compute_quaternions("3_acc2.csv", "3_gyro2.csv")
        pro_segments = segment_loops(gyro_pro, quats_pro)
        if len(pro_segments) >= 3:
            # プロ代表ループを選定
            M = len(pro_segments)
            pro_dtw = np.zeros((M, M))
            for i in range(M):
                for j in range(M):
                    pro_dtw[i,j] = sum(
                        fastdtw(pro_segments[i][k], pro_segments[j][k], dist=lambda a,b:abs(a-b))[0]
                        for k in ['w','x','y','z']
                    )
            valid_indices = np.arange(M)[1:-1]
            row_sums = pro_dtw.sum(axis=1)
            ref_idx = valid_indices[np.argmin(row_sums[1:-1])]
            ref_loop = pro_segments[ref_idx]

            # 各ループとの距離を計算
            distances = [
                sum(fastdtw(seg[k], ref_loop[k], dist=lambda a,b:abs(a-b))[0] for k in ['w','x','y','z'])
                for seg in segments
            ]
        else:
            distances = [0]*len(segments)
    except Exception as e:
        print("プロ比較エラー:", e)
        distances = [0]*len(segments)

    # --- 対角線にプロ距離を埋め込む ---
    for i, d in enumerate(distances):
        dtw_mat[i, i] = d

    # --- ヒートマップ描画（これが唯一の比較結果）---
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(dtw_mat, cmap='coolwarm')
    plt.colorbar(cax)
    ax.set_title('Loop Similarity (Diagonal = Pro Distance)')
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

    # --- プロ比較専用のヒートマップ（対角線だけ距離、他は真っ白） ---
    pro_mat = np.full((n, n), np.nan)  # NaNで塗りつぶし（白表示用）
    for i, d in enumerate(distances):
        pro_mat[i, i] = d  # 対角線にプロ距離

    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(pro_mat, cmap='coolwarm')
    plt.colorbar(cax)
    ax.set_title('Pro vs Each Loop (Diagonal Only)')
    tick_labels = [str(i+1) for i in range(n)]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    pro_heatmap_b64 = base64.b64encode(buf.getvalue()).decode('ascii')

   
    # ... （プロとの距離を対角線に上書きする処理はそのまま）
    for i, d in enumerate(distances):
        dtw_mat[i, i] = d

    # --- 自分自身のオリジナルDTWヒートマップ ---
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(self_dtw_mat, cmap='coolwarm')
    plt.colorbar(cax)
    ax.set_title('Self Loop Similarity (Original)')
    tick_labels = [str(i+1) for i in range(n)]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    self_heatmap_b64 = base64.b64encode(buf.getvalue()).decode('ascii')











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


    # --- ループ検出グラフ描画 ---
    if len(loops) >= 2:
        # 詳細グラフ（2個以上）
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(t_sec, y, color='orange')
        for idx, (v1, p, v2) in enumerate(loops):
            ax2.axvspan(t_sec.iloc[v1], t_sec.iloc[v2], color='red', alpha=0.3, label='1周' if idx == 0 else "")
        ax2.plot(t_sec.iloc[peaks], y[peaks], "go", label="ピーク")
        ax2.plot(t_sec.iloc[valleys], y[valleys], "ro", label="谷")
        ax2.set_title("ループ検出", fontproperties=font_prop)
        ax2.set_xlabel("時間 [秒]", fontproperties=font_prop)
        ax2.set_ylabel("角速度 gy [rad/s]", fontproperties=font_prop)
        ax2.legend(prop=font_prop)
        ax2.grid(True)
        buf2 = BytesIO()
        fig2.savefig(buf2, format='png')
        plt.close(fig2)
        loop_plot_b64 = base64.b64encode(buf2.getvalue()).decode('ascii')
    else:
        # 簡易グラフ（1個以下）
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(t_sec, y, color='orange', label='gy (filtered)')
        ax2.set_xlabel("時間 [秒]", fontproperties=font_prop)
        ax2.set_ylabel("角速度 gy [rad/s]", fontproperties=font_prop)
        ax2.set_title("ループ検出グラフ", fontproperties=font_prop)
        ax2.grid(True)
        ax2.legend()
        buf2 = BytesIO()
        fig2.savefig(buf2, format='png')
        plt.close(fig2)
        loop_plot_b64 = base64.b64encode(buf2.getvalue()).decode('ascii')

    # 仮の距離（プロ比較用）
    # distances = [35] * n  # 例として35固定。実際はプロ比較で計算済みを使う

    # スコア仮計算
    # score = 100.0 if n >= 2 else 0.0
    # stable_loop = 3 if n >= 2 else None

    # レーダーチャート生成
    radar_b64 = generate_radar_chart(
        score=score,
        loop_mean=loop_mean_duration,
        loop_std=loop_std_duration,
        stable_loop=stable_loop if stable_loop else 7,
        pro_distance=np.mean(distances) if distances else 70
    )

    # --- 結果まとめ ---
    result = {
        'score': score if len(loops) >= 2 else 0.0,
        'self_heatmap': self_heatmap_b64,  # ← 純粋な自分同士比較
        'heatmap': heatmap_b64,
        'pro_heatmap': pro_heatmap_b64,  # プロ距離だけのヒートマップ（新規）
        'loop_plot': loop_plot_b64,
        'stable_loop': stable_loop if len(loops) >= 2 else None,
        'loop_count': n,
        'loop_mean_duration': loop_mean_duration if len(loops) >= 2 else None,
        'loop_std_duration': loop_std_duration if len(loops) >= 2 else None,
        'loop_duration_list': loop_duration_list if len(loops) >= 2 else [],
        'compare_plot': compare_plot_b64,
        'radar_chart': radar_b64
    }
    return jsonify(result)

# --- 新しい保存用のエンドポイント ---
@app.route("/save_result", methods=["POST"])
def save_result():
    result = request.get_json()
    if not result:
        return jsonify({"error": "No result data"}), 400
    save_result_to_db(result)
    return jsonify({"status": "saved"})

@app.route('/viewer')
def viewer():
    return send_file('viewer.html')

import urllib.parse

@app.route("/results/<int:result_id>/csv", methods=["GET"])
def download_result_csv(result_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT acc_csv, gyro_csv, timestamp, name FROM results WHERE id = ?", (result_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return jsonify({"error": "Result not found"}), 404

    acc_csv, gyro_csv, timestamp, name = row
    safe_name = name or "result"

    jst_dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")


    # ZIPを作成
    import io, zipfile
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for label, content in [("acc", acc_csv or ''), ("gyro", gyro_csv or '')]:
            zi = zipfile.ZipInfo(f"{name or 'result'}_{label}.csv")
            # ZIPに書き込むファイルの更新日時をJSTで設定
            zi.date_time = (jst_dt.year, jst_dt.month, jst_dt.day,
                            jst_dt.hour, jst_dt.minute, jst_dt.second)
            zf.writestr(zi, content)
    buffer.seek(0)

    # HTTPヘッダ用にURLエンコード（UTF-8対応）
    quoted_name = urllib.parse.quote(f"{safe_name}_csv.zip")

    return Response(
        buffer,
        mimetype="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{quoted_name}"
        }
    )

@app.route("/results/<int:result_id>", methods=["DELETE"])
def delete_result(result_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM results WHERE id = ?", (result_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "deleted", "id": result_id})








    

    
    
        

    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

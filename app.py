import json
import uuid
import matplotlib
matplotlib.use('Agg')
from flask_cors import CORS
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from fastdtw import fastdtw
from ahrs.filters import Madgwick
import base64
import math
from matplotlib import font_manager
import sqlite3
import os
from flask import Flask, request, jsonify, send_file, Response
from datetime import datetime, timedelta
import urllib.parse

# フォント設定
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

app = Flask(__name__)
CORS(app)

# ── 進捗管理用ストア ─────────────────────────────────
progress_store = {}

# ── データベース初期化 ─────────────────────────────────
DB_PATH = "results.db"
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        name TEXT,
        total_score REAL,
        radar_chart TEXT,
        score REAL,
        pro_distance_mean REAL,
        loop_count INTEGER,
        stable_loop INTEGER,
        loop_mean_duration REAL,
        loop_std_duration REAL,
        loop_plot TEXT,
        self_heatmap TEXT,
        heatmap TEXT,
        pro_heatmap TEXT,
        compare_plot TEXT,
        combined_heatmap TEXT,
        acc_csv TEXT,
        gyro_csv TEXT,
        snap_median REAL,
        snap_std REAL,
        loop_duration_list TEXT,
        loop_max_acc_list TEXT
    )
    """)
    conn.commit()
    conn.close()

def save_result_to_db(result):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    jst_now = datetime.utcnow() + timedelta(hours=9)
    result["loop_duration_list"] = json.dumps(result.get("loop_duration_list", []), ensure_ascii=False)
    result["loop_max_acc_list"]  = json.dumps(result.get("loop_max_acc_list", []), ensure_ascii=False)
    cur.execute("""
        INSERT INTO results (
            timestamp, name, total_score, radar_chart, score, pro_distance_mean,
            loop_count, stable_loop, loop_mean_duration, loop_std_duration,
            loop_plot, self_heatmap, heatmap, pro_heatmap, compare_plot, combined_heatmap,
            acc_csv, gyro_csv, snap_median, snap_std, loop_duration_list, loop_max_acc_list
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        jst_now.strftime("%Y-%m-%d %H:%M:%S"),
        result.get("name"),
        result.get("total_score"),
        result.get("radar_chart"),
        result.get("score"),
        result.get("pro_distance_mean"),
        result.get("loop_count"),
        result.get("stable_loop"),
        result.get("loop_mean_duration"),
        result.get("loop_std_duration"),
        result.get("loop_plot"),
        result.get("self_heatmap"),
        result.get("heatmap"),
        result.get("pro_heatmap"),
        result.get("compare_plot"),
        result.get("combined_heatmap"),
        result.get("acc_csv"),
        result.get("gyro_csv"),
        result.get("snap_median"),
        result.get("snap_std"),
        result.get("loop_duration_list"),
        result.get("loop_max_acc_list")
    ))
    conn.commit()
    conn.close()

init_db()

# ── 進捗 API ─────────────────────────────────────────
@app.route('/start_analysis', methods=['POST'])
def start_analysis():
    task_id = str(uuid.uuid4())
    progress_store[task_id] = {'progress': 0, 'message': 'キュー登録完了'}
    return jsonify({'task_id': task_id})

@app.route('/progress/<task_id>', methods=['GET'])
def get_progress(task_id):
    info = progress_store.get(task_id)
    if not info:
        return jsonify({'error': 'Unknown task_id'}), 404
    return jsonify(info)

# ── ヘルパー関数 ─────────────────────────────────────
def encode_heatmap(mat: np.ndarray, title: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(mat, cmap='coolwarm')
    plt.colorbar(cax)
    ax.set_title(title)
    ticks = list(range(mat.shape[0]))
    labels = [str(i+1) for i in ticks]
    ax.set_xticks(ticks); ax.set_xticklabels(labels)
    ax.set_yticks(ticks); ax.set_yticklabels(labels)
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('ascii')

def load_and_compute_quaternions(acc_csv, gyro_csv, gain=0.33):
    acc  = pd.read_csv(acc_csv)
    gyro = pd.read_csv(gyro_csv)
    gyro["z"] = pd.to_numeric(gyro["z"], errors="coerce")
    dt  = (acc['time'].iloc[1] - acc['time'].iloc[0]) / 1000.0
    fs  = 1.0 / dt
    q   = [1.0, 0.0, 0.0, 0.0]
    mad = Madgwick(frequency=fs, gain=gain)
    quats = []
    for i in range(len(gyro)):
        gyr = [gyro.at[i,'x'], gyro.at[i,'y'], gyro.at[i,'z']]
        a   = [acc.at[i,'x'],  acc.at[i,'y'],  acc.at[i,'z']]
        q   = mad.updateIMU(q=q, gyr=gyr, acc=a)
        quats.append([gyro.at[i,'time']/1000, *q])
    return gyro, pd.DataFrame(quats, columns=["time","w","x","y","z"])

def segment_loops(gyro, quats):
    time_data = (gyro['time'] - gyro['time'].iloc[0]) / 1000.0
    y_raw     = gyro['y']
    y_smooth  = savgol_filter(y_raw, window_length=11, polyorder=3)
    mean_y, std_y = y_smooth.mean(), y_smooth.std()
    peaks, _   = find_peaks(y_smooth, height=mean_y+std_y)
    valleys, _ = find_peaks(-y_smooth, height=std_y-mean_y)
    valid_loops = []
    i = 0
    while i < len(valleys)-1:
        v1 = valleys[i]
        ps = [p for p in peaks if p>v1 and y_smooth[p]>mean_y+std_y]
        if not ps:
            i += 1; continue
        p = ps[0]
        vs2 = [v for v in valleys if v>p and y_smooth[v]<mean_y-std_y]
        if not vs2:
            i += 1; continue
        v2 = vs2[0]
        if (y_smooth[p]-y_smooth[v1]>=0 and
            y_smooth[p]-y_smooth[v2]>=0 and
            time_data[v2]-time_data[v1]<=1.0):
            valid_loops.append((v1,p,v2))
            i = valleys.tolist().index(v2)
        else:
            i += 1
    segments = []
    for v1, _, v2 in valid_loops:
        t_start, t_end = time_data[v1], time_data[v2]
        mask = (quats["time"]>=t_start)&(quats["time"]<=t_end)
        segments.append(quats[mask].reset_index(drop=True))
    return segments

def generate_radar_chart(score, loop_mean, loop_std, stable_loop, pro_distance, loop_count):
    # 各指標を0〜5にスケーリング
    if score is None:         s_score=0
    elif score>=100:          s_score=5
    elif score<=0:            s_score=0
    else:                     s_score=(score/100)*5

    if loop_mean is None:     s_mean=0
    elif loop_mean<=0.4:      s_mean=5
    elif loop_mean>=0.9:      s_mean=0
    else:                     s_mean=5*(0.9-loop_mean)/(0.9-0.4)

    if loop_std is None:      s_std=0
    elif loop_std<=0.05:      s_std=5
    elif loop_std>=0.2:       s_std=0
    else:                     s_std=5*(0.2-loop_std)/(0.2-0.05)

    if stable_loop is None or loop_count is None:
        s_stable = 0
    else:
        # 総ループ数に応じてスケーリング
        scale = loop_count / 10.0  # 10周を基準
        max_full_score_loop = int(round(2 * scale))   # 5点の閾値
        min_full_zero_loop = int(round(7 * scale))    # 0点の閾値

        if stable_loop <= max_full_score_loop:
            s_stable = 5
        elif stable_loop >= min_full_zero_loop:
            s_stable = 0
        else:
            s_stable = 5 * (min_full_zero_loop - stable_loop) / (min_full_zero_loop - max_full_score_loop)


    if pro_distance is None:  s_pro=0
    elif pro_distance<=20:    s_pro=5
    elif pro_distance>=120:   s_pro=0
    else:                     s_pro=5*(120-pro_distance)/(120-20)

    labels = ['自身の類似度','平均ループ時間','ループ時間のばらつき','安定開始ループ','プロ類似度']
    values = [s_score, s_mean, s_std, s_stable, s_pro]
    avg_score = np.mean(values) * 20
    values += values[:1]
    angles = np.linspace(0, 2*np.pi, len(labels)+1, endpoint=True)

    fig, ax = plt.subplots(figsize=(10,10), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
    ax.grid(False); ax.set_frame_on(False); ax.set_thetagrids([])
    ax.set_rgrids([1,2,3,4,5], angle=0, fontproperties=font_prop, fontsize=20)
    for r in range(1,6):
        ax.plot(angles, [r]*(len(labels)+1), color='gray', linewidth=1)
    for angle, label in zip(angles[:-1], labels):
        ax.text(angle, 6.3, label, ha='center', va='center', fontsize=25, fontproperties=font_prop)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.4)

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('ascii'), float(avg_score)

# 安定開始ループ検出用
# def detect_stable_loop_by_tail(dtw_matrix):
#     N = dtw_matrix.shape[0]
#     if N < 2:
#         return None
#     vals = dtw_matrix[np.triu_indices(N, k=1)]
#     if vals.size == 0:
#         return None
#     d_min, d_max = vals.min(), vals.max()
#     # 0に近いほど閾値が厳しくなる　0.5が今まで通り　1に近いほど閾値が甘くなる
#     alpha = 0.3
#     threshold = d_min + alpha * (d_max - d_min)

#     tail_len = N // 2
#     ref_idx = list(range(N - tail_len, N))
#     for i in range(N - tail_len):
#         if dtw_matrix[i, ref_idx].mean() <= threshold:
#             return i + 1
#     return None

def detect_stable_loop_by_tail(dtw_matrix, threshold_ratio=0.227):
    """
    代表ループ基準で安定開始ループ数を検出
    dtw_matrix: N×N の DTW距離行列
    threshold_ratio: 代表ループとの距離がこの割合以下なら安定とみなす（例: 0.2で最大距離の20%以下）
    戻り値: 安定化を開始したループ番号（1始まり）または None
    """
    N = dtw_matrix.shape[0]
    if N < 2:
        return None

    # 代表ループ = 行ごとの距離合計が最小のループ
    row_sums = dtw_matrix.sum(axis=1)
    rep_idx = int(np.argmin(row_sums))

    # 代表ループとの距離配列
    rep_distances = dtw_matrix[rep_idx, :]

    # 閾値（最大距離の一定割合）
    max_dist = rep_distances.max()
    threshold = max_dist * threshold_ratio

    # 安定ループ群（代表ループとの距離が閾値以下）
    stable_loops = [i for i, d in enumerate(rep_distances) if d <= threshold]

    if not stable_loops:
        return None

    # 最初の安定ループ番号（1始まり）
    return min(stable_loops) + 1


# ── 解析エンドポイント ─────────────────────────────────
@app.route('/analyze', methods=['POST'])
def analyze():
    task_id = request.args.get('task_id')
    if not task_id or task_id not in progress_store:
        return jsonify({'error':'Invalid task_id'}),400

    # 0: 開始
    progress_store[task_id] = {'progress':0,  'message':'解析開始…'}

    # 1: プロデータ読込
    progress_store[task_id] = {'progress':5,  'message':'プロデータ読込中…'}
    pro_acc, pro_gyro = "3_acc2.csv", "3_gyro2.csv"
    gyro_pro, quats_pro = load_and_compute_quaternions(pro_acc, pro_gyro)

    # 2: プロループ抽出
    progress_store[task_id] = {'progress':10, 'message':'プロループ抽出中…'}
    pro_segments = segment_loops(gyro_pro, quats_pro)

    # 3: JSON受信
    progress_store[task_id] = {'progress':15, 'message':'入力データ受信…'}
    payload = request.get_json(force=True)

    # 4: データ前処理
    progress_store[task_id] = {'progress':20, 'message':'データ前処理…'}
    acc_df  = pd.DataFrame(payload['acc'])
    gyro_df = pd.DataFrame(payload['gyro'])
    gyro_df['z'] = pd.to_numeric(gyro_df['gz'], errors='coerce')
    t0 = acc_df['t'].iloc[0]
    dt = (acc_df['t'].iloc[1] - t0) / 1000.0

    # 5: クォータニオン計算
    progress_store[task_id] = {'progress':25, 'message':'クォータニオン変換…'}
    mad = Madgwick(frequency=1.0/dt, gain=0.33)
    q = [1,0,0,0]; quats = []
    for i in range(len(gyro_df)):
        gyr = gyro_df.loc[i, ['gx','gy','z']].tolist()
        a   = acc_df.loc[i, ['ax','ay','az']].tolist()
        q = mad.updateIMU(q=q, gyr=gyr, acc=a)
        quats.append([gyro_df['t'].iloc[i]/1000, *q])
    quat_df = pd.DataFrame(quats, columns=['time','w','x','y','z'])

    # 6: フィルタ＆ピーク検出
    progress_store[task_id] = {'progress':30, 'message':'極値検出…'}
    y = savgol_filter(gyro_df['gy'], window_length=11, polyorder=3)
    peaks, _   = find_peaks(y, height=y.mean()+y.std())
    valleys, _ = find_peaks(-y, height=y.std()-y.mean())

    # 7: ループ検出
    progress_store[task_id] = {'progress':35, 'message':'ループごとにセグメント…'}
    t_sec = (gyro_df['t'] - t0) / 1000.0
    loops = []
    i = 0
    while i < len(valleys)-1:
        v1 = valleys[i]
        ps = [p for p in peaks if p>v1 and y[p]>y.mean()+y.std()]
        if ps:
            p = ps[0]
            vs2 = [v for v in valleys if v>p and y[v]<y.std()-y.mean()]
            if vs2 and (t_sec.iloc[vs2[0]]-t_sec.iloc[v1])<=1.0:
                loops.append((v1, p, vs2[0]))
                i = valleys.tolist().index(vs2[0])
                continue
        i += 1

    # 8: 自己比較行列計算
    progress_store[task_id] = {'progress':40, 'message':'自身の類似度を計算…'}
    n = len(loops)
    dtw_mat = np.zeros((n,n))
    segments = []
    for v1,_,v2 in loops:
        mask = (quat_df['time']>=t_sec.iloc[v1]) & (quat_df['time']<=t_sec.iloc[v2])
        segments.append(quat_df[mask].reset_index(drop=True))
    for a in range(n):
        for b in range(n):
            dtw_mat[a,b] = sum(
                fastdtw(segments[a][k], segments[b][k], dist=lambda x,y:abs(x-y))[0]
                for k in ['w','x','y','z']
            )

    # 9: プロ比較距離計算
        # 9: プロ比較距離計算
    progress_store[task_id] = {'progress':55, 'message':'プロの類似度を計算…'}
    distances = []
    try:
        # プロ側のループから代表ループを選定
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

        # 各自分ループとの距離を計算（前のapp.pyと同じロジック）
        distances = [
            sum(
                fastdtw(seg[k], ref_loop[k], dist=lambda a, b: abs(a - b))[0]
                for k in ['w', 'x', 'y', 'z']
            )
            for seg in segments
        ]
    except Exception as e:
        print("プロ比較エラー:", e)
        distances = [0] * n


    # 10: Self ヒートマップ
    progress_store[task_id] = {'progress':60, 'message':'自身のヒートマップ作成…'}
    self_hm = encode_heatmap(dtw_mat, 'Self Loop Similarity')

    # 11: Pro ヒートマップ
    progress_store[task_id] = {'progress':65, 'message':'プロのヒートマップ作成…'}
    pro_mat = np.full_like(dtw_mat, np.nan)
    for i,d in enumerate(distances):
        pro_mat[i,i] = d
    pro_hm = encode_heatmap(pro_mat, 'Pro vs Each Loop')

    # 12: ループ検出グラフ
    progress_store[task_id] = {'progress':70, 'message':'セグメンテーショングラフ作成…'}
    fig2, ax2 = plt.subplots(figsize=(12,6))
    ax2.plot(t_sec, y, color='orange')
    for idx,(v1,p,v2) in enumerate(loops):
        ax2.axvspan(t_sec.iloc[v1], t_sec.iloc[v2], color='red', alpha=0.3,
                    label='1周' if idx==0 else "")
    ax2.plot(t_sec.iloc[peaks], y[peaks], "go", label="ピーク")
    ax2.plot(t_sec.iloc[valleys], y[valleys], "ro", label="谷")
    ax2.set_title("ループ検出", fontproperties=font_prop)
    ax2.set_xlabel("時間 [秒]", fontproperties=font_prop)
    ax2.set_ylabel("角速度 gy [rad/s]", fontproperties=font_prop)
    ax2.legend(prop=font_prop); ax2.grid(True)
    buf2 = BytesIO(); fig2.savefig(buf2, format='png'); plt.close(fig2)
    loop_plot_b64 = base64.b64encode(buf2.getvalue()).decode('ascii')

    # 13: プロ比較バーグラフ
    progress_store[task_id] = {'progress':75, 'message':'プロ比較グラフ作成…'}
    fig3, ax3 = plt.subplots(figsize=(8,4))
    idxs = list(range(1, n+1))
    ax3.bar(idxs, distances, edgecolor='black')
    ax3.set_title("プロと各ループの距離比較", fontproperties=font_prop)
    ax3.set_xlabel("あなたのループ番号", fontproperties=font_prop)
    ax3.set_ylabel("DTW距離", fontproperties=font_prop)
    ax3.set_xticks(idxs); ax3.grid(True)
    buf3 = BytesIO(); fig3.savefig(buf3, format='png'); plt.close(fig3)
    compare_plot_b64 = base64.b64encode(buf3.getvalue()).decode('ascii')

    # スコア算出
    vals = dtw_mat[np.triu_indices(n, 1)]
    if vals.size > 0 and not np.isnan(vals).any():
        norm = np.zeros_like(vals) if vals.max()==vals.min() else (vals - vals.min())/(vals.max()-vals.min())
        score = float((100*(1.0 - norm)).mean())
    else:
        score = 0.0

    # 14: 安定開始ループ検出
    progress_store[task_id] = {'progress':80, 'message':'安定開始ループ検出…'}
    # stable_loop = detect_stable_loop_by_tail(dtw_mat)
    stable_loop = detect_stable_loop_by_tail(dtw_mat, threshold_ratio=0.227)


    
    # 15: ループ時間＆最大加速度リスト
    progress_store[task_id] = {'progress':85, 'message':'ループ時間計算…'}

    loop_durations     = []
    loop_duration_list = []
    loop_max_acc_list  = []

    for i, (v1, _, v2) in enumerate(loops):
        # ループ時間
        t_start = t_sec.iloc[v1]
        t_end   = t_sec.iloc[v2]
        duration = t_end - t_start
        loop_durations.append(duration)

        # 加速度ノルム計算
        seg = acc_df[(acc_df['t']/1000 >= t_start) & (acc_df['t']/1000 <= t_end)]
        if not seg.empty:
            norm     = np.sqrt(seg['ax']**2 + seg['ay']**2 + seg['az']**2)
            max_norm = norm.max()
            # 秒と m/s² を一行にまとめる
            loop_duration_list.append(f"ループ {i+1}: {duration:.3f} 秒　{max_norm:.3f} m/s²")
            loop_max_acc_list.append(max_norm)
        else:
            loop_duration_list.append(f"ループ {i+1}: {duration:.3f} 秒　- m/s²")
            loop_max_acc_list.append(None)



    # 16: スナップ統計
    progress_store[task_id] = {'progress':90, 'message':'ノルム計算…'}
    snap_vals = []
    for v1,_,v2 in loops:
        seg = acc_df[(acc_df['t']/1000>=t_sec.iloc[v1])&(acc_df['t']/1000<=t_sec.iloc[v2])]
        if not seg.empty:
            norm = np.sqrt(seg['ax']**2 + seg['ay']**2 + seg['az']**2)
            snap_vals.append(norm.max())
    snap_median = float(np.median(snap_vals)) if snap_vals else None
    snap_std    = float(np.std(snap_vals))    if snap_vals else None

    # 17: レーダーチャート再作成
    progress_store[task_id] = {'progress':95, 'message':'レーダーチャート作成…'}
    loop_mean_duration = float(np.mean(loop_durations)) if loop_durations else None
    loop_std_duration  = float(np.std(loop_durations))  if loop_durations else None
    radar_b64, total_score = generate_radar_chart(
        score=score,
        loop_mean=loop_mean_duration,
        loop_std=loop_std_duration,
        stable_loop=stable_loop,
        pro_distance=float(np.mean(distances)) if distances else None,
        loop_count=n
    )

    # 完了
    progress_store[task_id] = {'progress':100,'message':'完了'}
    result = {
        'self_heatmap':       self_hm,
        'pro_heatmap':        pro_hm,
        'loop_plot':          loop_plot_b64,
        'compare_plot':       compare_plot_b64,
        'radar_chart':        radar_b64,
        'total_score':        total_score,
        'score':              score,
        'loop_count':         n,
        'stable_loop':        stable_loop,
        'loop_mean_duration': loop_mean_duration,
        'loop_std_duration':  loop_std_duration,
        'loop_duration_list': loop_duration_list,
        'loop_max_acc_list':  loop_max_acc_list,
        'snap_median':        snap_median,
        'snap_std':           snap_std,
        'pro_distance_mean':  float(np.mean(distances)) if distances else None
    }
    return jsonify(result)

# ── 既存エンドポイント（保存・履歴取得など） ─────────────────────────────────
@app.route("/save_result", methods=["POST"])
def save_result():
    result = request.get_json()
    if not result:
        return jsonify({"error":"No result data"}),400
    save_result_to_db(result)
    return jsonify({"status":"saved"})

@app.route("/results", methods=["GET"])
def get_results():
    conn=sqlite3.connect(DB_PATH); cur=conn.cursor()
    cur.execute("""
        SELECT id,timestamp,name,score,total_score,pro_distance_mean,loop_count,stable_loop
        FROM results ORDER BY id DESC LIMIT 20
    """)
    rows=cur.fetchall(); conn.close()
    return jsonify([
        {"id":r[0],"timestamp":r[1],"name":r[2] or "無題",
         "score":r[3],"total_score":r[4],"pro_distance_mean":r[5],
         "loop_count":r[6],"stable_loop":r[7]}
        for r in rows
    ])

@app.route("/results/<int:result_id>", methods=["GET"])
def get_result_detail(result_id):
    conn=sqlite3.connect(DB_PATH); cur=conn.cursor()
    cur.execute("""
        SELECT timestamp,name,score,total_score,radar_chart,pro_distance_mean,
               loop_count,stable_loop,loop_mean_duration,loop_std_duration,
               loop_plot,self_heatmap,pro_heatmap,compare_plot,
               loop_duration_list,loop_max_acc_list,snap_median,snap_std
        FROM results WHERE id=?
    """,(result_id,))
    row=cur.fetchone(); conn.close()
    if not row:
        return jsonify({"error":"Result not found"}),404
    return jsonify({
        "timestamp":row[0],"name":row[1] or "無題","score":row[2],
        "total_score":row[3],"radar_chart":row[4],"pro_distance_mean":row[5],
        "loop_count":row[6],"stable_loop":row[7],"loop_mean_duration":row[8],
        "loop_std_duration":row[9],"loop_plot":row[10],"self_heatmap":row[11],
        "pro_heatmap":row[12],"compare_plot":row[13],
        "loop_duration_list":json.loads(row[14]),
        "loop_max_acc_list":json.loads(row[15]),
        "snap_median":row[16],"snap_std":row[17]
    })

@app.route("/")
def index():
    return send_file("index.html")

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
    conn=sqlite3.connect(DB_PATH); cur=conn.cursor()
    cur.execute("DELETE FROM results WHERE id=?",(result_id,))
    conn.commit(); conn.close()
    return jsonify({"status":"deleted","id":result_id})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)

import json
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
from celery import Celery
from celery.result import AsyncResult

font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()


#plt.rcParams['font.family'] = 'Hiragino Sans'

app = Flask(__name__)
CORS(app)

# Celery setup (1ファイル内に記述)
app.config.update(
    CELERY_BROKER_URL='redis://localhost:6379/0',
    CELERY_RESULT_BACKEND='redis://localhost:6379/0'
)

celery = Celery(
    app.import_name,
    broker=app.config['broker_url'],
    backend=app.config['result_backend']
)

celery.conf.update(app.config)
def encode_heatmap(mat: np.ndarray, title: str) -> str:
    """行列 mat をヒートマップ化して Base64 文字列で返す"""
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

init_db()

# --- DBに結果を保存 ---
def save_result_to_db(result):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    jst_now = datetime.utcnow() + timedelta(hours=9)
    result["loop_duration_list"] = json.dumps(result.get("loop_duration_list", []), ensure_ascii=False)
    result["loop_max_acc_list"] = json.dumps(result.get("loop_max_acc_list", []), ensure_ascii=False)

    cur.execute("""
        INSERT INTO results (
            timestamp, name, total_score, radar_chart, score, pro_distance_mean, loop_count, stable_loop,
            loop_mean_duration, loop_std_duration,
            loop_plot, self_heatmap, heatmap, pro_heatmap, compare_plot, combined_heatmap, acc_csv, gyro_csv,snap_median, snap_std, loop_duration_list, loop_max_acc_list
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        jst_now.strftime("%Y-%m-%d %H:%M:%S"),  # ← JSTを保存
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
        result.get("acc_csv"),  # 追加　
        result.get("gyro_csv"),  # 追加
        result.get("snap_median"),
        result.get("snap_std"),
        result.get("loop_duration_list"),  # ← ここ
        result.get("loop_max_acc_list")
    ))
    conn.commit()
    conn.close()

# --- 履歴一覧をJSONで返す ---
@app.route("/results", methods=["GET"])
def get_results():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, timestamp, name, score, total_score, pro_distance_mean,
               loop_count, stable_loop
        FROM results ORDER BY id DESC LIMIT 20
    """)
    rows = cur.fetchall()
    conn.close()
    return jsonify([
        {
            "id": r[0],
            "timestamp": r[1],
            "name": r[2] or "無題",
            "score": r[3],
            "total_score": r[4],
            "pro_distance_mean": r[5],
            "loop_count": r[6],
            "stable_loop": r[7]
        }
        for r in rows
    ])


# --- 特定の結果を取得（グラフ画像含む） ---
@app.route("/results/<int:result_id>", methods=["GET"])
def get_result_detail(result_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT timestamp, name, score, total_score, radar_chart, pro_distance_mean,
               loop_count, stable_loop,
               loop_mean_duration, loop_std_duration,
               loop_plot, self_heatmap, pro_heatmap, compare_plot, loop_duration_list, loop_max_acc_list, snap_median, snap_std 
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
        "total_score": row[3],
        "radar_chart": row[4],
        "pro_distance_mean": row[5],
        "loop_count": row[6],
        "stable_loop": row[7],
        "loop_mean_duration": row[8],
        "loop_std_duration": row[9],
        "loop_plot": row[10],
        "self_heatmap": row[11],
        "pro_heatmap": row[12],
        "compare_plot": row[13],
        "loop_duration_list": row[14],     # 例: [0.639, 0.411, ...]
        "loop_max_acc_list": row[15],      # 例: [163.04, 102.58, ...]
        "snap_median": row[16],            # 例: 100.06
        "snap_std": row[17]           # 例: 25.40


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


# --- レーダーチャート生成（各項目を個別スケーリング） ---
def generate_radar_chart(score, loop_mean, loop_std, stable_loop, pro_distance):
    # 1. 類似度スコア（100点で5、0点で0）
    if score is None:
        s_score = 0
    elif score >= 100:
        s_score = 5
    elif score <= 0:
        s_score = 0
    else:
        s_score = (score / 100) * 5  # 100点で5点

    # 2. 平均ループ時間（0.4s以下で5、0.9s以上で0）
    if loop_mean is None:
        s_mean = 0
    elif loop_mean <= 0.4:
        s_mean = 5
    elif loop_mean >= 0.9:
        s_mean = 0
    else:
        s_mean = 5 * (0.9 - loop_mean) / (0.9 - 0.4)  # 線形補間

    # 3. ループ時間の標準偏差（0.05s以内で5、0.2s以上で0）
    if loop_std is None:
        s_std = 0
    elif loop_std <= 0.05:
        s_std = 5
    elif loop_std >= 0.2:
        s_std = 0
    else:
        s_std = 5 * (0.2 - loop_std) / (0.2 - 0.05)  # 線形補間

    # 4. 安定開始ループ（2周目で5、7周目以降で0）
    if stable_loop is None:
        s_stable = 0
    elif stable_loop <= 2:
        s_stable = 5
    elif stable_loop >= 7:
        s_stable = 0
    else:
        s_stable = 5 * (7 - stable_loop) / (7 - 2)

    # 5. プロ距離（平均が20以下で5、70以上で0）
    if pro_distance is None:
        s_pro = 0
    elif pro_distance <= 20:
        s_pro = 5
    elif pro_distance >= 120:
        s_pro = 0
    else:
        s_pro = 5 * (120 - pro_distance) / (120 - 20)

    # --- レーダーチャートを作成 ---
    labels = [
        '自身の類似度',
        '　平均ループ時間',
        'ループ時間のばらつき',
        '安定開始ループ',
        'プロ類似度'
    ]
    values = [s_score, s_mean, s_std, s_stable, s_pro]
    avg_score = np.mean(values) * 20 
    values += values[:1]  # 円を閉じる

    angles = np.linspace(0, 2 * np.pi, len(labels) + 1, endpoint=True)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # デフォルトのグリッド（円）を消す
    ax.grid(False)
    ax.set_frame_on(False)

    # 角度の目盛りを非表示
    ax.set_thetagrids([])

    # 半径方向の目盛り（1〜5）
    ax.set_rgrids([1, 2, 3, 4, 5], angle=0, fontproperties=font_prop, fontsize=20)

    # 円グリッドを消して、多角形（五角形）グリッドを描画
    ax.xaxis.set_visible(False)  # デフォルトの放射線ラベル非表示
    for r in range(1, 6):  # 1〜5の同心五角形
        grid = [r] * (len(labels) + 1)
        ax.plot(angles, grid, color='gray', linewidth=1.0, linestyle='-')

    # 軸線（中心から各頂点までの線）
    for angle in angles[:-1]:
        ax.plot([angle, angle], [0, 5], color='gray', linewidth=1.0)

    # ラベルを外側 (r=5.5) に配置
    for angle, label in zip(angles[:-1], labels):
        ax.text(
            angle, 6.3, label,
            ha='center', va='center',
            fontsize=25,
            fontproperties=font_prop
        )

    # プロット
    ax.plot(angles, values, color='blue', linewidth=2)
    ax.fill(angles, values, color='skyblue', alpha=0.4)
    ax.set_ylim(0, 5)
    ax.grid(False)  # デフォルトグリッドをOFF

    # PNGとして返す
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('ascii'), float(avg_score)

@celery.task(bind=True)
def analyze_task(self, acc, gyro):
    import pandas as pd, numpy as np
    from ahrs.filters import Madgwick
    from fastdtw import fastdtw
    from scipy.signal import find_peaks, savgol_filter
    import base64
    import matplotlib.pyplot as plt
    from io import BytesIO

    def encode_image(fig):
        buf = BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()

    def generate_radar_chart(score, loop_mean, loop_std, stable_loop, pro_distance):
        labels = ['類似度', '平均時間', '時間ばらつき', '安定開始', 'プロ類似度']
        values = []
        values.append(min(score / 100 * 5, 5))
        values.append(5 if loop_mean <= 0.4 else 0 if loop_mean >= 0.9 else 5 * (0.9 - loop_mean) / 0.5)
        values.append(5 if loop_std <= 0.05 else 0 if loop_std >= 0.2 else 5 * (0.2 - loop_std) / 0.15)
        values.append(5 if stable_loop <= 2 else 0 if stable_loop >= 7 else 5 * (7 - stable_loop) / 5)
        values.append(5 if pro_distance <= 20 else 0 if pro_distance >= 120 else 5 * (120 - pro_distance) / 100)
        angles = np.linspace(0, 2 * np.pi, len(labels) + 1, endpoint=True)
        values += values[:1]
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles, values, 'b-', linewidth=2)
        ax.fill(angles, values, 'skyblue', alpha=0.5)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels([])
        return encode_image(fig), float(np.mean(values) * 20)

    self.update_state(state='PROGRESS', meta={'progress': 5})
    acc_df = pd.DataFrame(acc)
    gyro_df = pd.DataFrame(gyro)
    # gz カラムがなければ 'z' カラムを、そのまま使う
    if 'gz' in gyro_df.columns:
        gyro_df['z'] = pd.to_numeric(gyro_df['gz'], errors='coerce')
    elif 'z' in gyro_df.columns:
        gyro_df['z'] = pd.to_numeric(gyro_df['z'], errors='coerce')
    else:
        raise KeyError("gyro の入力に 'gz' も 'z' もありません")
    dt = (acc_df['t'].iloc[1] - acc_df['t'].iloc[0]) / 1000.0
    fs = 1.0 / dt

    mad = Madgwick(frequency=fs, gain=0.33)
    q = [1.0, 0.0, 0.0, 0.0]
    quats = []
    for i in range(len(gyro_df)):
        q = mad.updateIMU(q=q,
                          gyr=gyro_df.loc[i, ['gx','gy','gz']],
                          acc=acc_df.loc[i, ['ax','ay','az']])
        quats.append([gyro_df['t'][i]/1000.0, *q])
    quat_df = pd.DataFrame(quats, columns=['time','w','x','y','z'])

    self.update_state(state='PROGRESS', meta={'progress': 20})

    y = savgol_filter(gyro_df['gy'], 11, 3)
    mu, sigma = y.mean(), y.std()
    peaks, _ = find_peaks(y, height=mu+sigma)
    valleys, _ = find_peaks(-y, height=abs(mu-sigma))
    t_sec = (gyro_df['t'] - gyro_df['t'].iloc[0]) / 1000.0

    loops = []
    i = 0
    while i < len(valleys) - 1:
        v1 = valleys[i]
        ps = [p for p in peaks if p > v1]
        if not ps:
            i += 1
            continue
        p = ps[0]
        vs2 = [v for v in valleys if v > p]
        if not vs2:
            i += 1
            continue
        v2 = vs2[0]
        if t_sec.iloc[v2] - t_sec.iloc[v1] <= 1:
            loops.append((v1, p, v2))
            i = valleys.tolist().index(v2)
        else:
            i += 1

    segments = []
    for v1, p, v2 in loops:
        t1, t2 = t_sec.iloc[v1], t_sec.iloc[v2]
        segments.append(quat_df[(quat_df['time'] >= t1) & (quat_df['time'] <= t2)].reset_index(drop=True))

    n = len(segments)
    dtw_mat = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            dtw_mat[a, b] = sum(
                fastdtw(segments[a][k], segments[b][k], dist=lambda x,y: abs(x-y))[0]
                for k in ['w','x','y','z']
            )

    self.update_state(state='PROGRESS', meta={'progress': 50})

    vals = dtw_mat[np.triu_indices(n, 1)]
    score = float((100 * (1.0 - ((vals - vals.min()) / (vals.max() - vals.min()))) ).mean()) if vals.size > 0 else 0

    def detect_stable(dtw):
        N = dtw.shape[0]
        if N < 2: return None
        ref_idx = list(range(N - N//2, N))
        for i in range(N - N//2):
            mean_dist = dtw[i, ref_idx].mean()
            if mean_dist <= vals.mean():
                return i + 1
        return None
    stable_loop = detect_stable(dtw_mat)

    self.update_state(state='PROGRESS', meta={'progress': 60})

    pro_segments = segments[:min(5, len(segments))]
    ref_loop = pro_segments[0] if pro_segments else None
    distances = []
    for seg in segments:
        dist = sum(fastdtw(seg[k], ref_loop[k], dist=lambda a,b:abs(a-b))[0] for k in ['w','x','y','z'])
        distances.append(dist)
    pro_mean = float(np.mean(distances)) if distances else None

    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(dtw_mat, cmap='coolwarm')
    fig.colorbar(cax)
    self_heatmap = encode_image(fig)

    pro_mat = np.full((n, n), np.nan)
    for i, d in enumerate(distances):
        pro_mat[i, i] = d
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    cax2 = ax2.matshow(pro_mat, cmap='coolwarm', vmin=20, vmax=120)
    fig2.colorbar(cax2)
    pro_heatmap = encode_image(fig2)

    loop_durations = [t_sec.iloc[v2] - t_sec.iloc[v1] for v1, _, v2 in loops]
    loop_mean = float(np.mean(loop_durations)) if loop_durations else None
    loop_std = float(np.std(loop_durations)) if loop_durations else None
    radar_chart, total_score = generate_radar_chart(score, loop_mean, loop_std, stable_loop or 7, pro_mean or 120)

    snap_values = []
    for v1, _, v2 in loops:
        seg = acc_df[(acc_df['t']/1000 >= t_sec.iloc[v1]) & (acc_df['t']/1000 <= t_sec.iloc[v2])]
        norm = np.sqrt(seg['ax']**2 + seg['ay']**2 + seg['az']**2)
        snap_values.append(norm.max())
    snap_median = float(np.median(snap_values)) if snap_values else None
    snap_std = float(np.std(snap_values)) if snap_values else None

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(t_sec, y, color='orange')
    for v1, _, v2 in loops:
        ax3.axvspan(t_sec.iloc[v1], t_sec.iloc[v2], color='red', alpha=0.3)
    loop_plot = encode_image(fig3)

    self.update_state(state='PROGRESS', meta={'progress': 95})

    return {
        'score': round(score, 1),
        'total_score': round(total_score, 1),
        'stable_loop': stable_loop,
        'loop_count': n,
        'loop_mean_duration': loop_mean,
        'loop_std_duration': loop_std,
        'pro_distance_mean': pro_mean,
        'snap_median': snap_median,
        'snap_std': snap_std,
        'self_heatmap': self_heatmap,
        'pro_heatmap': pro_heatmap,
        'radar_chart': radar_chart,
        'loop_plot': loop_plot,
        'message': '解析完了'
    }



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


    # --- オリジナルの自己比較行列をバックアップ ---
    orig_self_mat = dtw_mat.copy()




    


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


    # --- 純粋な自己比較ヒートマップ ---
    self_heatmap_b64 = encode_heatmap(orig_self_mat, 'Self Loop Similarity (Original)')

    # # --- プロ距離を対角に埋め込んだ行列（まとめて正規化用） ---
    combined_for_heatmap = orig_self_mat.copy()
    for i, d in enumerate(distances):
        combined_for_heatmap[i, i] = d
    # heatmap_b64 = encode_heatmap(combined_for_heatmap, 'Loop Similarity\n(Self Off‑Diag + Pro Diag)')



    
    

    

    # --- プロ対角のみ行列 & ヒートマップ ---
    pro_mat = np.full((n, n), np.nan)
    for i, d in enumerate(distances):
        pro_mat[i, i] = d

    # 色のスケール: 0が青、最大が赤になるように固定
    vmin, vmax = 20, 120

    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(pro_mat, cmap='coolwarm', vmin=vmin, vmax=vmax)
    plt.colorbar(cax)
    ax.set_title('Pro vs Each Loop (Diagonal Only)')
    ticks = list(range(n))
    labels = [str(i+1) for i in ticks]
    ax.set_xticks(ticks); ax.set_xticklabels(labels)
    ax.set_yticks(ticks); ax.set_yticklabels(labels)
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    pro_heatmap_b64 = base64.b64encode(buf.getvalue()).decode('ascii')


   # --- combined_heatmap: Self の非対角 + Pro の対角 ---
    # 元の色スケールのまま別々に重ねる

    # 1) 自己比較行列のコピーを作り、対角線は NaN にして off_diag とする
    off_diag = orig_self_mat.copy()
    np.fill_diagonal(off_diag, np.nan)

    # プロ距離だけを格納する diag_mat（非対角は NaN）
    diag_mat = np.full_like(orig_self_mat, np.nan, dtype=float)
    for i, d in enumerate(distances):
        diag_mat[i, i] = d

    # ---- vmin/vmax 設定 ----
    # 自己比較スケール
    if np.all(np.isnan(off_diag)):  # n<=1 の場合
        vmin_self, vmax_self = 0, 1
    else:
        vmin_self, vmax_self = np.nanmin(off_diag), np.nanmax(off_diag)

    # プロ比較スケール（pro_heatmap_b64と同じ固定範囲）
    vmin_pro, vmax_pro = 20, 120

    # 3) 描画
    # 図の比率を正方形寄りにしてフォントサイズを調整
    fig, ax = plt.subplots(figsize=(7, 7), dpi=70)

    # まず off-diagonal（自己比較距離）を描画
    cax1 = ax.matshow(off_diag, cmap='coolwarm',
                    vmin=vmin_self, vmax=vmax_self)

    # 次に diagonal（プロ距離）を上に重ねる
    cax2 = ax.matshow(diag_mat, cmap='coolwarm',
                    vmin=vmin_pro, vmax=vmax_pro)

    # タイトルと軸設定
    ax.set_title('Loop Similarity\n(Self Off-Diag in Self-Scale + Pro Diag in Pro-Scale)')
    ticks = list(range(n))
    labels = [str(i+1) for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    plt.tight_layout()

 
    # カラーバー用の専用軸を追加して、座標で配置（左: x0, 下: y0, 幅, 高さ）
    cbar_ax1 = fig.add_axes([0.03, 0.15, 0.02, 0.7])  # 1本目（固定）
    cbar_ax2 = fig.add_axes([0.90, 0.15, 0.02, 0.7])  # 2本目（好きな位置）

    # それぞれカラーバーを作成
    cbar1 = fig.colorbar(cax1, cax=cbar_ax1)
    cbar1.set_label("Self Range")
    cbar1.ax.yaxis.set_label_position('left') 

    cbar2 = fig.colorbar(cax2, cax=cbar_ax2)
    cbar2.set_label("Pro Range")


    # 余白調整（タイトルと軸が重ならないように）
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.15, right=0.85)

    # 5) PNG → Base64 変換
    buf = BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    # combined_heatmap_b64 = base64.b64encode(buf.getvalue()).decode('ascii')



    














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
        loop_duration_list = []
        for i, (v1, _, v2) in enumerate(loops):
            t_start = t_sec.iloc[v1]
            t_end = t_sec.iloc[v2]
            duration = float(t_end - t_start)

            # 加速度セグメントの抽出とノルム最大値
            acc_segment = acc_df[(acc_df['t']/1000 >= t_start) & (acc_df['t']/1000 <= t_end)]
            if not acc_segment.empty:
                norm = np.sqrt(acc_segment['ax']**2 + acc_segment['ay']**2 + acc_segment['az']**2)
                max_norm = norm.max()
                loop_duration_list.append(f"ループ {i+1}: {duration:.3f} 秒 / {max_norm:.2f} m/s²")
            else:
                loop_duration_list.append(f"ループ {i+1}: {duration:.3f} 秒 / - m/s²")

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

    pro_distance_mean = float(np.mean(distances)) if distances else None


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

    

    # プロ距離の平均を事前に計算
    if distances:
        pro_distance_mean = float(np.mean(distances))
    else:
        pro_distance_mean = None  # セグメントが検出されなかった場合


    # レーダーチャート生成
    radar_b64, total_score = generate_radar_chart(
        score=score,
        loop_mean=loop_mean_duration,
        loop_std=loop_std_duration,
        stable_loop=stable_loop if stable_loop else 7,
        pro_distance=pro_distance_mean
    )

    # 加速度ノルムの最大値を各ループで計算
    loop_max_acc_list = []
    for i, (v1, p, v2) in enumerate(loops):
        t_start = t_sec.iloc[v1]
        t_end = t_sec.iloc[v2]
        acc_segment = acc_df[(acc_df['t']/1000 >= t_start) & (acc_df['t']/1000 <= t_end)]
        if not acc_segment.empty:
            norm = np.sqrt(acc_segment['ax']**2 + acc_segment['ay']**2 + acc_segment['az']**2)
            max_norm = norm.max()
            loop_max_acc_list.append(f"ループ {i+1}: {max_norm:.3f}  m/s²")


    # スナップ中央値・標準偏差の計算
    snap_values = []
    for i, (v1, _, v2) in enumerate(loops):
        t_start = t_sec.iloc[v1]
        t_end = t_sec.iloc[v2]
        acc_segment = acc_df[(acc_df['t']/1000 >= t_start) & (acc_df['t']/1000 <= t_end)]
        if not acc_segment.empty:
            norm = np.sqrt(acc_segment['ax']**2 + acc_segment['ay']**2 + acc_segment['az']**2)
            snap_values.append(norm.max())

    snap_median = float(np.median(snap_values)) if snap_values else None
    snap_std = float(np.std(snap_values)) if snap_values else None




    # --- 結果まとめ ---
    result = {
        'score': score if len(loops) >= 2 else 0.0,
        'total_score': total_score, 
        'self_heatmap': self_heatmap_b64,  # ← 純粋な自分同士比較
        # 'heatmap': heatmap_b64,
        'pro_heatmap': pro_heatmap_b64,  # プロ距離だけのヒートマップ（新規）
        # 'combined_heatmap': combined_heatmap_b64,  # ← 追加
        'loop_plot': loop_plot_b64,
        'stable_loop': stable_loop if len(loops) >= 2 else None,
        'loop_count': n,
        'loop_mean_duration': loop_mean_duration if len(loops) >= 2 else None,
        'loop_std_duration': loop_std_duration if len(loops) >= 2 else None,
        'loop_duration_list': loop_duration_list if len(loops) >= 2 else [],
        'compare_plot': compare_plot_b64,
        'radar_chart': radar_b64,
        'pro_distance_mean': pro_distance_mean,
        'loop_max_acc_list': loop_max_acc_list,
        'snap_median': snap_median,
        'snap_std': snap_std

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

# ==== タスク開始エンドポイント ====
@app.route('/start_analysis', methods=['POST'])
def start_analysis():
    data = request.get_json()
    print("=== start_analysis received ===")
    print(data)         # ← ここを追加
    print("acc len:", len(data.get('acc', [])))
    print("gyro len:", len(data.get('gyro', [])))
    acc = data.get('acc', [])
    gyro = data.get('gyro', [])
    task = analyze_task.apply_async(args=[acc, gyro])
    return jsonify({'task_id': task.id})

# ==== 進捗取得エンドポイント ====
@app.route('/progress/<task_id>', methods=['GET'])
def get_progress(task_id):
    task = AsyncResult(task_id, app=celery)
    if task.state == 'PENDING':
        return jsonify({'state': task.state, 'progress': 0})
    elif task.state == 'PROGRESS':
        return jsonify({'state': task.state, 'progress': task.info.get('progress', 0)})
    elif task.state == 'SUCCESS':
        return jsonify({
            'state': task.state,
            'progress': 100,
            'result': task.result  # ←結果付き
        })
    else:
        return jsonify({'state': task.state, 'progress': 0})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

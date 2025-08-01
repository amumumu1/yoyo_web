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

# ── ユーティリティ関数 ─────────────────────────────────
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

def detect_stable_loop_by_tail(dtw_matrix: np.ndarray) -> int:
    N = dtw_matrix.shape[0]
    if N < 2:
        return None
    vals = dtw_matrix[np.triu_indices(N, k=1)]
    if vals.size == 0:
        return None
    d_min, d_max = vals.min(), vals.max()
    threshold = (d_min + d_max) / 2
    tail_len = N // 2
    ref_idx = list(range(N - tail_len, N))
    for i in range(N - tail_len):
        mean_dist = dtw_matrix[i, ref_idx].mean()
        if mean_dist <= threshold:
            return i + 1
    return None

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
        a   = [acc.at[i,'x'], acc.at[i,'y'], acc.at[i,'z']]
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

def generate_radar_chart(score, loop_mean, loop_std, stable_loop, pro_distance):
    # スケール化（省略）
    # …旧コードと同じ…
    labels = ['自身の類似度','平均ループ時間','ループ時間のばらつき','安定開始ループ','プロ類似度']
    # …同じ…
    fig, ax = plt.subplots(figsize=(10,10),subplot_kw=dict(polar=True))
    # …同じ…
    buf=BytesIO(); fig.savefig(buf,format='png',bbox_inches='tight'); plt.close(fig)
    avg_score = float(np.mean([s_score, s_mean, s_std, s_stable, s_pro]) * 20)
    return base64.b64encode(buf.getvalue()).decode('ascii'), avg_score

# ── 解析エンドポイント ─────────────────────────────────
@app.route('/analyze', methods=['POST'])
def analyze():
    task_id = request.args.get('task_id')
    if not task_id or task_id not in progress_store:
        return jsonify({'error':'Invalid task_id'}),400

    progress_store[task_id]={'progress':0,'message':'解析開始…'}

    # 1. プロデータ読み込み
    progress_store[task_id]={'progress':5,'message':'プロデータ読込中…'}
    pro_acc="3_acc2.csv"; pro_gyro="3_gyro2.csv"
    gyro_pro, quats_pro = load_and_compute_quaternions(pro_acc,pro_gyro)

    # 2. プロループ抽出
    progress_store[task_id]={'progress':10,'message':'プロループ抽出中…'}
    pro_segments = segment_loops(gyro_pro,quats_pro)

    # 3. JSON受信
    progress_store[task_id]={'progress':15,'message':'入力データ受信…'}
    payload = request.get_json(force=True)

    # 4. データ前処理
    progress_store[task_id]={'progress':20,'message':'データ前処理…'}
    acc_df=pd.DataFrame(payload['acc'])
    gyro_df=pd.DataFrame(payload['gyro'])
    gyro_df['z']=pd.to_numeric(gyro_df['gz'],errors='coerce')
    dt=(acc_df['t'].iloc[1]-acc_df['t'].iloc[0])/1000.0

    # 5. ユーザークォータニオン計算
    progress_store[task_id]={'progress':25,'message':'姿勢計算中…'}
    mad=Madgwick(frequency=1.0/dt,gain=0.33)
    q=[1,0,0,0]; quats=[]
    for i in range(len(gyro_df)):
        gyr=gyro_df.loc[i,['gx','gy','z']].tolist()
        acc=acc_df.loc[i,['ax','ay','az']].tolist()
        q=mad.updateIMU(q=q,gyr=gyr,acc=acc)
        quats.append([gyro_df['t'][i]/1000,*q])
    quat_df=pd.DataFrame(quats,columns=['time','w','x','y','z'])

    # 6. フィルタ＆ピーク検出
    progress_store[task_id]={'progress':30,'message':'ピーク検出中…'}
    y=savgol_filter(gyro_df['gy'],window_length=11,polyorder=3)
    peaks,_=find_peaks(y,height=y.mean()+y.std())
    valleys,_=find_peaks(-y,height=y.std()-y.mean())

    # 7. ループ検出
    progress_store[task_id]={'progress':40,'message':'ループ検出中…'}
    loops=[]; t_sec=(gyro_df['t']-gyro_df['t'].iloc[0])/1000.0; i=0
    while i<len(valleys)-1:
        v1=valleys[i]
        ps=[p for p in peaks if p>v1 and y[p]>y.mean()+y.std()]
        if ps:
            p=ps[0]; vs2=[v for v in valleys if v>p and y[v]<y.mean()-y.std()]
            if vs2 and (t_sec.iloc[vs2[0]]-t_sec.iloc[v1])<=1.0:
                loops.append((v1,p,vs2[0])); i=valleys.tolist().index(vs2[0]); continue
        i+=1

    # 8. 自己比較行列計算
    progress_store[task_id]={'progress':50,'message':'自己比較行列計算…'}
    n=len(loops); dtw_mat=np.zeros((n,n)); segments=[]
    for v1,_,v2 in loops:
        mask=(quat_df['time']>=t_sec.iloc[v1])&(quat_df['time']<=t_sec.iloc[v2])
        segments.append(quat_df[mask].reset_index(drop=True))
    for a in range(n):
        for b in range(n):
            dtw_mat[a,b]=sum(fastdtw(segments[a][k],segments[b][k],dist=lambda x,y:abs(x-y))[0]
                             for k in ['w','x','y','z'])

    # 9. プロ比較距離
    progress_store[task_id]={'progress':60,'message':'プロ比較距離計算…'}
    distances=[]
    try:
        M=len(pro_segments)
        pro_dtw=np.zeros((M,M))
        for i in range(M):
            for j in range(M):
                pro_dtw[i,j]=sum(fastdtw(pro_segments[i][k],pro_segments[j][k],dist=lambda a,b:abs(a-b))[0]
                                 for k in ['w','x','y','z'])
        valid_idx=np.arange(M)[1:-1]
        ref_idx=valid_idx[np.argmin(pro_dtw.sum(axis=1)[1:-1])]
        ref_loop=pro_segments[ref_idx]
        distances=[sum(fastdtw(seg[k],ref_loop[k],dist=lambda a,b:abs(a-b))[0]
                       for k in ['w','x','y','z']) for seg in segments]
    except:
        distances=[0]*n

    # 10. 各種統計をここでまとめて計算
    stable_loop         = detect_stable_loop_by_tail(dtw_mat)
    loop_durations      = [ float(t_sec[v2] - t_sec[v1]) for v1,_,v2 in loops ]
    loop_mean_duration  = float(np.mean(loop_durations)) if loop_durations else None
    loop_std_duration   = float(np.std(loop_durations))  if loop_durations else None

    loop_duration_list  = []
    loop_max_acc_list   = []
    snap_values         = []
    for idx,(v1,p,v2) in enumerate(loops):
        t0, t1 = t_sec[v1], t_sec[v2]
        duration = t1 - t0
        acc_seg = acc_df[(acc_df['t']/1000>=t0)&(acc_df['t']/1000<=t1)]
        if not acc_seg.empty:
            norm = np.sqrt(acc_seg['ax']**2 + acc_seg['ay']**2 + acc_seg['az']**2)
            max_norm = float(norm.max())
        else:
            max_norm = None
        loop_duration_list.append(f"ループ {idx+1}: {duration:.3f} 秒 / {max_norm or 0:.2f} m/s²")
        loop_max_acc_list.append(f"ループ {idx+1}: {max_norm or 0:.3f} m/s²")
        if max_norm is not None:
            snap_values.append(max_norm)

    snap_median = float(np.median(snap_values)) if snap_values else None
    snap_std    = float(np.std(snap_values))    if snap_values else None

    # 11. Selfヒートマップ作成
    progress_store[task_id]={'progress':65,'message':'自身ヒートマップ作成…'}
    self_hm = encode_heatmap(dtw_mat, 'Self Loop Similarity')

    # 12. Proヒートマップ作成
    progress_store[task_id]={'progress':70,'message':'プロヒートマップ作成…'}
    pro_mat = np.full_like(dtw_mat, np.nan)
    for i,d in enumerate(distances): pro_mat[i,i] = d
    pro_hm = encode_heatmap(pro_mat, 'Pro vs Each Loop')

    # 13. ループ検出グラフ作成
    progress_store[task_id]={'progress':75,'message':'ループ検出グラフ作成…'}
    fig2,ax2=plt.subplots(figsize=(12,6))
    ax2.plot(t_sec,y,color='orange')
    for idx,(v1,p,v2) in enumerate(loops):
        ax2.axvspan(t_sec[v1],t_sec[v2],color='red',alpha=0.3,label='1周' if idx==0 else "")
    ax2.plot(t_sec[peaks],y[peaks],"go",label="ピーク")
    ax2.plot(t_sec[valleys],y[valleys],"ro",label="谷")
    ax2.set_title("ループ検出",fontproperties=font_prop)
    ax2.set_xlabel("時間 [秒]",fontproperties=font_prop)
    ax2.set_ylabel("角速度 gy [rad/s]",fontproperties=font_prop)
    ax2.legend(prop=font_prop); ax2.grid(True)
    buf2=BytesIO(); fig2.savefig(buf2,format='png'); plt.close(fig2)
    loop_plot_b64 = base64.b64encode(buf2.getvalue()).decode('ascii')

    # 14. プロ比較バーグラフ作成
    progress_store[task_id]={'progress':80,'message':'バーグラフ作成…'}
    fig3,ax3=plt.subplots(figsize=(8,4))
    idxs=list(range(1,n+1))
    ax3.bar(idxs, distances, edgecolor='black')
    ax3.set_title("プロと各ループの距離比較", fontproperties=font_prop)
    ax3.set_xlabel("あなたのループ番号", fontproperties=font_prop)
    ax3.set_ylabel("DTW距離", fontproperties=font_prop)
    ax3.set_xticks(idxs); ax3.grid(True)
    buf3=BytesIO(); fig3.savefig(buf3,format='png'); plt.close(fig3)
    compare_plot_b64 = base64.b64encode(buf3.getvalue()).decode('ascii')

    # 15. レーダーチャート作成
    progress_store[task_id]={'progress':90,'message':'レーダーチャート作成…'}
    radar_b64, total_score = generate_radar_chart(
        score=float((100*(1.0-(1.0-np.triu(dtw_mat,k=1).mean()))) if n>1 else 0.0),
        loop_mean=loop_mean_duration,
        loop_std=loop_std_duration,
        stable_loop=stable_loop,
        pro_distance=float(np.mean(distances)) if distances else None
    )

    # 16. 完了
    progress_store[task_id]={'progress':100,'message':'完了'}
    result = {
        'self_heatmap'       : self_hm,
        'pro_heatmap'        : pro_hm,
        'loop_plot'          : loop_plot_b64,
        'compare_plot'       : compare_plot_b64,
        'radar_chart'        : radar_b64,
        'total_score'        : total_score,
        'score'              : float((100*(1.0-(1.0-np.triu(dtw_mat,k=1).mean()))) if n>1 else 0.0),
        'loop_count'         : n,
        'pro_distance_mean'  : float(np.mean(distances)) if distances else None,
        # ここから不足分
        'stable_loop'        : stable_loop,
        'loop_mean_duration' : loop_mean_duration,
        'loop_std_duration'  : loop_std_duration,
        'loop_duration_list' : loop_duration_list,
        'loop_max_acc_list'  : loop_max_acc_list,
        'snap_median'        : snap_median,
        'snap_std'           : snap_std
    }
    return jsonify(result)

# ── 既存エンドポイント ─────────────────────────────────
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
    conn=sqlite3.connect(DB_PATH); cur=conn.cursor()
    cur.execute("SELECT acc_csv,gyro_csv,timestamp,name FROM results WHERE id=?",(result_id,))
    row=cur.fetchone(); conn.close()
    if not row: return jsonify({"error":"Result not found"}),404
    acc_csv,gyro_csv,timestamp,name=row
    jst_dt=datetime.strptime(timestamp,"%Y-%m-%d %H:%M:%S")
    import io,zipfile
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as zf:
        for label,content in [("acc",acc_csv or ""),("gyro",gyro_csv or "")]:
            zi=zipfile.ZipInfo(f"{name or 'result'}_{label}.csv")
            zi.date_time=(jst_dt.year,jst_dt.month,jst_dt.day,jst_dt.hour,jst_dt.minute,jst_dt.second)
            zf.writestr(zi,content)
    buf.seek(0)
    quoted_name = name or "result"
    return Response(buf,mimetype="application/zip",
                    headers={"Content-Disposition":f"attachment; filename*=UTF-8''{quoted_name}_csv.zip"})

@app.route("/results/<int:result_id>", methods=["DELETE"])
def delete_result(result_id):
    conn=sqlite3.connect(DB_PATH); cur=conn.cursor()
    cur.execute("DELETE FROM results WHERE id=?",(result_id,))
    conn.commit(); conn.close()
    return jsonify({"status":"deleted","id":result_id})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)

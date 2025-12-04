# ============================================
# comment_patterns.py
# ヨーヨー技能評価システム：コメント辞書一覧
# ============================================

# --------------------------------------------
# ① タイプ分類（31種類）
# --------------------------------------------
types_dict = {
    # --- A：総合 ---
    "overall_good": "タイプ：総合上手型",

    "mid_type": "タイプ：中間バランス型",

    # --- B：単指標 上手い ---
    "self_sim_good": "タイプ：自己類似度上手型",
    "pro_sim_good": "タイプ：プロ類似度上手型",
    "stable_start_good": "タイプ：安定開始上手型",
    "loop_var_good": "タイプ：ループばらつき少型",
    "snap_var_good": "タイプ：スナップ安定型",

    # --- C：単指標 下手 ---
    "self_sim_bad": "タイプ：自己類似度不安定型",
    "pro_sim_bad": "タイプ：プロ類似度低型",
    "stable_start_bad": "タイプ：安定開始遅型",
    "loop_var_bad": "タイプ：ループばらつき多型",
    "snap_var_bad": "タイプ：スナップばらつき型",

    # --- D：強み × 弱み（20種類） ---
    # 自己類似度 上手 × 他4つ 下手
    "self_good_pro_bad": "タイプ：自己類似度上手 × プロ類似度低型",
    "self_good_stable_bad": "タイプ：自己類似度上手 × 安定開始遅型",
    "self_good_loopvar_bad": "タイプ：自己類似度上手 × ループばらつき多型",
    "self_good_snapvar_bad": "タイプ：自己類似度上手 × スナップばらつき型",

    # プロ類似度 上手 × 他4つ 下手
    "pro_good_self_bad": "タイプ：プロ類似度上手 × 自己類似度不安定型",
    "pro_good_stable_bad": "タイプ：プロ類似度上手 × 安定開始遅型",
    "pro_good_loopvar_bad": "タイプ：プロ類似度上手 × ループばらつき多型",
    "pro_good_snapvar_bad": "タイプ：プロ類似度上手 × スナップばらつき型",

    # 安定開始 上手 × 他4つ 下手
    "stable_good_self_bad": "タイプ：安定開始上手 × 自己類似度不安定型",
    "stable_good_pro_bad": "タイプ：安定開始上手 × プロ類似度低型",
    "stable_good_loopvar_bad": "タイプ：安定開始上手 × ループばらつき多型",
    "stable_good_snapvar_bad": "タイプ：安定開始上手 × スナップばらつき型",

    # ループばらつき少 × 他4つ 下手
    "loopvar_good_self_bad": "タイプ：ループばらつき少型 × 自己類似度不安定型",
    "loopvar_good_pro_bad": "タイプ：ループばらつき少型 × プロ類似度低型",
    "loopvar_good_stable_bad": "タイプ：ループばらつき少型 × 安定開始遅型",
    "loopvar_good_snapvar_bad": "タイプ：ループばらつき少型 × スナップばらつき型",

    # スナップ安定 × 他4つ 下手
    "snapvar_good_self_bad": "タイプ：スナップ安定型 × 自己類似度不安定型",
    "snapvar_good_pro_bad": "タイプ：スナップ安定型 × プロ類似度低型",
    "snapvar_good_stable_bad": "タイプ：スナップ安定型 × 安定開始遅型",
    "snapvar_good_loopvar_bad": "タイプ：スナップ安定型 × ループばらつき多型"
}

# --------------------------------------------
# ② 総評（3種類）
# --------------------------------------------
summary_dict = {
    "good": "全体の指標が安定しており、非常に良いパフォーマンスが実現できています。",
    "mid": "良い部分が多い一方で、改善するとさらに伸びるポイントもあります。",
    "bad": "全体的に課題が目立つため、明確な改善ポイントに取り組むと一気に良くなります。"
}

# --------------------------------------------
# ③ 強みコメント（5種類）
# --------------------------------------------
strength_dict = {
    "self_sim": "ループの再現性が高く、軌道が毎回安定しています。",
    "pro_sim": "プロの軌道に近い動きができており、フォーム精度が高いです。",
    "stable_start": "立ち上がりが速く、序盤からスムーズに安定状態に入れています。",
    "loop_var": "テンポのゆらぎが少なく、一定のリズムを維持できています。",
    "snap_var": "スナップの強弱が安定しており、毎回同じ力で返せています。"
}

# --------------------------------------------
# ④ 弱みコメント（5種類）
# --------------------------------------------
weakness_dict = {
    "self_sim": "ループごとの軌道が揺れやすく、再現性が十分ではありません。",
    "pro_sim": "プロの軌道から離れたフォームになりやすく、改善の余地があります。",
    "stable_start": "安定状態に入るまでに時間がかかり、序盤の乱れが目立ちます。",
    "loop_var": "テンポが一定にならず、ループのリズムが揺れやすい状態です。",
    "snap_var": "スナップの力が毎回変わりやすく、軌道に無駄な揺れが出ています。"
}

# --------------------------------------------
# ⑤ 改善案（5種類）
# --------------------------------------------
improvement_dict = {
    "self_sim": "同じ振り幅と角度を意識して、数ループだけ正確に繰り返す練習を行いましょう。",
    "pro_sim": "プロのスナップタイミングと腕の角度を意識して模倣練習を行いましょう。",
    "stable_start": "最初の3ループを丁寧にゆっくり振ることで立ち上がりが安定しやすくなります。",
    "loop_var": "メトロノームに合わせて一定のテンポを保つ練習をしてみましょう。",
    "snap_var": "返す瞬間だけ力を一定にするスナップ練習を繰り返すと安定します。"
}

# comment_patterns.py

def classify_type(scores):
    # internal → dictionary key prefix
    prefix_map = {
        "self_sim": "self",
        "pro_sim": "pro",
        "stable_start": "stable",
        "loop_var": "loopvar",
        "snap_var": "snapvar"
    }

    good_keys = [k for k,v in scores.items() if v >= 4]
    bad_keys  = [k for k,v in scores.items() if v <= 2]

    # ① 全部good
    if len(good_keys) == 5:
        return "overall_good"

    # ② 全部bad（任意）
    if len(bad_keys) == 5:
        return "overall_bad"

    # ③ good × bad
    if good_keys and bad_keys:
        best_good = max(good_keys, key=lambda k: scores[k])
        worst_bad = min(bad_keys, key=lambda k: scores[k])
        return f"{prefix_map[best_good]}_good_{prefix_map[worst_bad]}_bad"

    # ④ 単指標 good
    if good_keys:
        best_good = max(good_keys, key=lambda k: scores[k])
        return f"{prefix_map[best_good]}_good"

    # ⑤ 単指標 bad
    if bad_keys:
        worst_bad = min(bad_keys, key=lambda k: scores[k])
        return f"{prefix_map[worst_bad]}_bad"

    return "mid_type"


def generate_comments(scores):
    # --- 強み（最高スコア） ---
    strength_key = max(scores, key=lambda k: scores[k])

    # --- 弱み（最低スコア） ---
    weakness_key = min(scores, key=lambda k: scores[k])

    # --- 改善案 ---
    improvement = improvement_dict[weakness_key]

    return {
        "strength": strength_dict[strength_key],
        "weakness": weakness_dict[weakness_key],
        "improvement": improvement
    }

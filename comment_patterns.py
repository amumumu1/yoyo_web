# ============================================
# comment_patterns.py
# ヨーヨー技能評価システム：コメント辞書一覧
# ============================================

# --------------------------------------------
# ① タイプ分類（31種類）
# --------------------------------------------
types_dict = {
    # --- A：総合 ---
    "overall_good": "インサイド・ループ マスター",
    "overall_bad": "これから爆伸びビギナー", 
    "mid_type": "バランス型プレイヤー",

    # --- B：単指標 上手い ---
    "self_good": "ループそっくり プレイヤー",
    "pro_good": "プロそっくり プレイヤー",
    "stable_good": "投げ出し抜群 プレイヤー",
    "loopvar_good": "ループのテンポ一定 プレイヤー",
    "snapvar_good": "スナップの強さ一定 プレイヤー",

    # --- C：単指標 下手 ---
    "self_bad": "ループ類似度に難あり プレイヤー",
    "pro_bad": "個性派プレイヤー",
    "stable_bad": "投げ出し不安定 プレイヤー",
    "loopvar_bad": "ループのテンポ乱れがち プレイヤー",
    "snapvar_bad": "スナップの強さばらばら プレイヤー",

    # --- D：強み × 弱み（20種類） ---
    # 自己類似度 上手 × 他4つ 下手
    "self_good_pro_bad": "ループそっくり × 個性派 プレイヤー",
    "self_good_stable_bad": "ループそっくり × 投げ出し不安定 プレイヤー",
    "self_good_loopvar_bad": "ループそっくり × テンポ乱れがち プレイヤー",
    "self_good_snapvar_bad": "ループそっくり × スナップばらばら プレイヤー",


    # プロ類似度 上手 × 他4つ 下手
    "pro_good_self_bad": "プロそっくり × 類似度に難あり プレイヤー",
    "pro_good_stable_bad": "プロそっくり × 投げ出し不安定 プレイヤー",
    "pro_good_loopvar_bad": "プロそっくり × テンポ乱れがち プレイヤー",
    "pro_good_snapvar_bad": "プロそっくり × スナップばらばら プレイヤー",


    # 安定開始 上手 × 他4つ 下手
    "stable_good_self_bad": "投げ出し抜群 × 類似度に難あり プレイヤー",
    "stable_good_pro_bad": "投げ出し抜群 × 個性派 プレイヤー",
    "stable_good_loopvar_bad": "投げ出し抜群 × テンポ乱れがち プレイヤー",
    "stable_good_snapvar_bad": "投げ出し抜群 × スナップばらばら プレイヤー",


    # ループばらつき少 × 他4つ 下手
    "loopvar_good_self_bad": "テンポ一定 × 類似度に難あり プレイヤー",
    "loopvar_good_pro_bad": "テンポ一定 × 個性派 プレイヤー",
    "loopvar_good_stable_bad": "テンポ一定 × 投げ出し不安定 プレイヤー",
    "loopvar_good_snapvar_bad": "テンポ一定 × スナップばらばら プレイヤー",


    # スナップ安定 × 他4つ 下手
    "snapvar_good_self_bad": "スナップ一定 × 類似度に難あり プレイヤー",
    "snapvar_good_pro_bad": "スナップ一定 × 個性派 プレイヤー",
    "snapvar_good_stable_bad": "スナップ一定 × 投げ出し不安定 プレイヤー",
    "snapvar_good_loopvar_bad": "スナップ一定 × テンポ乱れがち プレイヤー",
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
    "self_sim": 
        "各ループで振り幅と腕の角度が毎回同じになるよう、体全体がぶれないことを意識して練習を行いましょう。"
        "肩や肘の位置を固定し、手首の動きだけでループを行うことで、動作全体のズレを抑えることができます。"
        "連続して長く回すのではなく、短い回数を区切って繰り返し、同じ動作を再現できているかを確認してください。",
    "pro_sim": 
        "プロの動画を参考に、スナップのタイミングと腕の振り角度に注目して模倣練習を行いましょう。"
        "特に、ヨーヨーを返す瞬間の手首の使い方と力の入り方を意識し、自身との差を確認しながら調整してください。",

    "stable_start": 
        "開始直後の動作が不安定な場合は、ループ数を段階的に増やす練習を行いましょう。"
        "まずは1周のループを安定して行いキャッチできることを確認し、安定したら2周、3周と徐々に回数を増やしていきます。"
        "立ち上がりで無理に連続ループを行わず、安定した状態を作ってから回数を増やすことを優先してください。",

    "loop_var": 
        "各ループの間隔が揃うよう、投げ出しから返しまでの一連の動作を毎回同じリズムで行う練習を行いましょう。"
        "速さよりも動作の一定性を重視し、ループごとの差が小さくなることを目標としてください。",

    "snap_var": 
        "スナップのばらつきが大きい場合は、スナップ動作の瞬間だけに意識を集中させた練習が効果的です。"
        "力の大きさやタイミングを揃えることを意識し、同じ感覚で繰り返せているかを確認しながら行いましょう。"
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

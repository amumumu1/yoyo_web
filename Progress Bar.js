// progressBar.js（フロントエンド側）

// 進捗バー表示用HTML例：
// <div id="loading" style="display: none;">
//   <p>解析中...</p>
//   <progress id="progressBar" value="0" max="100" style="width: 400px; height: 30px;"></progress>
// </div>

async function startRealAnalysis(accData, gyroData) {
  const loading = document.getElementById("loading");
  const bar = document.getElementById("progressBar");
  loading.style.display = "block";
  bar.value = 0;

  try {
    const res = await fetch("/start_analysis", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ acc: accData, gyro: gyroData })
    });

    const { task_id } = await res.json();
    if (!task_id) throw new Error("タスクID取得に失敗しました");

    pollProgress(task_id);
  } catch (e) {
    alert("解析開始エラー: " + e.message);
    loading.style.display = "none";
  }
}

function pollProgress(taskId) {
  const bar = document.getElementById("progressBar");
  const loading = document.getElementById("loading");

  const interval = setInterval(async () => {
    try {
      const res = await fetch(`/progress/${taskId}`);
      const data = await res.json();

      bar.value = data.progress;

      if (data.done || data.progress >= 100) {
        clearInterval(interval);
        loading.style.display = "none";
        renderResult(data.result);  // ← あなたの結果描画関数に置き換えてください
      }
    } catch (err) {
      clearInterval(interval);
      loading.style.display = "none";
      alert("進捗取得エラー: " + err.message);
    }
  }, 500); // 0.5秒ごとに確認
}

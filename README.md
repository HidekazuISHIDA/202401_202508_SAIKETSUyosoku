# 🏥 A病院 待ち人数・待ち時間 統合予測（Streamlit Cloud）

## この版のポイント
- `xgboost.Booster()` でモデルをロードし、`DMatrix` で推論します  
  → **scikit-learn不要**（`ImportError` を回避）
- 祝日判定は `data/syukujitsu.csv` を使用（`jpholiday` 不要）

## 必須ファイル（models/）
- model_A_timeseries.json
- columns_A_timeseries.json
- model_A_waittime_30min_FULL.json
- model_A_queue_30min_FULL.json
- columns_A_multi_30min_FULL.json

## デプロイ
Streamlit Cloud で main file を `app.py` にして Deploy。
反映されない場合は **Clear cache → Reboot**。

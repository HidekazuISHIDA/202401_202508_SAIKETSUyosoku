import json
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
import numpy as np
import xgboost as xgb
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="A病院 待ち人数・待ち時間 予測",
    page_icon="🏥",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

COUNT_MODEL_PATH = MODELS_DIR / "model_A_timeseries.json"
COUNT_COLUMNS_PATH = MODELS_DIR / "columns_A_timeseries.json"

WAITTIME_MODEL_PATH = MODELS_DIR / "model_A_waittime_30min_FULL.json"
QUEUE_MODEL_PATH = MODELS_DIR / "model_A_queue_30min_FULL.json"
MULTI_COLUMNS_PATH = MODELS_DIR / "columns_A_multi_30min_FULL.json"

HOLIDAY_CSV_PATH = DATA_DIR / "syukujitsu.csv"


@st.cache_resource(show_spinner=False)
def load_models_and_columns():
    # Booster-based loading (NO scikit-learn dependency)
    count_booster = xgb.Booster()
    count_booster.load_model(str(COUNT_MODEL_PATH))
    with open(COUNT_COLUMNS_PATH, "r", encoding="utf-8") as f:
        count_cols = json.load(f)

    wait_booster = xgb.Booster()
    wait_booster.load_model(str(WAITTIME_MODEL_PATH))

    queue_booster = xgb.Booster()
    queue_booster.load_model(str(QUEUE_MODEL_PATH))

    with open(MULTI_COLUMNS_PATH, "r", encoding="utf-8") as f:
        multi_cols = json.load(f)

    return count_booster, count_cols, wait_booster, queue_booster, multi_cols


@st.cache_data(show_spinner=False)
def load_holiday_set():
    if not HOLIDAY_CSV_PATH.exists():
        return set()

    try:
        h = pd.read_csv(HOLIDAY_CSV_PATH, encoding="shift_jis")
    except Exception:
        h = pd.read_csv(HOLIDAY_CSV_PATH, encoding="utf-8")

    date_col = "国民の祝日・休日月日"
    if date_col not in h.columns:
        for c in h.columns:
            if "月日" in c or "日付" in c or "date" in c.lower():
                date_col = c
                break

    h[date_col] = pd.to_datetime(h[date_col], errors="coerce")
    h = h.dropna(subset=[date_col])
    return set(h[date_col].dt.date.tolist())


def is_holiday_like(d: pd.Timestamp, holiday_set: set) -> bool:
    dd = d.date()
    if dd in holiday_set:
        return True
    if d.weekday() >= 5:
        return True
    if (d.month == 12 and d.day >= 29) or (d.month == 1 and d.day <= 3):
        return True
    return False


def make_time_slots(target_date: pd.Timestamp) -> pd.DatetimeIndex:
    start = target_date.replace(hour=8, minute=0, second=0, microsecond=0)
    end = target_date.replace(hour=18, minute=0, second=0, microsecond=0)
    return pd.date_range(start=start, end=end, freq="30min")


def build_zero_row(columns):
    return pd.DataFrame(np.zeros((1, len(columns))), columns=columns)


def booster_predict(booster: xgb.Booster, X: pd.DataFrame) -> np.ndarray:
    dm = xgb.DMatrix(X)
    return booster.predict(dm)


def simulate_day(
    target_date: pd.Timestamp,
    total_patients: int,
    weather_label: str,
    holiday_set: set,
    count_booster: xgb.Booster,
    count_cols: list,
    wait_booster: xgb.Booster,
    queue_booster: xgb.Booster,
    multi_cols: list,
):
    is_holiday_daily = is_holiday_like(target_date, holiday_set)
    prev_date = target_date - pd.Timedelta(days=1)
    is_prev_holiday = is_holiday_like(prev_date, holiday_set)

    results = []
    lags = {"lag_30min": 0.0, "lag_60min": 0.0, "lag_90min": 0.0}
    queue_at_start = 0

    for ts in make_time_slots(target_date):
        # 1) predict reception count
        cf = build_zero_row(count_cols)
        cf.loc[0, "hour"] = ts.hour
        cf.loc[0, "minute"] = ts.minute
        if "is_first_slot" in cf.columns:
            cf.loc[0, "is_first_slot"] = 1 if (ts.hour == 8 and ts.minute == 0) else 0
        if "is_second_slot" in cf.columns:
            cf.loc[0, "is_second_slot"] = 1 if (ts.hour == 8 and ts.minute == 30) else 0

        if "月" in cf.columns:
            cf.loc[0, "月"] = ts.month
        if "週回数" in cf.columns:
            cf.loc[0, "週回数"] = (ts.day - 1) // 7 + 1
        if "前日祝日フラグ" in cf.columns:
            cf.loc[0, "前日祝日フラグ"] = int(is_prev_holiday)

        if "total_outpatient_count" in cf.columns:
            cf.loc[0, "total_outpatient_count"] = total_patients
        if "is_holiday" in cf.columns:
            cf.loc[0, "is_holiday"] = int(is_holiday_daily)

        if "雨フラグ" in cf.columns:
            cf.loc[0, "雨フラグ"] = 1 if "雨" in weather_label else 0
        if "雪フラグ" in cf.columns:
            cf.loc[0, "雪フラグ"] = 1 if "雪" in weather_label else 0

        weather_cat = f"天気カテゴリ_{weather_label[0]}"
        if weather_cat in cf.columns:
            cf.loc[0, weather_cat] = 1

        dow = f"dayofweek_{ts.dayofweek}"
        if dow in cf.columns:
            cf.loc[0, dow] = 1

        rolling_mean = (lags["lag_30min"] + lags["lag_60min"]) / 2.0
        for k, v in lags.items():
            if k in cf.columns:
                cf.loc[0, k] = v
        if "rolling_mean_60min" in cf.columns:
            cf.loc[0, "rolling_mean_60min"] = rolling_mean

        pred_reception = float(booster_predict(count_booster, cf[count_cols])[0])
        predicted_reception_count = max(0, int(round(pred_reception)))

        # 2) predict queue & waittime
        mf = build_zero_row(multi_cols)
        mf.loc[0, "hour"] = ts.hour
        mf.loc[0, "minute"] = ts.minute

        if "reception_count" in mf.columns:
            mf.loc[0, "reception_count"] = predicted_reception_count
        if "queue_at_start_of_slot" in mf.columns:
            mf.loc[0, "queue_at_start_of_slot"] = queue_at_start

        if "月" in mf.columns:
            mf.loc[0, "月"] = ts.month
        if "週回数" in mf.columns:
            mf.loc[0, "週回数"] = (ts.day - 1) // 7 + 1
        if "前日祝日フラグ" in mf.columns:
            mf.loc[0, "前日祝日フラグ"] = int(is_prev_holiday)

        if "total_outpatient_count" in mf.columns:
            mf.loc[0, "total_outpatient_count"] = total_patients
        if "is_holiday" in mf.columns:
            mf.loc[0, "is_holiday"] = int(is_holiday_daily)

        if "雨フラグ" in mf.columns:
            mf.loc[0, "雨フラグ"] = 1 if "雨" in weather_label else 0
        if "雪フラグ" in mf.columns:
            mf.loc[0, "雪フラグ"] = 1 if "雪" in weather_label else 0

        weather_cat_m = f"天気カテゴリ_{weather_label[0]}"
        if weather_cat_m in mf.columns:
            mf.loc[0, weather_cat_m] = 1

        dow_m = f"dayofweek_{ts.dayofweek}"
        if dow_m in mf.columns:
            mf.loc[0, dow_m] = 1

        pred_queue = float(booster_predict(queue_booster, mf[multi_cols])[0])
        predicted_queue = max(0, int(round(pred_queue)))

        pred_wait = float(booster_predict(wait_booster, mf[multi_cols])[0])
        predicted_wait = max(0, int(round(pred_wait)))

        results.append(
            {
                "時間帯": ts.strftime("%H:%M"),
                "予測受付数": predicted_reception_count,
                "予測待ち人数(人)": predicted_queue,
                "予測平均待ち時間(分)": predicted_wait,
            }
        )

        lags = {"lag_30min": predicted_reception_count, "lag_60min": lags["lag_30min"], "lag_90min": lags["lag_60min"]}
        queue_at_start = predicted_queue

    return pd.DataFrame(results)


def plot_results(df: pd.DataFrame, title: str):
    fig, ax1 = plt.subplots(figsize=(14, 5))
    max_queue = float(df["予測待ち人数(人)"].max())
    max_wait = float(df["予測平均待ち時間(分)"].max())

    bars = ax1.bar(df["時間帯"], df["予測待ち人数(人)"])
    ax1.set_xlabel("時間帯")
    ax1.set_ylabel("予測待ち人数(人)")
    ax1.tick_params(axis="x", rotation=45)
    ax1.set_ylim(0, max_queue * 1.3 if max_queue > 0 else 1)

    for bar in bars:
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            0.02,
            int(bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
            transform=ax1.get_xaxis_transform(),
        )

    ax2 = ax1.twinx()
    ax2.plot(df["時間帯"], df["予測平均待ち時間(分)"], marker="o")
    ax2.set_ylabel("予測平均待ち時間(分)")
    ax2.set_ylim(0, max_wait * 1.3 if max_wait > 0 else 1)

    for i, v in enumerate(df["予測平均待ち時間(分)"].astype(int).tolist()):
        ax2.annotate(v, (df["時間帯"].iloc[i], df["予測平均待ち時間(分)"].iloc[i]), textcoords="offset points", xytext=(0, 14), ha="center", fontsize=10)

    plt.title(title)
    fig.tight_layout()
    return fig


st.title("🏥 A病院 待ち人数・待ち時間 統合予測（Streamlit Cloud）")
st.caption("models/ に学習済みモデル(JSON)とカラム(JSON)を配置してください。祝日判定は data/syukujitsu.csv を使用します。")

required = [COUNT_MODEL_PATH, COUNT_COLUMNS_PATH, WAITTIME_MODEL_PATH, QUEUE_MODEL_PATH, MULTI_COLUMNS_PATH, HOLIDAY_CSV_PATH]
missing = [p.name for p in required if not p.exists()]
if missing:
    st.error("必要ファイルが不足しています。以下をリポジトリに配置してください：")
    for name in missing:
        st.write(f"- {name}")
    st.stop()

holiday_set = load_holiday_set()
count_booster, count_cols, wait_booster, queue_booster, multi_cols = load_models_and_columns()
st.success("✅ モデル読込完了（Boosterモード：scikit-learn不要）")

with st.sidebar:
    st.header("入力")
    target = st.date_input("予測対象日", value=date.today() + timedelta(days=1))
    total_patients = st.number_input("延べ外来患者数", min_value=0, value=1200, step=10)
    weather = st.selectbox("天気予報", options=["晴", "曇", "雨", "雪", "快晴", "薄曇"], index=0)
    run = st.button("シミュレーション実行", type="primary")

if run:
    with st.spinner("シミュレーション計算中..."):
        target_ts = pd.to_datetime(target)
        df = simulate_day(
            target_ts,
            int(total_patients),
            str(weather),
            holiday_set,
            count_booster,
            count_cols,
            wait_booster,
            queue_booster,
            multi_cols,
        )

    st.subheader(f"結果：{target_ts.strftime('%Y-%m-%d')}")
    fig = plot_results(df, f"{target_ts.strftime('%Y-%m-%d')} の予測待ち人数と平均待ち時間")
    st.pyplot(fig, clear_figure=True)
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button("CSVをダウンロード", data=csv, file_name=f"A_prediction_{target_ts.strftime('%Y%m%d')}.csv", mime="text/csv")

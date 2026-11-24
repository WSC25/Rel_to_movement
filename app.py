from pathlib import Path
import altair as alt
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
import streamlit as st

# ==========================================
# 0. 網頁頁面基本設定與校正器類別定義
# ==========================================
st.set_page_config(page_title="投球表現與相似投手分析", page_icon="⚾", layout="wide")
st.title("⚾ 投球表現預測 & 相似投手比對(挑出的是比賽中的其中一顆 而非投手整體平均)")


class StadiumExtensionCalibrator:

    def __init__(self, reference_stadium=None):
        self.reference_stadium = reference_stadium
        self.offsets_ = {}
        self.global_bias_ = 0.0

    def fit(self, df, pitcher_col='pName', pitch_col='球種', stadium_col='球場', target_col='Extension'):
        data = df[[pitcher_col, pitch_col, stadium_col, target_col]].dropna().copy()
        data[target_col] = pd.to_numeric(data[target_col], errors='coerce')
        data = data.dropna(subset=[target_col])

        data['pitcher_pitch'] = data[pitcher_col].astype(str) + '_' + data[pitcher_col].astype(str)
        counts = data['pitcher_pitch'].value_counts()
        valid_groups = counts[counts >= 2].index
        data = data[data['pitcher_pitch'].isin(valid_groups)]

        formula = f'{target_col} ~ C({stadium_col}) - 1'
        model = smf.mixedlm(formula, data, groups=data['pitcher_pitch'])
        result = model.fit(reml=True)

        stadium_effects = {}
        for param_name, val in result.fe_params.items():
            if f'C({stadium_col})[' in param_name:
                st_name = param_name.split('[')[1].rstrip(']')
                stadium_effects[st_name] = val

        if self.reference_stadium and self.reference_stadium in stadium_effects:
            baseline = stadium_effects[self.reference_stadium]
        else:
            baseline = np.mean(list(stadium_effects.values()))

        self.offsets_ = {st: val - baseline for st, val in stadium_effects.items()}
        return self

    def transform(self, df, stadium_col='球場', target_col='Extension', output_col='Extension_calibrated'):
        df_out = df.copy()
        bias = df_out[stadium_col].map(self.offsets_).fillna(0.0)
        df_out[output_col] = pd.to_numeric(df_out[target_col], errors='coerce') - bias
        return df_out

    def get_offsets_summary(self):
        return pd.Series(self.offsets_, name='Stadium_Offset').sort_values()


# 各球場推算投手丘高度 (cm)
MOUND_HEIGHT_CM = {
    "澄清湖": 29.62,
    "屏東": 29.28,
    "天母": 28.74,
    "桃園": 28.27,
    "新莊": 27.65,
    "青埔": 27.57,
    "亞太主": 27.20,
    "台鋼科大": 27.15,
    "大巨蛋": 26.73,
    "斗六": 25.56,
    "台中": 25.40,
}

CALIBRATOR_PATH = Path(__file__).resolve().parent / "extension_calibrator.joblib"

# ==========================================
# 1. 讀取資料、校正模型與訓練預測模型
# ==========================================
pitch_type_col = '球種'
raw_ext_col = 'Extension'
cal_ext_col = 'Extension_calibrated'
features = ['RelHeight_Correct', 'RelSide', cal_ext_col, 'RelSpeed']
targets = [
    'VertBreak', 'InducedVertBreak', 'HorzBreak',
    'VertApprAngle', 'HorzApprAngle', 'Stuff+'
]


@st.cache_resource
def load_extension_calibrator():
    if CALIBRATOR_PATH.exists():
        return joblib.load(CALIBRATOR_PATH)
    return None


calibrator = load_extension_calibrator()
if calibrator is None:
    st.warning(f"找不到延伸校正模型檔案（{CALIBRATOR_PATH}），目前將以原始 Extension 數值計算。")


def calibrate_extension_feet(raw_extension_feet, stadium):
    if calibrator is not None and hasattr(calibrator, 'offsets_'):
        offset = float(calibrator.offsets_.get(stadium, 0.0))
        return float(raw_extension_feet) - offset
    return float(raw_extension_feet)


@st.cache_resource
def load_data_and_train(_calibrator):
    df = pd.read_parquet('pitch_data.parquet')

    if cal_ext_col not in df.columns and raw_ext_col in df.columns:
        if _calibrator is not None and hasattr(_calibrator, 'transform') and '球場' in df.columns:
            df = _calibrator.transform(df, stadium_col='球場', target_col=raw_ext_col, output_col=cal_ext_col)
        else:
            df[cal_ext_col] = pd.to_numeric(df[raw_ext_col], errors='coerce')

    models = {}
    for p_type, group in df.groupby(pitch_type_col):
        models[p_type] = {}
        for target in targets:
            sub_df = group[features + [target]].dropna()
            if len(sub_df) > 30:
                X = sub_df[features]
                y = sub_df[target]
                reg = LinearRegression()
                reg.fit(X, y)
                models[p_type][target] = reg
    return df, models


try:
    df, models = load_data_and_train(calibrator)
    available_pitches = list(models.keys())
except Exception as e:
    st.error(f"讀取資料失敗，請確認 'pitch_data.parquet' 檔案是否存在。錯誤訊息：{e}")
    st.stop()


# ==========================================
# 2. 核心分析函數
# ==========================================
def predict_pitch_performance_dual_unit(pitch_type, rel_height_cm, rel_side_cm, calibrated_extension_feet, relspeed_kmh):
    rel_height_feet = rel_height_cm / 30.48
    rel_side_feet = rel_side_cm / 30.48
    relspeed_mph = relspeed_kmh / 1.60934

    X_new = pd.DataFrame([[rel_height_feet, rel_side_feet, calibrated_extension_feet, relspeed_mph]], columns=features)

    predictions = {}
    raw_preds = {}
    for target, model in models[pitch_type].items():
        pred_imperial = float(model.predict(X_new)[0])
        raw_preds[target] = pred_imperial
        if target in ['VertBreak', 'InducedVertBreak', 'HorzBreak']:
            pred_metric = pred_imperial * 2.54
            predictions[target] = (f"{round(pred_metric, 2)} cm", f"{round(pred_imperial, 2)} in")
        else:
            unit_label = "度 (°)" if 'Angle' in target else "分"
            predictions[target] = (f"{round(pred_imperial, 2)} {unit_label}", None)
    return predictions, raw_preds


def find_similar_pitchers(df, pitch_type, rel_height_cm, rel_side_cm, calibrated_extension_feet, relspeed_kmh, top_n=10):
    sub_df = df[df[pitch_type_col] == pitch_type].dropna(subset=features).copy()
    if len(sub_df) < top_n:
        return pd.DataFrame()

    pitcher_col = 'pName' if 'pName' in sub_df.columns else None

    input_features = [[
        rel_height_cm / 30.48,
        rel_side_cm / 30.48,
        calibrated_extension_feet,
        relspeed_kmh / 1.60934
    ]]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(sub_df[features])
    X_input_scaled = scaler.transform(input_features)

    n_neighbors_to_fetch = min(len(sub_df), max(top_n * 20, 200))
    nn = NearestNeighbors(n_neighbors=n_neighbors_to_fetch, metric='euclidean')
    nn.fit(X_train_scaled)
    distances, indices = nn.kneighbors(X_input_scaled)

    selected_indices = []
    seen_pitchers = set()

    for idx in indices[0]:
        if pitcher_col is not None:
            p_name = sub_df.iloc[idx][pitcher_col]
            if p_name in seen_pitchers:
                continue
            seen_pitchers.add(p_name)
        selected_indices.append(idx)
        if len(selected_indices) == top_n:
            break

    return sub_df.iloc[selected_indices].copy()


def _get_symmetric_domain(series_list, extra_val=None):
    vals = []
    for s in series_list:
        if s is not None and len(s) > 0:
            clean = pd.to_numeric(s, errors='coerce').dropna()
            if not clean.empty:
                vals.extend([float(clean.min()), float(clean.max())])
    if extra_val is not None and pd.notna(extra_val):
        vals.append(float(extra_val))
    if not vals:
        return None
    max_abs = max(abs(min(vals)), abs(max(vals)))
    if max_abs == 0:
        max_abs = 1.0
    pad = max_abs * 0.08
    return [-max_abs - pad, max_abs + pad]


def build_xy_scatter_chart(
    bg_df,
    sim_df,
    x_col,
    y_col,
    x_title,
    y_title,
    chart_title,
    query_x,
    query_y,
    show_zero_axes=False,
    chart_height=360,
):
    layers = []

    if show_zero_axes:
        x_domain = _get_symmetric_domain(
            [
                bg_df[x_col] if (not bg_df.empty and x_col in bg_df.columns) else None,
                sim_df[x_col] if (not sim_df.empty and x_col in sim_df.columns) else None,
            ],
            extra_val=query_x,
        )
        y_domain = _get_symmetric_domain(
            [
                bg_df[y_col] if (not bg_df.empty and y_col in bg_df.columns) else None,
                sim_df[y_col] if (not sim_df.empty and y_col in sim_df.columns) else None,
            ],
            extra_val=query_y,
        )
        x_scale = alt.Scale(domain=x_domain) if x_domain else alt.Undefined
        y_scale = alt.Scale(domain=y_domain) if y_domain else alt.Undefined

        layers.append(
            alt.Chart(pd.DataFrame({"x0": [0.0]}))
            .mark_rule(color="#475569", strokeWidth=2)
            .encode(x=alt.X("x0:Q", scale=x_scale, title=x_title))
        )
        layers.append(
            alt.Chart(pd.DataFrame({"y0": [0.0]}))
            .mark_rule(color="#475569", strokeWidth=2)
            .encode(y=alt.Y("y0:Q", scale=y_scale, title=y_title))
        )
    else:
        x_scale = alt.Undefined
        y_scale = alt.Undefined

    if not bg_df.empty:
        bg_clean = bg_df[[x_col, y_col]].dropna().copy()
        if len(bg_clean) > 3000:
            bg_clean = bg_clean.sample(3000, random_state=42)
        layers.append(
            alt.Chart(bg_clean)
            .mark_circle(size=24, opacity=0.22)
            .encode(
                x=alt.X(f"{x_col}:Q", scale=x_scale, title=x_title),
                y=alt.Y(f"{y_col}:Q", scale=y_scale, title=y_title),
                color=alt.value("#94A3B8"),
                tooltip=[alt.Tooltip(f"{x_col}:Q", format=".2f"), alt.Tooltip(f"{y_col}:Q", format=".2f")],
            )
        )

    if not sim_df.empty and x_col in sim_df.columns and y_col in sim_df.columns:
        sim_clean = sim_df.dropna(subset=[x_col, y_col]).copy()
        if not sim_clean.empty:
            tooltip_cols = []
            if 'pName' in sim_clean.columns:
                tooltip_cols.append('pName:N')
            if '球場' in sim_clean.columns:
                tooltip_cols.append('球場:N')
            tooltip_cols.extend([
                alt.Tooltip(f"{x_col}:Q", format=".2f"),
                alt.Tooltip(f"{y_col}:Q", format=".2f"),
            ])
            layers.append(
                alt.Chart(sim_clean)
                .mark_circle(size=65, opacity=0.85, stroke="#FFFFFF", strokeWidth=1)
                .encode(
                    x=alt.X(f"{x_col}:Q", scale=x_scale, title=x_title),
                    y=alt.Y(f"{y_col}:Q", scale=y_scale, title=y_title),
                    color=alt.value("#2563EB"),
                    tooltip=tooltip_cols,
                )
            )

    if query_x is not None and query_y is not None and pd.notna(query_x) and pd.notna(query_y):
        q_df = pd.DataFrame({"x_val": [float(query_x)], "y_val": [float(query_y)]})
        layers.append(
            alt.Chart(q_df)
            .mark_point(shape="diamond", size=180, filled=True, color="#DC2626", stroke="#111827", strokeWidth=1.5)
            .encode(
                x=alt.X("x_val:Q", scale=x_scale, title=x_title),
                y=alt.Y("y_val:Q", scale=y_scale, title=y_title),
                tooltip=[
                    alt.Tooltip("x_val:Q", title=x_title, format=".2f"),
                    alt.Tooltip("y_val:Q", title=y_title, format=".2f"),
                ],
            )
        )

    if not layers:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_point().properties(title=chart_title, height=chart_height)

    return alt.layer(*layers).properties(title=chart_title, height=chart_height).interactive()


# ==========================================
# 3. 網頁側邊欄：使用者輸入控制區
# ==========================================
st.sidebar.header("🎛️ 輸入投球參數 (公制)")

selected_pitch = st.sidebar.selectbox("選擇球種", available_pitches, index=0)
selected_stadium = st.sidebar.selectbox("選擇球場", list(MOUND_HEIGHT_CM.keys()), index=0)
mound_height_cm = MOUND_HEIGHT_CM[selected_stadium]
st.sidebar.caption(f"推算投手丘高度：{mound_height_cm:.2f} cm")

rel_speed_kmh = st.sidebar.number_input("球速 (km/h)", min_value=80.0, max_value=175.0, value=137.0, step=0.5)
raw_rel_height_cm = st.sidebar.number_input("出手高度 (cm)", min_value=50.0, max_value=250.0, value=177.0, step=1.0)
rel_height_cm = raw_rel_height_cm - mound_height_cm
st.sidebar.caption(f"扣除投手丘後出手高度：{rel_height_cm:.2f} cm")

rel_side_cm = st.sidebar.number_input("側向位置 (cm)", min_value=-150.0, max_value=150.0, value=-46.0, step=1.0)
raw_extension_cm = st.sidebar.number_input("原始延伸距離 Extension (cm)", min_value=80.0, max_value=250.0, value=147.0, step=1.0)

raw_extension_feet = raw_extension_cm / 30.48
calibrated_extension_feet = calibrate_extension_feet(raw_extension_feet, selected_stadium)
calibrated_extension_cm = calibrated_extension_feet * 30.48
st.sidebar.caption(f"球場校正後延伸：{calibrated_extension_cm:.2f} cm ({calibrated_extension_feet:.2f} ft)")

top_n = st.sidebar.slider("顯示相似投手數量", min_value=3, max_value=30, value=30)

# ==========================================
# 4. 主畫面輸出區
# ==========================================
st.subheader(
    f"📊 1. 預測表現結果（球種：{selected_pitch}｜球場：{selected_stadium}，"
    f"扣除丘高 {mound_height_cm:.2f} cm 後高度 {rel_height_cm:.2f} cm｜"
    f"校正後延伸 {calibrated_extension_cm:.2f} cm）"
)

preds, raw_preds = predict_pitch_performance_dual_unit(
    selected_pitch, rel_height_cm, rel_side_cm, calibrated_extension_feet, rel_speed_kmh
)

cols = st.columns(3)
for idx, (metric, (primary_val, sub_val)) in enumerate(preds.items()):
    with cols[idx % 3]:
        st.metric(
            label=metric,
            value=primary_val,
            delta=f"英制: {sub_val}" if sub_val else None,
            delta_color="off"
        )

st.divider()

similar_df = find_similar_pitchers(
    df, selected_pitch, rel_height_cm, rel_side_cm, calibrated_extension_feet, rel_speed_kmh, top_n=top_n
)

# ==========================================
# 4.5 中段視覺化（統一公制 cm）：IVB-HB / VB-HB / 出手高度-側度 / 延伸-出手側度
# ==========================================
st.subheader("📈 投球軌跡與出手幾何 X-Y 圖（公制 cm｜灰點：同球種歷史｜藍點：相似投手｜紅菱形：目前輸入/預測）")

bg_pitch_df = df[df[pitch_type_col] == selected_pitch].copy()
for col_in, col_cm in [
    ('InducedVertBreak', 'IVB_cm'),
    ('VertBreak', 'VB_cm'),
    ('HorzBreak', 'HB_cm'),
]:
    if col_in in bg_pitch_df.columns:
        bg_pitch_df[col_cm] = pd.to_numeric(bg_pitch_df[col_in], errors='coerce') * 2.54

for col_in, col_cm in [
    ('RelHeight_Correct', 'RelHeight_cm'),
    ('RelSide', 'RelSide_cm'),
    (cal_ext_col, 'Extension_calibrated_cm'),
]:
    if col_in in bg_pitch_df.columns:
        bg_pitch_df[col_cm] = pd.to_numeric(bg_pitch_df[col_in], errors='coerce') * 30.48

sim_plot_df = similar_df.copy()
if not sim_plot_df.empty:
    for col_in, col_cm in [
        ('InducedVertBreak', 'IVB_cm'),
        ('VertBreak', 'VB_cm'),
        ('HorzBreak', 'HB_cm'),
    ]:
        if col_in in sim_plot_df.columns:
            sim_plot_df[col_cm] = pd.to_numeric(sim_plot_df[col_in], errors='coerce') * 2.54

    for col_in, col_cm in [
        ('RelHeight_Correct', 'RelHeight_cm'),
        ('RelSide', 'RelSide_cm'),
        (cal_ext_col, 'Extension_calibrated_cm'),
    ]:
        if col_in in sim_plot_df.columns:
            sim_plot_df[col_cm] = pd.to_numeric(sim_plot_df[col_in], errors='coerce') * 30.48

pred_ivb_cm = raw_preds['InducedVertBreak'] * 2.54 if 'InducedVertBreak' in raw_preds else None
pred_vb_cm = raw_preds['VertBreak'] * 2.54 if 'VertBreak' in raw_preds else None
pred_hb_cm = raw_preds['HorzBreak'] * 2.54 if 'HorzBreak' in raw_preds else None

mid_chart_height = 360

row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    st.altair_chart(
        build_xy_scatter_chart(
            bg_pitch_df, sim_plot_df, "HB_cm", "IVB_cm",
            "HorzBreak (cm)", "InducedVertBreak (cm)", "IVB - HB 圖 (cm)",
            pred_hb_cm, pred_ivb_cm,
            show_zero_axes=True,
            chart_height=mid_chart_height,
        ),
        use_container_width=True
    )

with row1_col2:
    st.altair_chart(
        build_xy_scatter_chart(
            bg_pitch_df, sim_plot_df, "HB_cm", "VB_cm",
            "HorzBreak (cm)", "VertBreak (cm)", "VB - HB 圖 (cm)",
            pred_hb_cm, pred_vb_cm,
            show_zero_axes=True,
            chart_height=mid_chart_height,
        ),
        use_container_width=True
    )

row2_col1, row2_col2 = st.columns(2)
with row2_col1:
    st.altair_chart(
        build_xy_scatter_chart(
            bg_pitch_df, sim_plot_df, "RelSide_cm", "RelHeight_cm",
            "RelSide 側度 (cm)", "RelHeight_Correct 出手高度 (cm)", "面向捕手 出手高度 - 側度 圖 (cm)",
            rel_side_cm, rel_height_cm,
            show_zero_axes=False,
            chart_height=mid_chart_height,
        ),
        use_container_width=True
    )

with row2_col2:
    st.altair_chart(
        build_xy_scatter_chart(
            bg_pitch_df, sim_plot_df, "RelSide_cm", "Extension_calibrated_cm",
            "RelSide 側度 (cm)", "Extension_calibrated 校正後延伸 (cm)", "延伸 - 出手側度 圖 (cm)",
            rel_side_cm, calibrated_extension_cm,
            show_zero_axes=False,
            chart_height=mid_chart_height,
        ),
        use_container_width=True
    )

st.divider()

# ==========================================
# 5. 相似投手表格
# ==========================================
st.subheader(f"🔍 2. 與輸入條件最相似的前 {top_n} 位投手（單一球比賽紀錄）")

if not similar_df.empty:
    target_cols = ['年', '球場','pName', '球種', 'RelHeight_Correct', 'RelSide', raw_ext_col, cal_ext_col, 'RelSpeed',
                   'VertBreak', 'InducedVertBreak', 'HorzBreak', 'Stuff+']
    display_cols = [c for c in target_cols if c in similar_df.columns]
    st.dataframe(similar_df[display_cols], use_container_width=True, hide_index=True)
else:
    st.warning("找不到足夠的相似樣本。")

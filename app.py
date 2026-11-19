import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# ==========================================
# 0. 網頁頁面基本設定
# ==========================================
st.set_page_config(page_title="投球表現與相似投手分析", page_icon="⚾", layout="wide")
st.title("⚾ 投球表現預測 & 相似投手比對(挑出的是比賽中的其中一顆 而非投手整體平均)")

# ==========================================
# 1. 讀取資料與訓練模型（使用快取加速，避免每次點擊都重訓）
# ==========================================
pitch_type_col = '球種'
features = ['RelHeight_Correct', 'RelSide', 'Extension', 'RelSpeed']
targets = [
    'VertBreak', 'InducedVertBreak', 'HorzBreak', 
    'VertApprAngle', 'HorzApprAngle', 'Stuff+'
]

@st.cache_resource
def load_data_and_train():
    # ⚠️ 請將這裡換成你實際的資料檔案路徑（例如 CSV 或 Excel）
    df = pd.read_parquet('pitch_data.parquet') 
    
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

# 載入資料與模型
try:
    df, models = load_data_and_train()
    available_pitches = list(models.keys())
except Exception as e:
    st.error(f"讀取資料失敗，請確認 'pitch_data.csv' 檔案是否存在。錯誤訊息：{e}")
    st.stop()

# ==========================================
# 2. 核心分析函數
# ==========================================
def predict_pitch_performance_dual_unit(pitch_type, rel_height_cm, rel_side_cm, extension_cm, relspeed_kmh):
    rel_height_feet = rel_height_cm / 30.48
    rel_side_feet = rel_side_cm / 30.48
    extension_feet = extension_cm / 30.48
    relspeed_mph = relspeed_kmh / 1.60934
    
    X_new = pd.DataFrame([[rel_height_feet, rel_side_feet, extension_feet, relspeed_mph]], columns=features)
    
    predictions = {}
    for target, model in models[pitch_type].items():
        pred_imperial = model.predict(X_new)[0]
        if target in ['VertBreak', 'InducedVertBreak', 'HorzBreak']:
            pred_metric = pred_imperial * 2.54
            predictions[target] = (f"{round(pred_metric, 2)} cm", f"{round(pred_imperial, 2)} in")
        else:
            unit_label = "度 (°)" if 'Angle' in target else "分"
            predictions[target] = (f"{round(pred_imperial, 2)} {unit_label}", None)
    return predictions

def find_similar_pitchers(df, pitch_type, rel_height_cm, rel_side_cm, extension_cm, relspeed_kmh, top_n=10):
    sub_df = df[df[pitch_type_col] == pitch_type].dropna(subset=features).copy()
    if len(sub_df) < top_n:
        return pd.DataFrame()

    pitcher_col = 'pName' if 'pName' in sub_df.columns else None

    input_features = [[
        rel_height_cm / 30.48,
        rel_side_cm / 30.48,
        extension_cm / 30.48,
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

# ==========================================
# 3. 網頁側邊欄：使用者輸入控制區
# ==========================================
st.sidebar.header("🎛️ 輸入投球參數 (公制)")

selected_pitch = st.sidebar.selectbox("選擇球種", available_pitches, index=0)
rel_speed_kmh = st.sidebar.number_input("球速 (km/h)", min_value=80.0, max_value=175.0, value=137.0, step=0.5)
rel_height_cm = st.sidebar.number_input("出手高度 (cm)", min_value=50.0, max_value=250.0, value=177.0, step=1.0)
rel_side_cm = st.sidebar.number_input("側向位置 (cm)", min_value=-150.0, max_value=150.0, value=-46.0, step=1.0)
extension_cm = st.sidebar.number_input("延伸距離 (cm)", min_value=80.0, max_value=250.0, value=147.0, step=1.0)
top_n = st.sidebar.slider("顯示相似投手數量", min_value=3, max_value=20, value=20)

# ==========================================
# 4. 主畫面輸出區
# ==========================================
st.subheader(f"📊 1. 預測表現結果（球種：{selected_pitch}）")

preds = predict_pitch_performance_dual_unit(selected_pitch, rel_height_cm, rel_side_cm, extension_cm, rel_speed_kmh)

# 用 3 欄卡片式排版呈現指標
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

st.subheader(f"🔍 2. 與輸入條件最相似的前 {top_n} 位投手（單一球比賽紀錄）")
similar_df = find_similar_pitchers(df, selected_pitch, rel_height_cm, rel_side_cm, extension_cm, rel_speed_kmh, top_n=top_n)

if not similar_df.empty:
    target_cols = ['年', 'pName', '球種', 'RelHeight_Correct', 'RelSide', 'Extension', 'RelSpeed', 
                   'VertBreak', 'InducedVertBreak', 'HorzBreak', 'Stuff+']
    display_cols = [c for c in target_cols if c in similar_df.columns]
    
    # 顯示可排序、可下載的互動式表格
    st.dataframe(similar_df[display_cols], use_container_width=True, hide_index=True)
else:
    st.warning("找不到足夠的相似樣本。")

import streamlit as st

st.set_page_config(page_title="3D化学模型メーカー", page_icon="🧪", layout="centered")

st.title("🧪 3D化学模型メーカー (3Dプリント対応)")
st.markdown("""
中学・高校の化学で学ぶ分子や結晶構造の **3Dプリント用データ（OBJ形式）** を生成するツールです。

### 👈 左のサイドメニューから作成モードを選んでください

#### 🧬 1. 分子模型maker
#### 🧊 2. 単位格子maker (金属・イオン結晶)
#### 💎 3. 炭素の同素体maker
""")

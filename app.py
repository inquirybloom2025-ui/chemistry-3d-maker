app_code = r"""
import streamlit as st
import numpy as np
import trimesh

# --- 1. 格子の枠（フレーム）を作成する関数 ---
def create_lattice_frame(w, h, d, thickness=0.1):
    """指定されたサイズの立方体枠を作成"""
    lines = [
        ([0,0,0], [w,0,0]), ([0,0,0], [0,h,0]), ([0,0,0], [0,0,d]),
        ([w,h,d], [0,h,d]), ([w,h,d], [w,0,d]), ([w,h,d], [w,h,0]),
        ([w,0,0], [w,h,0]), ([w,0,0], [w,0,d]),
        ([0,h,0], [w,h,0]), ([0,h,0], [0,h,d]),
        ([0,0,d], [w,0,d]), ([0,0,d], [0,h,d])
    ]
    meshes = []
    for s, e in lines:
        v = np.array(e) - np.array(s)
        length = np.linalg.norm(v)
        cyl = trimesh.creation.cylinder(radius=thickness, height=length, sections=8)
        # 回転と配置
        rot = trimesh.geometry.align_vectors([0, 0, 1], v)
        cyl.apply_transform(rot)
        cyl.apply_translation(np.array(s) + v/2)
        meshes.append(cyl)
    return trimesh.util.concatenate(meshes)

# --- 2. 結晶構造の生成 (繰り返し対応) ---
def generate_crystal(c_type, style, scale, repeat):
    nx, ny, nz = repeat
    meshes = []
    a = scale
    r_main = a * 0.15
    r_sub = a * 0.10
    
    # 繰り返しループ
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                offset = np.array([ix, iy, iz]) * a
                
                # --- NaCl (塩化ナトリウム) ---
                if "NaCl" in c_type:
                    # 4x4x4の格子点（単位格子内の全原子）
                    for dx in [0, 0.5]:
                        for dy in [0, 0.5]:
                            for dz in [0, 0.5]:
                                pos = (np.array([dx, dy, dz]) * a) + offset
                                # イオンの判定
                                is_cl = (round(dx*2) + round(dy*2) + round(dz*2)) % 2 == 0
                                r = r_main if is_cl else r_sub
                                atom = trimesh.creation.icosphere(subdivisions=2, radius=r)
                                atom.apply_translation(pos)
                                meshes.append(atom)

                # --- BCC (体心立方) ---
                elif "BCC" in c_type:
                    points = [[0,0,0], [0.5,0.5,0.5]]
                    for p in points:
                        atom = trimesh.creation.icosphere(subdivisions=2, radius=r_main)
                        atom.apply_translation(np.array(p)*a + offset)
                        meshes.append(atom)

                # --- FCC (面心立方) ---
                elif "FCC" in c_type:
                    points = [[0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]]
                    for p in points:
                        atom = trimesh.creation.icosphere(subdivisions=2, radius=r_main)
                        atom.apply_translation(np.array(p)*a + offset)
                        meshes.append(atom)

    # 全体を囲う枠を追加 (バラバラ防止)
    frame = create_lattice_frame(a*nx, a*ny, a*nz, thickness=a*0.05)
    meshes.append(frame)
    
    return trimesh.util.concatenate(meshes)

# --- UI ---
st.title("💎 繰り返し結晶モデル生成 (枠付き)")
c_type = st.selectbox("結晶構造", ["NaCl (塩化ナトリウム)", "BCC (体心立方)", "FCC (面心立方)"])
repeat_n = st.slider("繰り返し回数 (x, y, z共通)", 1, 3, 2)

if st.button("3Dモデル作成"):
    with st.spinner("構築中..."):
        mesh = generate_crystal(c_type, "Standard", 10.0, [repeat_n, repeat_n, repeat_n])
        mesh.export("crystal.obj")
        st.success(f"{repeat_n}x{repeat_n}x{repeat_n} の格子を作成しました")
        st.download_button("📥 OBJダウンロード", open("crystal.obj","rb"), file_name="crystal.obj")
"""
with open("app.py", "w", encoding="utf-8") as f:
    f.write(app_code)

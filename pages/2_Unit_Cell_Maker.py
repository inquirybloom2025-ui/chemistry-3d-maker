import streamlit as st
import numpy as np
import trimesh

# --- 1. 安全なスライス処理 ---
def safe_slice(mesh, normal, origin):
    if mesh is None or mesh.is_empty: return None
    bounds = mesh.bounds
    tol = 1e-4
    
    # 軸に平行なカットの場合は高速にスキップ判定
    if normal[0] == 1 and normal[1] == 0 and normal[2] == 0:
        if bounds[0][0] >= origin[0] - tol: return mesh
        if bounds[1][0] <= origin[0] + tol: return None
    elif normal[0] == -1 and normal[1] == 0 and normal[2] == 0:
        if bounds[1][0] <= origin[0] + tol: return mesh
        if bounds[0][0] >= origin[0] - tol: return None
    elif normal[0] == 0 and normal[1] == 1 and normal[2] == 0:
        if bounds[0][1] >= origin[1] - tol: return mesh
        if bounds[1][1] <= origin[1] + tol: return None
    elif normal[0] == 0 and normal[1] == -1 and normal[2] == 0:
        if bounds[1][1] <= origin[1] + tol: return mesh
        if bounds[0][1] >= origin[1] - tol: return None
    elif normal[0] == 0 and normal[1] == 0 and normal[2] == 1:
        if bounds[0][2] >= origin[2] - tol: return mesh
        if bounds[1][2] <= origin[2] + tol: return None
    elif normal[0] == 0 and normal[1] == 0 and normal[2] == -1:
        if bounds[1][2] <= origin[2] + tol: return mesh
        if bounds[0][2] >= origin[2] - tol: return None
        
    try:
        return trimesh.intersections.slice_mesh_plane(mesh, plane_normal=normal, plane_origin=origin, cap=True)
    except:
        return None

# --- 2. 結晶モデル生成のメイン関数 ---
def create_advanced_model(c_type, style, scale, do_cut, bond_thickness_ratio, rep, diagonal_cut):
    meshes = []
    a = scale
    
    # 枠線の太さは結合棒の半分(0.5倍)に設定
    thickness = a * bond_thickness_ratio * 0.5 
    is_space_filling = (style == "Space Filling (充填 - 棒なし)")

    atoms_data = []

    # 繰り返し回数に応じた原子座標の生成
    if "NaCl" in c_type:
        r_big = 0.28 * a if is_space_filling else 0.15 * a
        r_small = 0.22 * a if is_space_filling else 0.10 * a
        for x in np.arange(0, rep + 0.1, 0.5):
            for y in np.arange(0, rep + 0.1, 0.5):
                for z in np.arange(0, rep + 0.1, 0.5):
                    is_cl = (int(round(x*2)) + int(round(y*2)) + int(round(z*2))) % 2 == 0
                    r = r_big if is_cl else r_small
                    atoms_data.append(([x*a, y*a, z*a], r))
        nn_dist = 0.5 * a

    elif "BCC" in c_type:
        r = (np.sqrt(3)/4)*a if is_space_filling else 0.15*a
        for ix in range(rep):
            for iy in range(rep):
                for iz in range(rep):
                    base = np.array([ix, iy, iz])
                    coords = [base + [0,0,0], base + [1,0,0], base + [0,1,0], base + [0,0,1], 
                              base + [1,1,0], base + [1,0,1], base + [0,1,1], base + [1,1,1], 
                              base + [0.5,0.5,0.5]]
                    for c in coords:
                        atoms_data.append(([c[0]*a, c[1]*a, c[2]*a], r))
        nn_dist = (np.sqrt(3)/2) * a

    elif "FCC" in c_type:
        r = (np.sqrt(2)/4)*a if is_space_filling else 0.15*a
        for ix in range(rep):
            for iy in range(rep):
                for iz in range(rep):
                    base = np.array([ix, iy, iz])
                    coords = [base + [0,0,0], base + [1,0,0], base + [0,1,0], base + [0,0,1], 
                              base + [1,1,0], base + [1,0,1], base + [0,1,1], base + [1,1,1],
                              base + [0.5,0.5,0], base + [0.5,0.5,1], base + [0.5,0,0.5], 
                              base + [0.5,1,0.5], base + [0,0.5,0.5], base + [1,0.5,0.5]]
                    for c in coords:
                        atoms_data.append(([c[0]*a, c[1]*a, c[2]*a], r))
        nn_dist = (np.sqrt(2)/2) * a

    # 重複する原子を削除（隣り合う単位格子の境界部分）
    unique_atoms = []
    seen = set()
    for pos, r in atoms_data:
        key = (round(pos[0], 3), round(pos[1], 3), round(pos[2], 3))
        if key not in seen:
            seen.add(key)
            unique_atoms.append((pos, r))

    # 斜め切断用の法線ベクトル (110面)
    diag_normal = np.array([1.0, -1.0, 0.0])
    diag_normal = diag_normal / np.linalg.norm(diag_normal)

    # 原子のメッシュ生成とカット
    for pos, r in unique_atoms:
        mesh = trimesh.creation.icosphere(subdivisions=4 if is_space_filling else 3, radius=r)
        rot_hack = trimesh.transformations.rotation_matrix(0.123, [1, 1, 1])
        mesh.apply_transform(rot_hack)
        mesh.apply_translation(pos)
        
        if do_cut:
            max_v = a * rep
            mesh = safe_slice(mesh, [1,0,0], [0,0,0])
            mesh = safe_slice(mesh, [-1,0,0], [max_v,0,0])
            mesh = safe_slice(mesh, [0,1,0], [0,0,0])
            mesh = safe_slice(mesh, [0,-1,0], [0,max_v,0])
            mesh = safe_slice(mesh, [0,0,1], [0,0,0])
            mesh = safe_slice(mesh, [0,0,-1], [0,0,max_v])
            
        if diagonal_cut:
            mesh = safe_slice(mesh, diag_normal, [0, 0, 0])
        
        if mesh and not mesh.is_empty:
            meshes.append(mesh)

    # 結合棒の生成とカット
    if not is_space_filling:
        bond_radius = a * bond_thickness_ratio
        num_atoms = len(unique_atoms)
        for i in range(num_atoms):
            for j in range(i+1, num_atoms):
                p1 = np.array(unique_atoms[i][0])
                p2 = np.array(unique_atoms[j][0])
                dist = np.linalg.norm(p2 - p1)
                
                if abs(dist - nn_dist) < 1e-3:
                    cyl = trimesh.creation.cylinder(radius=bond_radius, height=dist, sections=10)
                    vec = p2 - p1
                    z_axis = np.array([0,0,1])
                    ax = np.cross(z_axis, vec)
                    if np.linalg.norm(ax) < 1e-6:
                        rot = np.eye(4) if vec[2] > 0 else trimesh.transformations.rotation_matrix(np.pi, [1,0,0])
                    else:
                        ang = np.arccos(np.dot(z_axis, vec) / dist)
                        rot = trimesh.transformations.rotation_matrix(ang, ax)
                    
                    cyl.apply_transform(trimesh.transformations.translation_matrix((p1 + p2) / 2) @ rot)
                    
                    if do_cut:
                        max_v = a * rep
                        cyl = safe_slice(cyl, [1,0,0], [0,0,0])
                        cyl = safe_slice(cyl, [-1,0,0], [max_v,0,0])
                        cyl = safe_slice(cyl, [0,1,0], [0,0,0])
                        cyl = safe_slice(cyl, [0,-1,0], [0,max_v,0])
                        cyl = safe_slice(cyl, [0,0,1], [0,0,0])
                        cyl = safe_slice(cyl, [0,0,-1], [0,0,max_v])
                        
                    if diagonal_cut:
                        cyl = safe_slice(cyl, diag_normal, [0, 0, 0])
                        
                    if cyl and not cyl.is_empty:
                        meshes.append(cyl)

    # 枠線の生成（繰り返し数に対応し、斜め切断時は枠も切断する）
    width = height = depth = a * rep
    lines = [
        ([0,0,0], [width,0,0]), ([0,0,0], [0,height,0]), ([0,0,0], [0,0,depth]),
        ([width,height,depth], [0,height,depth]), ([width,height,depth], [width,0,depth]), ([width,height,depth], [width,height,0]),
        ([width,0,0], [width,height,0]), ([width,0,0], [width,0,depth]),
        ([0,height,0], [width,height,0]), ([0,height,0], [0,height,depth]),
        ([0,0,depth], [width,0,depth]), ([0,0,depth], [0,height,depth])
    ]
    for start, end in lines:
        p1 = np.array(start); p2 = np.array(end)
        vec = p2 - p1
        length = np.linalg.norm(vec)
        if length < 1e-6: continue

        cyl = trimesh.creation.cylinder(radius=thickness, height=length, sections=8)
        z_axis = np.array([0, 0, 1])
        ax = np.cross(z_axis, vec)
        if np.linalg.norm(ax) < 1e-6:
            rot = np.eye(4) if vec[2] > 0 else trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
        else:
            ang = np.arccos(np.dot(z_axis, vec) / length)
            rot = trimesh.transformations.rotation_matrix(ang, ax)

        cyl.apply_transform(trimesh.transformations.translation_matrix((p1 + p2) / 2) @ rot)
        
        if diagonal_cut:
            cyl = safe_slice(cyl, diag_normal, [0,0,0])
            
        if cyl and not cyl.is_empty:
            meshes.append(cyl)

    if not meshes: return None
    combined = trimesh.util.concatenate(meshes)
    try: combined.fix_normals()
    except: pass
    return combined

# --- UI (Streamlit) ---
st.set_page_config(page_title="単位格子メーカー", page_icon="🧊", layout="wide")
st.title("🧪 教科書完全準拠：結晶モデルメーカー")

c_type = st.sidebar.selectbox("結晶構造", ["NaCl (塩化ナトリウム)", "BCC (体心立方)", "FCC (面心立方)"])
style = st.sidebar.radio("スタイル", ["Space Filling (充填 - 棒なし)", "Ball and Stick (球棒 - 棒あり)"])

st.sidebar.markdown("---")
st.sidebar.header("モデル設定")

# 新機能：繰り返しの数
rep = st.sidebar.slider("繰り返しの数 (XYZ方向)", min_value=1, max_value=3, value=1, help="単位格子を連続させて造形します")

# 新機能：斜めに切断モード
diagonal_cut = st.sidebar.checkbox("単位格子を斜めに切断 (110面)", value=False, help="体心立方格子などで、中心の原子が接する美しい断面を観察できます。")

# 棒ありの時だけ「細さ調整スライダー」を表示
if style == "Ball and Stick (球棒 - 棒あり)":
    bond_thickness = st.sidebar.slider("結合棒の太さ（※枠線は自動でこの半分の細さになります）", 
                               min_value=0.05, max_value=0.30, value=0.12, step=0.01)
else:
    bond_thickness = 0.12

c1, c2 = st.columns([1, 1])
with c1:
    st.info("👈 左側のメニューから設定を選び、「モデル作成」ボタンを押してください。")
with c2:
    if st.button("モデル作成 (OBJ形式)", type="primary"):
        with st.spinner("メッシュ構築中（面のカット処理を行っています）..."):
            full_mesh = create_advanced_model(c_type, style, 10.0, True, bond_thickness, rep, diagonal_cut)
            if full_mesh and not full_mesh.is_empty:
                full_mesh.export("model.obj")
                st.success("モデルが完成しました！")
                with open("model.obj", "rb") as f:
                    st.download_button("📥 OBJファイルをダウンロード", f, file_name="crystal_model.obj")
            else:
                st.error("メッシュの生成に失敗しました。")

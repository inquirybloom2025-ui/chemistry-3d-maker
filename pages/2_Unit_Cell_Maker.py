import streamlit as st
import numpy as np
import trimesh

# --- 1. 格子の枠（フレーム）を作成する関数 ---
def create_lattice_frame(width, height, depth, thickness=0.2):
    """格子の外周に細いシリンダーの枠を作る"""
    lines = [
        ([0,0,0], [width,0,0]), ([0,0,0], [0,height,0]), ([0,0,0], [0,0,depth]),
        ([width,height,depth], [0,height,depth]), ([width,height,depth], [width,0,depth]), ([width,height,depth], [width,height,0]),
        ([width,0,0], [width,height,0]), ([width,0,0], [width,0,depth]),
        ([0,height,0], [width,height,0]), ([0,height,0], [0,height,depth]),
        ([0,0,depth], [width,0,depth]), ([0,0,depth], [0,height,depth])
    ]
    frame_meshes = []
    for start, end in lines:
        p1 = np.array(start); p2 = np.array(end)
        vec = p2 - p1
        length = np.linalg.norm(vec)
        if length < 1e-6: continue

        cyl = trimesh.creation.cylinder(radius=thickness, height=length, sections=8)
        z = np.array([0, 0, 1])
        ax = np.cross(z, vec)
        if np.linalg.norm(ax) < 1e-6:
            rot = np.eye(4) if vec[2] > 0 else trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
        else:
            ang = np.arccos(np.dot(z, vec) / length)
            rot = trimesh.transformations.rotation_matrix(ang, ax)

        cyl.apply_transform(trimesh.transformations.translation_matrix((p1 + p2) / 2) @ rot)
        frame_meshes.append(cyl)
    return trimesh.util.concatenate(frame_meshes) if frame_meshes else None

# --- 2. 安全なスライス処理 ---
def safe_slice(mesh, normal, origin):
    if mesh is None or mesh.is_empty: return None
    bounds = mesh.bounds
    tol = 1e-4
    if normal[0] == 1:
        if bounds[0][0] >= origin[0] - tol: return mesh
        if bounds[1][0] <= origin[0] + tol: return None
    elif normal[0] == -1:
        if bounds[1][0] <= origin[0] + tol: return mesh
        if bounds[0][0] >= origin[0] - tol: return None
    elif normal[1] == 1:
        if bounds[0][1] >= origin[1] - tol: return mesh
        if bounds[1][1] <= origin[1] + tol: return None
    elif normal[1] == -1:
        if bounds[1][1] <= origin[1] + tol: return mesh
        if bounds[0][1] >= origin[1] - tol: return None
    elif normal[2] == 1:
        if bounds[0][2] >= origin[2] - tol: return mesh
        if bounds[1][2] <= origin[2] + tol: return None
    elif normal[2] == -1:
        if bounds[1][2] <= origin[2] + tol: return mesh
        if bounds[0][2] >= origin[2] - tol: return None
    return trimesh.intersections.slice_mesh_plane(mesh, plane_normal=normal, plane_origin=origin, cap=True)

# --- 3. 結晶モデル生成のメイン関数 ---
def create_advanced_model(c_type, style, scale, do_cut, bond_thickness_ratio):
    meshes = []
    a = scale
    
    # 枠線の太さは結合棒の半分(0.5倍)に設定
    thickness = a * bond_thickness_ratio * 0.5 
    is_space_filling = (style == "Space Filling (充填 - 棒なし)")

    atoms_data = []

    if "NaCl" in c_type:
        r_big = 0.28 * a if is_space_filling else 0.15 * a
        r_small = 0.22 * a if is_space_filling else 0.10 * a
        for x in [0, 0.5, 1]:
            for y in [0, 0.5, 1]:
                for z in [0, 0.5, 1]:
                    is_cl = ((x*2 + y*2 + z*2) % 2 == 0)
                    r = r_big if is_cl else r_small
                    atoms_data.append(([x*a, y*a, z*a], r))
        nn_dist = 0.5 * a

    elif "BCC" in c_type:
        r = (np.sqrt(3)/4)*a if is_space_filling else 0.15*a
        coords = [[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [1,1,1], [0.5,0.5,0.5]]
        for c in coords:
            atoms_data.append(([c[0]*a, c[1]*a, c[2]*a], r))
        nn_dist = (np.sqrt(3)/2) * a

    elif "FCC" in c_type:
        r = (np.sqrt(2)/4)*a if is_space_filling else 0.15*a
        coords = [[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [1,1,1],
                  [0.5,0.5,0], [0.5,0.5,1], [0.5,0,0.5], [0.5,1,0.5], [0,0.5,0.5], [1,0.5,0.5]]
        for c in coords:
            atoms_data.append(([c[0]*a, c[1]*a, c[2]*a], r))
        nn_dist = (np.sqrt(2)/2) * a

    for pos, r in atoms_data:
        mesh = trimesh.creation.icosphere(subdivisions=4 if is_space_filling else 3, radius=r)
        rot_hack = trimesh.transformations.rotation_matrix(0.123, [1, 1, 1])
        mesh.apply_transform(rot_hack)
        mesh.apply_translation(pos)
        
        if do_cut:
            mesh = safe_slice(mesh, [1,0,0], [0,0,0])
            mesh = safe_slice(mesh, [-1,0,0], [a,0,0])
            mesh = safe_slice(mesh, [0,1,0], [0,0,0])
            mesh = safe_slice(mesh, [0,-1,0], [0,a,0])
            mesh = safe_slice(mesh, [0,0,1], [0,0,0])
            mesh = safe_slice(mesh, [0,0,-1], [0,0,a])
        
        if mesh and not mesh.is_empty:
            meshes.append(mesh)

    if not is_space_filling:
        # ユーザーがスライダーで設定した細さを適用
        bond_radius = a * bond_thickness_ratio
        num_atoms = len(atoms_data)
        for i in range(num_atoms):
            for j in range(i+1, num_atoms):
                p1 = np.array(atoms_data[i][0])
                p2 = np.array(atoms_data[j][0])
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
                    meshes.append(cyl)

    frame = create_lattice_frame(a, a, a, thickness)
    if frame: meshes.append(frame)

    if not meshes: return None
    combined = trimesh.util.concatenate(meshes)
    try: combined.fix_normals()
    except: pass
    return combined

# --- UI (Streamlit) ---
st.title("🧪 教科書完全準拠：結晶モデルメーカー")
c_type = st.selectbox("結晶構造", ["NaCl (塩化ナトリウム)", "BCC (体心立方)", "FCC (面心立方)"])
style = st.radio("スタイル", ["Space Filling (充填 - 棒なし)", "Ball and Stick (球棒 - 棒あり)"])

# 棒ありの時だけ「細さ調整スライダー」を表示
if style == "Ball and Stick (球棒 - 棒あり)":
    st.markdown("---")
    bond_thickness = st.slider("結合棒の太さ（※枠線は自動でこの半分の細さになります）", 
                               min_value=0.01, max_value=0.08, value=0.04, step=0.005)
else:
    bond_thickness = 0.04

if st.button("3Dモデル(OBJ)を生成"):
    with st.spinner("メッシュ構築中（面のカット処理を行っています）..."):
        full_mesh = create_advanced_model(c_type, style, 10.0, True, bond_thickness)
        if full_mesh and not full_mesh.is_empty:
            full_mesh.export("model.obj")
            st.success("モデルが完成しました！")
            with open("model.obj", "rb") as f:
                st.download_button("📥 OBJファイルをダウンロード", f, file_name="crystal_model.obj")
        else:
            st.error("メッシュの生成に失敗しました。")

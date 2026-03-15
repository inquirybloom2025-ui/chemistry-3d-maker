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
        p1 = np.array(start)
        p2 = np.array(end)
        vec = p2 - p1
        length = np.linalg.norm(vec)
        if length < 1e-6: continue

        # 最初から正しい長さでシリンダーを作成
        cyl = trimesh.creation.cylinder(radius=thickness, height=length, sections=8)
        
        # 安全な回転処理
        z = np.array([0, 0, 1])
        ax = np.cross(z, vec)
        if np.linalg.norm(ax) < 1e-6:
            rot = np.eye(4) if vec[2] > 0 else trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
        else:
            ang = np.arccos(np.dot(z, vec) / length)
            rot = trimesh.transformations.rotation_matrix(ang, ax)

        # 回転と中心への移動を同時に適用
        cyl.apply_transform(trimesh.transformations.translation_matrix((p1 + p2) / 2) @ rot)
        frame_meshes.append(cyl)

    return trimesh.util.concatenate(frame_meshes) if frame_meshes else None

# --- 2. 原子パーツ作成 (スライス対応) ---
def get_oriented_part(radius, pos, box_w, box_h, box_d, do_cut):
    try:
        # 分割数（subdivisions）を3にして滑らかさを確保
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=radius)
        mesh.apply_translation(pos)
        
        if not do_cut: return mesh
        
        # 正しいカット処理（plane_originは必ず [x, y, z] の座標配列にする）
        cut_planes = [
            ([1, 0, 0], [0, 0, 0]),
            ([-1, 0, 0], [box_w, 0, 0]),
            ([0, 1, 0], [0, 0, 0]),
            ([0, -1, 0], [0, box_h, 0]),
            ([0, 0, 1], [0, 0, 0]),
            ([0, 0, -1], [0, 0, box_d])
        ]
        
        for normal, origin in cut_planes:
            mesh = trimesh.intersections.slice_mesh_plane(mesh, plane_normal=normal, plane_origin=origin, cap=True)
            # カットの途中で空っぽになったらNoneを返す
            if mesh is None or mesh.is_empty:
                return None
                
        return mesh
    except Exception:
        # 計算エラーが起きた場合は安全にスキップ
        return None

def create_advanced_model(c_type, style, scale, do_cut):
    meshes = []
    a = scale
    thickness = a * 0.03 # 枠の太さ

    # 結晶構造ごとの原子配置
    if "NaCl" in c_type:
        r_cl, r_na = (0.23*a, 0.15*a) if style == "Space Filling (充填)" else (0.15*a, 0.10*a)
        for x in [0, 0.5, 1]:
            for y in [0, 0.5, 1]:
                for z in [0, 0.5, 1]:
                    r = r_cl if (round(x*2)+round(y*2)+round(z*2)) % 2 == 0 else r_na
                    p = get_oriented_part(r, [x*a, y*a, z*a], a, a, a, do_cut)
                    if p and not p.is_empty: meshes.append(p)
        frame = create_lattice_frame(a, a, a, thickness)
        if frame: meshes.append(frame)

    elif "BCC" in c_type:
        r = (np.sqrt(3)/4)*a if style == "Space Filling (充填)" else 0.15*a
        for c in [[0.5,0.5,0.5], [0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [1,1,1]]:
            p = get_oriented_part(r, np.array(c)*a, a, a, a, do_cut)
            if p and not p.is_empty: meshes.append(p)
        frame = create_lattice_frame(a, a, a, thickness)
        if frame: meshes.append(frame)

    elif "FCC" in c_type:
        r = (np.sqrt(2)/4)*a if style == "Space Filling (充填)" else 0.15*a
        coords = [[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [1,1,1],
                  [0.5,0.5,0], [0.5,0.5,1], [0.5,0,0.5], [0.5,1,0.5], [0,0.5,0.5], [1,0.5,0.5]]
        for c in coords:
            p = get_oriented_part(r, np.array(c)*a, a, a, a, do_cut)
            if p and not p.is_empty: meshes.append(p)
        frame = create_lattice_frame(a, a, a, thickness)
        if frame: meshes.append(frame)

    else:
        # 汎用枠
        r = 0.15*a
        p = get_oriented_part(r, [0.5*a, 0.5*a, 0.5*a], a, a, a, do_cut)
        if p and not p.is_empty: meshes.append(p)
        frame = create_lattice_frame(a, a, a, thickness)
        if frame: meshes.append(frame)

    if not meshes:
        return None

    combined = trimesh.util.concatenate(meshes)
    try: combined.fix_normals()
    except: pass
    return combined

# --- UI (Streamlit) ---
st.title("🧪 枠付き結晶モデルメーカー")
c_type = st.selectbox("結晶構造", ["NaCl (塩化ナトリウム)", "BCC (体心立方)", "FCC (面心立方)"])
style = st.radio("スタイル", ["Space Filling (充填)", "Ball and Stick (球棒)"])

if st.button("3Dモデル(OBJ)を生成"):
    with st.spinner("メッシュ構築中..."):
        full_mesh = create_advanced_model(c_type, style, 10.0, True)
        if full_mesh and not full_mesh.is_empty:
            full_mesh.export("model.obj")
            st.success("枠付きモデルが完成しました！")
            st.download_button("📥 OBJファイルをダウンロード", open("model.obj","rb"), file_name="crystal_model.obj")
        else:
            st.error("メッシュの生成に失敗しました。")

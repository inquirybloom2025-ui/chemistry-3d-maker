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

# --- 2. 結晶モデル生成のメイン関数 ---
def create_advanced_model(c_type, style, scale, do_cut):
    meshes = []
    a = scale
    thickness = a * 0.03 # 枠の太さ
    is_space_filling = (style == "Space Filling (充填)")

    atoms_data = [] # (座標, 半径) のリスト

    # --- 結晶構造ごとの原子座標と半径の設定 ---
    if "NaCl" in c_type:
        # 充填モデルの時は、大きい原子(Cl)と小さい原子(Na)がピッタリくっつくサイズ(合計0.5a)にする
        r_big = 0.28 * a if is_space_filling else 0.15 * a
        r_small = 0.22 * a if is_space_filling else 0.10 * a
        # 3x3x3のグリッドで27個の原子を配置（枠の線上や角を含む）
        for x in [0, 0.5, 1]:
            for y in [0, 0.5, 1]:
                for z in [0, 0.5, 1]:
                    is_cl = ((x*2 + y*2 + z*2) % 2 == 0)
                    r = r_big if is_cl else r_small
                    atoms_data.append(([x*a, y*a, z*a], r))
        nn_dist = 0.5 * a # 結合（棒）をつなぐ距離

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

    # --- 原子(球)の描画とスライス（切断）処理 ---

import pymeshlab
import os
import trimesh
import numpy as np

ids = ['FaceTalk_170725_00137_TA', 'FaceTalk_170728_03272_TA', 'FaceTalk_170731_00024_TA', 'FaceTalk_170809_00138_TA',
       'FaceTalk_170811_03274_TA', 'FaceTalk_170811_03275_TA', 'FaceTalk_170904_00128_TA', 'FaceTalk_170904_03276_TA',
       'FaceTalk_170908_03277_TA', 'FaceTalk_170912_03278_TA', 'FaceTalk_170913_03279_TA', 'FaceTalk_170915_00223_TA']

ids = ['FaceTalk_170915_00223_TA']

exprs = ['bareteeth', 'cheeks_in', 'eyebrow', 'high_smile', 'lips_back', 'lips_up',
         'mouth_down', 'mouth_extreme', 'mouth_middle', 'mouth_open', 'mouth_side', 'mouth_up']
#exprs = ['bareteeth']


ref_path = "../../datasets/COMA"
base_path = "../../datasets/COMA_scans"
inter_path = "../../datasets/COMA_inter"
output_path = "../../datasets/COMA_preprocessed_scans"

def export_mesh(V, F, file_name):
    """
    Export mesh as .ply file from vertices coordinates and face connectivity
    """
    result = trimesh.exchange.ply.export_ply(trimesh.Trimesh(V,F), encoding='ascii')
    output_file = open(file_name, "wb+")
    output_file.write(result)
    output_file.close()

if not os.path.exists(inter_path):
    os.mkdir(inter_path)

if not os.path.exists(output_path):
    os.mkdir(output_path)

def count_mesh(dir):
    c = 0
    for elt in dir:
        if 'obj' in elt:
            c += 1
    return c

for id in ids:
    for expr in exprs:
        i = 1
        #length_expr = len(os.listdir(base_path + "/" + id + "/" + expr))
        length_expr = count_mesh(os.listdir(base_path + "/" + id + "/" + expr))
        print(length_expr)
        while i <= length_expr:
            if not os.path.exists(output_path + "/" + id + "/" + expr + "/" + str(expr) + f'.{i:06}'+".ply"):
                path = base_path + "/" + id + "/" + expr + "/" + str(expr) + f'.{i:06}' + ".obj"
                V,F = trimesh.load(path).vertices, trimesh.load(path).faces
                if os.path.exists(ref_path + "/" + id + "/" + expr + "/" + str(expr) + f'.{i:06}' + ".ply"):
                    V_ref = trimesh.load(ref_path + "/" + id + "/" + expr + "/" + str(expr) + f'.{i:06}' + ".ply").vertices

                    V = (V - np.mean(V, axis=0))/((np.max(V) - np.min(V)) / (np.max(V_ref) - np.min(V_ref)))
                    trans = np.stack((np.zeros(V.shape[0]), -0.03 * np.ones(V.shape[0]),
                                      -0.03 * np.ones(V.shape[0]))).T
                    V = (V + trans)*1.1

                    if not os.path.exists(inter_path + "/" + id):
                        os.mkdir(inter_path + "/" + id)
                    if not os.path.exists(inter_path + "/" + id + "/" + expr):
                        os.mkdir(inter_path + "/" + id + "/" + expr)
                    export_mesh(V, F, inter_path + "/" + id + "/" + expr + "/" + str(expr) + f'.{i:06}' + ".ply")

                    # create a new MeshSet
                    ms = pymeshlab.MeshSet()
                    ms.load_new_mesh(inter_path + "/" + id + "/" + expr + "/" + str(expr) + f'.{i:06}'+".ply")
                    ms.load_filter_script('preprocessing_scans.mlx')
                    ms.apply_filter_script()

                    if not os.path.exists(output_path + "/" + id):
                        os.mkdir(output_path + "/" + id)
                    if not os.path.exists(output_path + "/" + id + "/" + expr):
                        os.mkdir(output_path + "/" + id + "/" + expr)

                    ms.save_current_mesh(output_path + "/" + id + "/" + expr + "/" + str(expr) + f'.{i:06}'+".ply")



            i += 1


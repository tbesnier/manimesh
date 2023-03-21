import pymeshlab
import os

ids = ['FaceTalk_170725_00137_TA', 'FaceTalk_170728_03272_TA', 'FaceTalk_170731_00024_TA', 'FaceTalk_170809_00138_TA',
       'FaceTalk_170811_03274_TA', 'FaceTalk_170811_03275_TA', 'FaceTalk_170904_00128_TA', 'FaceTalk_170904_03276_TA',
       'FaceTalk_170908_03277_TA', 'FaceTalk_170912_03278_TA', 'FaceTalk_170913_03279_TA', 'FaceTalk_170915_00223_TA']

ids = ['FaceTalk_170728_03272_TA']

exprs = ['bareteeth', 'cheeks_in', 'eyebrow', 'high_smile', 'lips_back', 'lips_up',
         'mouth_down', 'mouth_extreme', 'mouth_middle', 'mouth_open', 'mouth_side', 'mouth_up']

exprs = ['bareteeth', 'high_smile', 'mouth_extreme']


base_path = "../../datasets/COMA_3"
output_path = "../../datasets/COMA_remeshed_Variable"

if not os.path.exists(output_path):
    os.mkdir(output_path)

for id in ids:
    for expr in exprs:
        i = 1
        length_expr = len(os.listdir(base_path + "/" + id + "/" + expr))
        while i <= length_expr:
            if not os.path.exists(output_path + "/" + id + "/" + expr + "/" + str(expr) + f'.{i:06}' + ".ply"):
                # create a new MeshSet
                ms = pymeshlab.MeshSet()
                ms.load_new_mesh(base_path + "/" + id + "/" + expr + "/" + str(expr) + f'.{i:06}'+".ply")
                ms.load_filter_script('variable.mlx')
                ms.apply_filter_script()

                if not os.path.exists(output_path + "/" + id):
                    os.mkdir(output_path + "/" + id)
                if not os.path.exists(output_path + "/" + id + "/" + expr):
                    os.mkdir(output_path + "/" + id + "/" + expr)

                ms.save_current_mesh(output_path + "/" + id + "/" + expr + "/" + str(expr) + f'.{i:06}'+".ply")
            i += 1


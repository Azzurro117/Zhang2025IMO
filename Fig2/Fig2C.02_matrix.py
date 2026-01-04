#! /usr/bin/python

import os, csv
import re
import itertools
import math
import numpy as np

def load_metrics_file(folder, file_path):
	iden_l, sim_l = [], []
	with open("%s/%s" % (folder, file_path)) as f:
		for line in f:
			line = line.strip().split(",")
			if line[1].startswith("S"):
				continue
			iden_l.append(float(line[2]))
			sim_l.append(float(line[3]))
	return iden_l, sim_l

def convert_to_triangle(arr):
    n = len(arr)
    # 验证数组长度是否为三角形数
    discriminant = 1 + 8 * n
    sqrt_disc = math.sqrt(discriminant)
    if not sqrt_disc.is_integer():
        raise ValueError(f"长度 {n} 不是三角形数（如1,3,6,10等），无法生成完美倒三角")
    k = (int(sqrt_disc) - 1) // 2  # 计算行数
    
    # 生成倒三角结构
    triangle = []
    start = 0
    for i in range(k):
        row_length = k - i  # 当前行有效元素数
        end = start + row_length
        row = [''] * i + arr[start:end]  # 前补空字符串
        triangle.append(row)
        start = end
    return triangle

def export_csv(matrix, current_folder, output_file):
	matrix_T = np.array(matrix).transpose()
	outf = csv.writer(open("%s/%s" % (current_folder, output_file), "w", newline=""))
	for line in matrix_T:
		outf.writerow(line)

if __name__ == "__main__":
	current_folder = os.path.dirname(os.path.abspath(__file__))
	input_file = "pairwise_metrics.csv" 
	output_file_iden = "pairwise_matrix.iden.csv"
	output_file_sim = "pairwise_matrix.sim.csv"
	iden_l, sim_l = load_metrics_file(current_folder, input_file)
	iden_tri = convert_to_triangle(iden_l)
	sim_tri = convert_to_triangle(sim_l)
	export_csv(iden_tri, current_folder, output_file_iden)
	export_csv(sim_tri, current_folder, output_file_sim)

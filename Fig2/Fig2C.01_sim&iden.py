#! /usr/bin/python

import os, csv
import re
import itertools
import Levenshtein

def load_aligned_sequences(folder, fasta_path):
	"""加载含gap的对齐FASTA文件"""
	sequences = {}
	current_seq = ""
	with open("%s/%s" % (folder, fasta_path)) as f:
		for line in f:
			line = line.strip()
			if line.startswith(">"):
				current_seq = line[1:].split()[0]
				sequences[current_seq] = []
			else:
				sequences[current_seq].extend(list(line.upper()))
	return sequences

def calculate_metrics(seq1, seq2):
    """计算两序列的Identity和Similarity"""
    identity_match = 0
    similarity_match = 0
    valid_sites = 0      # 用于Identity计算的位点（排除gap和N）
    total_sites = 0      # 用于Similarity计算的位点（排除N）
    ambiguous = 0        # 含N的位点
    
    for a, b in zip(seq1, seq2):
        # 跳过含N的位点
        if a == 'N' or b == 'N':
            ambiguous += 1
            continue
        
        total_sites += 1  # Similarity统计所有非N位点
        
        # Similarity计算逻辑（包含gap处理）
        if a == b or (a == '-' and b == '-'):
            similarity_match += 1
        
        # Identity计算逻辑（排除gap）
        if a != '-' and b != '-':
            valid_sites += 1
            if a == b:
                identity_match += 1
    
    # 处理除零错误
    identity = identity_match / valid_sites if valid_sites > 0 else 0.0
    similarity = similarity_match / total_sites if total_sites > 0 else 0.0
    
    return {
        'identity': round(identity, 4),
        'similarity': round(similarity, 4),
        'valid_sites': valid_sites,
        'total_sites': total_sites,
        'ambiguous': ambiguous
    }

def pairwise_analysis(sequences, folder, output_path):
    """执行两两分析并输出结果"""
    results = []
    ids = list(sequences.keys())
    
    for id1, id2 in itertools.combinations(ids, 2):
        metrics = calculate_metrics(sequences[id1], sequences[id2])
        results.append({
            'Sequence1': id1,
            'Sequence2': id2,
            'Identity': metrics['identity'],
            'Similarity': metrics['similarity'],
            'Valid_Sites': metrics['valid_sites'],
            'Total_Sites': metrics['total_sites'],
            'Ambiguous_Sites': metrics['ambiguous'],
            'Alignment_Length': len(sequences[id1])
        })
    
    # 写入TSV文件
    with open("%s/%s" % (folder, output_path), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys(), delimiter=',')
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    current_folder = os.path.dirname(os.path.abspath(__file__))
    gene_name = "08.5b"
    sequence_set = "All_seq_queue_aligned.%s" % gene_name
    input_file = "%s.fas" % sequence_set # 替换为你的文件路径
    output_file = "pairwise_metrics.%s.csv" % gene_name

    # 读取和处理数据
    sequences = load_aligned_sequences(current_folder, input_file)
    pairwise_analysis(sequences, current_folder, output_file)
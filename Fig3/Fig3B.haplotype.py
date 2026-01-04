#! /usr/bin/python

import os, csv
import re
import Levenshtein
import networkx  as nx
from Bio import SeqIO
from Bio.Seq import Seq
from collections import defaultdict
import matplotlib.pyplot as plt

def read_sequences(filename):
	sequences_dict = defaultdict(list)
	lengths = set()
    
	for record in SeqIO.parse("%s/%s" % (current_folder, filename), "fasta"):
		seq = str(record.seq)
		sequences_dict[seq].append(record.id)  # 记录序列和对应的原始ID
		lengths.add(len(seq))
    
	if len(lengths) > 1:
		raise ValueError("错误：序列长度不一致，请确保已对齐")
	return sequences_dict

def hamming_distance(seq1, seq2):
	return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))

def build_haplotype_network(sequences_dict):
	unique_seqs = list(sequences_dict.keys())
	haplotype_ids = {seq: f'H{i+1}' for i, seq in enumerate(unique_seqs)}
    
	G = nx.Graph()
	for seq in unique_seqs:
		G.add_node(haplotype_ids[seq], size=len(sequences_dict[seq]), sequence=seq, original_ids=sequences_dict[seq])  # 确保属性名一致

	edge_list = []
	for i in range(len(unique_seqs)):
		for j in range(i+1, len(unique_seqs)):
			G.add_edge(haplotype_ids[unique_seqs[i]], haplotype_ids[unique_seqs[j]])
			G[haplotype_ids[unique_seqs[i]]][haplotype_ids[unique_seqs[j]]]["weight"] = hamming_distance(unique_seqs[i], unique_seqs[j])

	return nx.minimum_spanning_tree(G), haplotype_ids

def export_network_files(G, current_folder, nodef, edgef): #导出网络结构文件
	with open("%s/%s" % (current_folder, nodef), "w") as f:
		f.write("HaplotypeID\tFrequency\tSequence\n")
		for node in G.nodes:
			data = G.nodes[node]
			f.write(f"{node}\t{data['size']}\t{data['sequence']}\n")
    
    # 输出边信息
	with open("%s/%s" % (current_folder, edgef), "w") as f:
		f.write("Source\tTarget\tWeight\n")
		for edge in G.edges:
			f.write(f"{edge[0]}\t{edge[1]}\t{G[edge[0]][edge[1]]["weight"]}\n")

def export_haplotype_ids(G, current_folder, haplf): #导出单倍型与原始ID对应文件
	with open("%s/%s" % (current_folder, haplf), "w") as f:
		f.write("HaplotypeID\tOriginalIDs\n")
		for node in G.nodes:
			original_ids = ",".join(G.nodes[node]["original_ids"])
			f.write(f"{node}\t{original_ids}\n")
            
def plot_network(G):
    plt.figure(figsize=(20, 12))
    pos = nx.spring_layout(G)
    sizes = [G.nodes[node]['size'] * 100 for node in G]  # 节点大小根据频次调整
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, edge_color='gray')
    nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'label'))
    plt.title("AA network")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
	current_folder = os.path.dirname(os.path.abspath(__file__))
	sequence_set = "N-linked_glycation_sites_NTD"
	input_file = "Seq.%s.fas" % sequence_set # 替换为你的文件路径
	nodef = "network_nodes_%s.txt" % sequence_set
	edgef = "network_edges_%s.txt" % sequence_set
	haplf = "haplotype_original_ids_%s.txt" % sequence_set

    # 读取和处理数据
	sequences_dict = read_sequences(input_file)
	G, haplotype_ids = build_haplotype_network(sequences_dict)

    # 文件输出
	export_network_files(G, current_folder, nodef, edgef)
	export_haplotype_ids(G, current_folder, haplf)
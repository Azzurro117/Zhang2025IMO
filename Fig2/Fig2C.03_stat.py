#! /usr/bin/python

import os, csv
import re
import numpy as np
import Levenshtein
from Bio import SeqIO
from Bio.Seq import Seq

current_folder = os.path.dirname(os.path.abspath(__file__))

fn = "pairwise_metrics.08.5b"
out = "08.5b"

inf1 = csv.reader(open("%s/%s" % (current_folder, "Name.csv"), "r"))
inf2 = csv.reader(open("%s/%s.csv" % (current_folder, fn), "r"))
outf_iden = csv.writer(open("%s/%s" % (current_folder, "pairwise_matrix.iden.%s_iden.csv" % out), "w", newline = ""))
outf_sim = csv.writer(open("%s/%s" % (current_folder, "pairwise_matrix.iden.%s_sim.csv" % out), "w", newline = ""))

d = {}
for seq_name in inf1:
    if not seq_name[1] in d.keys():
        d[seq_name[1]] = []
    if seq_name[1] in d.keys():
        d[seq_name[1]].append(seq_name[0])

d_iden, d_sim = {}, {}
for key1 in d.keys():
    for key2 in d.keys():
        d2key = "%s_%s" % (key1, key2)
        d_iden[d2key] = []
        d_sim[d2key] = []

for line in inf2:
    for i in d.keys():
        for j in d.keys():
            if line[0] in d[i] and line[1] in d[j]:
                d_iden["%s_%s" % (i, j)].append(float(line[2]))
                d_sim["%s_%s" % (i, j)].append(float(line[3]))

for key_iden in d_iden.keys():
    if len(d_iden[key_iden]) > 0:
        out_iden = [key_iden, np.max(d_iden[key_iden]), np.min(d_iden[key_iden]), np.mean(d_iden[key_iden])]
        outf_iden.writerow(out_iden)

for key_sim in d_sim.keys():
    if len(d_sim[key_sim]) > 0:
        out_sim = [key_sim, np.max(d_sim[key_sim]), np.min(d_sim[key_sim]), np.mean(d_sim[key_sim])]
        outf_sim.writerow(out_sim)
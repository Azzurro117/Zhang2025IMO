#! /usr/bin/python

import os
from scipy.stats import ks_2samp

inc_h12 = [0.4, 0.4, 0.5, 0.5, 0.6, 0.6, 0.3, 0.4, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8, 1, 1, 0.3, 0.8, 0.2, 0.9]
inc_h9 = [0.5,0.8,0.2,0.3,0.3,0.3,0.3,0.4,0.5,0.5,0.6,0.6,0.7,0.8,1,1,0.2,0.2,0.2,0.2,0.2,0.25,0.25,0.3,0.3]
inc_h18 = [0.1, 0.3, 0.5, 0.6, 1, 1, 0.5, 0.7, 0.8]
mor_h12 = [0.03,0.1,0.002,0.05,0,0.04,0.01,0.01,0.04,0.002,0.03,0.1,0.015,0.07,0,0.052,0.08,0.06,0.03,0.25]
mor_h9 = [0,0.04,0.05,0.001,0.015,0.02,0.03,0.002,0.034,0.038,0.018,0.08,0.01,0.035,0.045,0.035,0.05,0.05,0.03,0.04,0.04,0.05,0.05,0.05,0.05]
mor_h18 = [0.005,0.001,0.01,0.018,0.01,0.002,0.03,0.1,0.015]

result_inc_h12_h9 = ks_2samp(inc_h12, inc_h9)
result_inc_h12_h18 = ks_2samp(inc_h12, inc_h18)
result_inc_h9_h18 = ks_2samp(inc_h9, inc_h18)
print(result_inc_h12_h9[1])
print(result_inc_h12_h18[1])
print(result_inc_h9_h18[1])

result_mor_h12_h9 = ks_2samp(mor_h12, mor_h9)
result_mor_h12_h18 = ks_2samp(mor_h12, mor_h18)
result_mor_h9_h18 = ks_2samp(mor_h9, mor_h18)
print(result_mor_h12_h9[1])
print(result_mor_h12_h18[1])
print(result_mor_h9_h18[1])


inc_nn = [0.2,0.8,0.6,0.7]
inc_ns = [0.4,0.4,0.5,0.5,0.6,0.5,0.8,0.1,0.3,0.5,0.6,1,1]
inc_ss = [0.3,0.4,0.4,0.5,0.6,0.7,0.8,0.8,1,1]
mor_nn = [0.05,0.06,0.15,0.01]
mor_ns = [0.03,0.1,0.002,0.05,0,0,0.04,0.005,0.001,0.01,0.018,0.01,0.002]
mor_ss = [0.01,0.01,0.04,0.002,0.03,0.1,0.015,0.07,0,0.052]

result_inc_nn_ns = ks_2samp(inc_nn, inc_ns)
result_inc_nn_ss = ks_2samp(inc_nn, inc_ss)
result_inc_ns_ss = ks_2samp(inc_ns, inc_ss)
print(result_inc_nn_ns[1])
print(result_inc_nn_ss[1])
print(result_inc_ns_ss[1])


result_mor_nn_ns = ks_2samp(mor_nn, mor_ns)
result_mor_nn_ss = ks_2samp(mor_nn, mor_ss)
result_mor_ns_ss = ks_2samp(mor_ns, mor_ss)
print(result_mor_nn_ns[1])
print(result_mor_nn_ss[1])
print(result_mor_ns_ss[1])

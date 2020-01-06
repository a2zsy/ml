# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 15:38:43 2020

@author: 小源源
"""
import pandas as pd


col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]
data_transform_1 = pd.read_table(r"E:\study\ml\work\data\nsl_kdd\KDDTrain+.txt",header=None, sep=',',names = col_names)
print(data_transform_1.head(10))
data_transform_1.to_csv(r"E:\study\ml\work\data\nsl_kdd\KDDTrain+.csv")    

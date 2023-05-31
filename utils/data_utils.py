import wfdb     #导入wfdb包读取数据文件
from IPython.display import display
import numpy as np
import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
from torch import nn

AAMI=['N','V','/','L','R']

#rootdir = 'data/european-st-t-database-1.0.0'
rootdir = 'data/mit-bih-arrhythmia-database-1.0.0'
#rootdir = 'data/mit-bih-st-change-database-1.0.0'
#rootdir = 'data/sudden-cardiac-death-holter-database-1.0.0'

class EcgDataset(Dataset):
    def __init__(self, file, labels):
        self.labels = labels
        self.label_map = {label: i for i, label in enumerate(labels)}
        self.data = pd.read_csv(file)
        super().__init__()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data.iloc[index, 0:len(self.data.iloc[0,:])-1]
        y = self.data.iloc[index, len(self.data.iloc[0,:])-1]
        x = torch.tensor(x,dtype=torch.float)
        y_one_hot = nn.functional.one_hot(torch.tensor(self.label_map[y]), len(self.labels))
        return x.unsqueeze(0), y_one_hot

def get_namelist(rootdir):
    files = os.listdir(rootdir)
    name_list=[]            # name_list
    for file in files:
        if file[-3:] == 'atr':     # 选取文件的前五个字符（可以根据数据文件的命名特征进行修改）
            name_list.append(file[0:-4])
    return name_list

def get_namelist_with_lead(rootdir, lead_type):
    name_list = get_namelist(rootdir)
    namelist_with_lead = []
    for name in name_list:      # 遍历每一个人
        record = wfdb.rdrecord(rootdir+'/'+name)  # 读取一条记录（100），不用加扩展名
        if lead_type in record.sig_name:       # 记录MLII导联的数据
            namelist_with_lead.append(name)
    return namelist_with_lead

def count_type(rootdir, lead_type):
    namelist_with_lead = get_namelist_with_lead(rootdir, lead_type = lead_type)
    type={}
    for name in namelist_with_lead:      # 遍历每一个人
        annotation = wfdb.rdann(rootdir+'/'+name, 'atr')  # 读取一条记录的atr文件，扩展名atr
        for symbol in annotation.symbol:            # 记录下这个人所有的标记类型
            if symbol in list(type.keys()):
                type[symbol]+=1
            else:
                type[symbol]=1
        print('sympbol_name',type)
    sorted(type.items(),key=lambda d:d[1],reverse=True)
    return type

def get_segmented_signal(rootdir, lead_type = 'MLII', window_size = 360, heartbeat_types = AAMI):
    namelist_with_lead = get_namelist_with_lead(rootdir, lead_type = lead_type)
    heartbeat_segments = []
    heartbeat_labels = []
    heartbeat_counter = {t: 0 for t in heartbeat_types}
    record = wfdb.rdrecord(rootdir + '/' + namelist_with_lead[0])  # Load a specific record
    lead_index = record.sig_name.index(lead_type)  # 找到MLII导联对应的索引

    for person in namelist_with_lead:
        record = wfdb.rdrecord(rootdir + '/' + person, channels=[lead_index])  # Load a specific record 
        annotation = wfdb.rdann(rootdir + '/' + person, 'atr')  # Load the corresponding annotation file
        labels = annotation.symbol
        for i, label in enumerate(labels):
            if label in heartbeat_types:
                start = annotation.sample[i] - window_size // 2
                end = annotation.sample[i] + window_size // 2
                if start >= 0 and end < len(record.p_signal):
                    segment = record.p_signal[start:end].flatten()
                    heartbeat_counter[label] += 1
                    heartbeat_segments.append(segment)
                    heartbeat_labels.append(label)
    total = 0
    for heartbeat_type, count in heartbeat_counter.items():
        total += count
        print(f"heartbeat_type: {heartbeat_type}: {count}")
    print(f"Toal: {total}")
    return heartbeat_segments, heartbeat_labels

def get_segmented_signal_with_sample(rootdir, lead_type = 'MLII', window_size = 360, heartbeat_types = AAMI):
    namelist_with_lead = get_namelist_with_lead(rootdir, lead_type = lead_type)
    heartbeat_segments = []
    heartbeat_labels = []
    heartbeat_counter = {t: 0 for t in heartbeat_types}
    record = wfdb.rdrecord(rootdir + '/' + namelist_with_lead[0])  # Load a specific record
    lead_index = record.sig_name.index(lead_type)  # 找到MLII导联对应的索引

    for person in namelist_with_lead:
        record = wfdb.rdrecord(rootdir + '/' + person, channels=[lead_index])  # Load a specific record 
        annotation = wfdb.rdann(rootdir + '/' + person, 'atr')  # Load the corresponding annotation file
        labels = annotation.symbol
        for i, label in enumerate(labels):
            if label in heartbeat_types:
                if random.random() > 1/5: continue
                if label=='N':
                    if random.random() > 1/7: continue
                start = annotation.sample[i] - window_size // 2
                end = annotation.sample[i] + window_size // 2
                if start >= 0 and end < len(record.p_signal):
                    segment = record.p_signal[start:end].flatten()
                    heartbeat_counter[label] += 1
                    heartbeat_segments.append(segment)
                    heartbeat_labels.append(label)
    total = 0
    for heartbeat_type, count in heartbeat_counter.items():
        total += count
        print(f"heartbeat_type: {heartbeat_type}: {count}")
    print(f"Toal: {total}")
    return heartbeat_segments, heartbeat_labels

def save_dataset(heartbeat_segments, heartbeat_labels, test_size = 0.25):
    print('begin to save dataset!')
    X_train, X_test, Y_train, Y_test = train_test_split(heartbeat_segments, heartbeat_labels, test_size=test_size, shuffle=True)
    train_df = pd.DataFrame(X_train)
    train_df['Y'] = Y_train
    train_df.to_csv('train.csv', index=False)

    test_df = pd.DataFrame(X_test)
    test_df['Y'] = Y_test
    test_df.to_csv('test.csv', index=False)
    print('Finished!')

# heartbeat_segments, heartbeat_labels = get_segmented_signal_with_sample(rootdir, lead_type = 'MLII', window_size = 360, heartbeat_types = AAMI)
# save_dataset(heartbeat_segments, heartbeat_labels, test_size = 0.25)
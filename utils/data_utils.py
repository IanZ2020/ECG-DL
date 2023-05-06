import wfdb     #导入wfdb包读取数据文件
from IPython.display import display
import numpy as np
import pandas as pd
import os
import scipy
from scipy import signal
from sklearn.model_selection import train_test_split

AAMI=['N','V','L','R','A','F','S']

rootdir = 'data/european-st-t-database-1.0.0'
#rootdir = 'data/mit-bih-arrhythmia-database-1.0.0'
# rootdir = 'data/mit-bih-st-change-database-1.0.0'
# rootdir = 'data/sudden-cardiac-death-holter-database-1.0.0'        #心脏性猝死数据库

files = os.listdir(rootdir) #列出文件夹下所有文件名
name_list=[]            # name_list
MLII=[]                 # 用MLII型导联采集的人（根据选择的不同导联方式会有变换）
type={}                 # 标记及其数量

for file in files:
    if file[-3:] == 'atr':     # 选取文件的前五个字符（可以根据数据文件的命名特征进行修改）
        name_list.append(file[0:-4])

for name in name_list:      # 遍历每一个人
    record = wfdb.rdrecord(rootdir+'/'+name)  # 读取一条记录（100），不用加扩展名
    if 'MLIII' in record.sig_name:       # 记录MLII导联的数据
        MLII.append(name)               # 记录下这个人
    annotation = wfdb.rdann(rootdir+'/'+name, 'atr')  # 读取一条记录的atr文件，扩展名atr
    for symbol in annotation.symbol:            # 记录下这个人所有的标记类型
        if symbol in list(type.keys()):
            type[symbol]+=1
        else:
            type[symbol]=1
    print('sympbol_name',type)
sorted(type.items(),key=lambda d:d[1],reverse=True)

record = wfdb.rdrecord(rootdir+'/'+name_list[0])
f=record.fs      # 数据库的原始采样频率
segmented_len=10        # 将数据片段裁剪为10s

label_count=0
count=0
abnormal=0

segmented_data = []             # 最后数据集中的X
segmented_label = []            # 最后数据集中的Y
print('begin!')

for person in MLII:        # 读取导联方式为MLII的数据
    k = 0
    whole_signal=wfdb.rdrecord(rootdir + '/' + person).p_signal.transpose()     # 这个人的一整条数据
    lenth = len(whole_signal[0])
    while (k+1)*f*segmented_len<=lenth:    # 只要不到最后一组数据点
        record = wfdb.rdrecord(rootdir + '/' + person, sampfrom=k * f * segmented_len,sampto=(k + 1) * f * segmented_len)  # 读取一条记录（100），不用加扩展名
        annotation = wfdb.rdann(rootdir + '/' + person, 'atr', sampfrom=k * f * segmented_len,sampto=(k + 1) * f * segmented_len)  # 读取一条记录的atr文件，扩展名atr
        lead_index = record.sig_name.index('MLIII')  # 找到MLII导联对应的索引
        signal = record.p_signal.transpose()  # 两个导联，转置之后方便画图
        # segmented_data.append(signal[lead_index])   # 只记录MLII导联的数据段
        symbols=annotation.symbol


        re_signal = scipy.signal.resample(signal[lead_index], 3600)  # 采样
        re_signal_3 = np.round(re_signal, 3)
        if len(symbols) == 0:
            k+=1
            continue
        elif symbols.count('N') == len(symbols) or symbols.count('N') + symbols.count('/') == len(symbols):  # 如果全是'N'或'/'和'N'的组合，就标记为N
            label = 'N'
        else:
            labels = []
            for i in symbols:
                if i != 'N' and i != '/':
                    labels.append(i)
            label = max(set(labels), key=labels.count)
        
        if label not in AAMI:
            k+=1
            continue
        if(label != 'N'):
            abnormal += 1
        
        count+=1
        print('resignal', re_signal_3)
        print('symbols', symbols, len(symbols))
        segmented_data.append(re_signal_3)
        segmented_label.append(label)
        print(label + ' No. '+ str(count) +' No. of abnormal cases= '+ str(abnormal))
        k+=1
print('begin to save dataset!')

X_train, X_test, Y_train, Y_test = train_test_split(segmented_data, segmented_label, test_size=0.25, shuffle=True)
train_df = pd.DataFrame(X_train)
train_df['Y'] = Y_train
train_df.to_csv('train.csv', index=False)

test_df = pd.DataFrame(X_test)
test_df['Y'] = Y_test
test_df.to_csv('test.csv', index=False)
print('Finished!')

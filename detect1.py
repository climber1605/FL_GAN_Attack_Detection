import numpy as np
import math
import pickle as pkl
from pathlib import Path
import matplotlib.pyplot as plt

np.random.seed(0)
clients_per_round = 10

def search_suspects(data):
    suspects = []
    window_size = 6
    step = 1
    f_threshold = 14.94
    #t_threshold = 4.144

    #window_size = 4
    #step = 1
    #f_threshold = 9.28
    #t_threshold = 4.144

    #window_size = 5
    #step = 1
    #f_threshold = 23.15
    #t_threshold = 3.833
    for i in range(len(data)):
        window1 = data[i][:window_size]
        var1 = np.var(window1, ddof=1)
        avg1 = np.mean(window1)
        for j in range(step, len(data[i]) - window_size, step):
            window2 = data[i][j: j + window_size]
            var2 = np.var(window2, ddof=1)
            if var1 < var2:
                f = var2 / var1
            else:
                f = var1 / var2
            avg2 = np.mean(window2)
            t = abs((avg1 - avg2) / (math.sqrt((sum((x - avg1) ** 2 for x in window1) + sum((x - avg2) ** 2 for x in window2)) / (2 * window_size - 2) * (2 / window_size))))
            # if i == 7:
            #     print(i, j, f, t)
            #if f > f_threshold and t > t_threshold:
            if f > f_threshold:
                suspects.append(i)
                print(i, j, var1, var2, f)
                break
            var1 = var2
            avg1 = avg2
            window1 = window2

    # if len(suspects):
    #     print('Found {} suspects'.format(len(suspects)))
    #     print(suspects)
    # else:
    #     print('Not found')
    return suspects

def search_suspects1(data):
    # time series outlier detection based on sliding window prediction
    k = 10
    t = 2.539
    k = 6
    #t = 2.571
    #t = 2.861
    t = 10
    suspects = []

    for l in range(clients_per_round):
        predicted = []
        low = []
        up = []
        for i in range(len(data[l])):
            s = w = 0
            minj, maxj = max(0, i - k), min(len(data[l]), i + k + 1)
            for j in range(minj, i):
                s += (j - minj + 1) * data[l][j]
                w += j - minj + 1
            for j in range(i + 1, maxj):
                s += (maxj - j) * data[l][j]
                w += maxj - j
            s /= w
            predicted.append(s)
            temp = np.array(data[l][minj: maxj])
            r = t * math.sqrt(np.var(temp)) * math.sqrt(1 + 1 / (2 * k))
            #print(s,r,s+r,s-r)
            up.append(s + r)
            low.append(s - r)

        plt.cla()
        plt.plot(data[l], 'b', label='real')
        plt.plot(predicted, 'r', label='predicted')
        plt.plot(up, 'b--', label='predicted upper')
        plt.plot(low, 'r--', label='predicted lower')
        plt.legend()
        #plt.show()
        plt.close()

        for i in range(len(data[l])):
            if not low[i] <= data[l][i] <= up[i]:
                #print('Found outlier')
                #print(l, i, low[i], data[l][i], up[i])
                suspects.append(l)
                break
    return suspects

def search_suspects2(data):
    suspects = []
    grow_threshold = 10
    y_threshold = 1e-3
    for l in range(clients_per_round):
        for i in range(1, len(data[l])):
            if data[l][i] > y_threshold and data[l][i] > abs(data[l][i - 1]) * grow_threshold:
                print('Found outlier')
                print(l, i, data[l][i], data[l][i - 1], data[l][i] / data[l][i - 1])
                suspects.append(l)
                break
    return suspects
    
def search_suspects3(data):
    suspects = []
    num_points = 6
    #epsilon = 0.025
    epsilon = 0.015

    # num_points = 10
    # epsilon = 0.03
    
    for l in range(clients_per_round):
        for i in range(len(data[l]) - num_points):
            all_larger = True
            for k in range(i + 1, min(i + num_points + 1, len(data[l]))):
                if data[l][k] <= data[l][i] + epsilon:
                    all_larger = False 
                    break
            if all_larger:
                print('Found outlier')
                print(l, i, data[l][i:i+num_points+1])
                suspects.append(l)
                break
    return suspects
     
def search_suspects4(data):
    suspects = set()
  
    # best hyper-parameters for MNIST
    num_points = 3
    threshold = 1.5
    
    # best hyper-parameters for GTSRB
    # num_points = 4
    # threshold = 2
    
    data = np.array(data)
    cnt = [0] * clients_per_round
    for i in range(len(data[0])):
        for l in range(clients_per_round):
            temp = np.concatenate((data[:l, i], data[l+1:, i]))
            avg = temp.mean()
            #if l == 0:
            #    print(l, i, data[l][i], avg, data[l][i] / avg)
            if data[l][i] > threshold * avg:
                cnt[l] += 1
                if cnt[l] >= num_points:
                    #print('Found outlier in search_suspects4')
                    #print(l, i, data[l][i], avg, data[l][i] / avg)
                    suspects.add(l)
                    break
            else:
                cnt[l] = 0
    return list(suspects)

def search_suspects5(data):
    suspects = []
    num_points = 5
    threshold = 1.2
    
    for l in range(clients_per_round):
        for i in range(len(data[l]) - num_points):
            all_larger = True
            for k in range(i + 1, min(i + num_points + 1, len(data[l]))):
                if data[l][k] <= data[l][i] * threshold:
                    all_larger = False 
                    break
            if all_larger:
                print('Found outlier')
                print(l, i, data[l][i:i+num_points+1])
                suspects.append(l)
                break
    return suspects
    

import scipy as sp   
from scipy.optimize import leastsq  

def func(p,x):
    k, b = p
    return k * x + b
 
def error(p, x, y):
    return func(p,x) - y
    
def search_suspects6(data):
    suspects = set()
    
    # best hyper-parameters for MNIST
    # window_size = 9
    # step = 4
    # grow_threshold = 100
    # k_threshold = 0.005

    # best hyper-parameters for GTSRB
    window_size = 9
    step = 4
    grow_threshold = 100
    k_threshold = 0.005
    
    max_slope = [float('-inf')] * clients_per_round
    data = np.array(data)
    
    for i in range(0, len(data[0]) - window_size + 1, step):
        for l in range(clients_per_round):
            para = leastsq(error, [1, 0], args=(np.array((range(window_size))), data[l][i: i + window_size]))
            max_slope[l] = max(max_slope[l], para[0][0])
        
        for l in range(clients_per_round):
            others = np.array([max_slope[j] for j in range(len(max_slope)) if j != l])
            avg = others.mean()
            #y_max = data[l][i : i + window_size].max()
            #if max_slope[l] > k_threshold and max_slope[l] > grow_threshold * other_max and y_max < y_threshold:
            if max_slope[l] > k_threshold and max_slope[l] > grow_threshold * avg:
                #print('Found outlier in search_suspects6')
                #print(l, i, data[l][i:i+window_size], para, max_slope[l], other_max)
                #for k in range(clients_per_round):
                #    print(k, i, data[k][:i+window_size], max_slope[k])
                suspects.add(l)
                break
    return list(suspects)

# def search_suspects6(data):
    # data = np.array(data)
    # suspects = set()
    
    # # best hyper-parameters for MNIST
    # # window_size = 5
    # # step = 2
    # # grow_threshold = 100
    # # k_threshold = 0.002

    # # best hyper-parameters for GTSRB
    # window_size = 10
    # step = 5
    # grow_threshold = 100
    # k_threshold = 0.005

    # for i in range(0, len(data[0]) - window_size + 1, step):
        # slopes = [float('-inf')] * clients_per_round
        # for l in range(clients_per_round):
            # para = leastsq(error, [1, 0], args=(np.array((range(window_size))), data[l][i: i + window_size]))
            # slopes[l] = para[0][0]
        
        # for l in range(clients_per_round):
            # others = np.array([slopes[j] for j in range(len(slopes)) if j != l])
            # avg = others.mean()
            # #y_max = data[l][i : i + window_size].max()
            # #if max_slope[l] > k_threshold and max_slope[l] > grow_threshold * other_max and y_max < y_threshold:
            # if slopes[l] > k_threshold and slopes[l] > grow_threshold * avg:
                # #print('Found outlier in search_suspects6')
                # #print(l, i, data[l][i:i+window_size], para, max_slope[l], other_max)
                # #for k in range(clients_per_round):
                # #    print(k, i, data[k][:i+window_size], max_slope[k])
                # suspects.add(l)
                # break
    # return list(suspects)

colors = ['r--', 'g--', 'b--', 'c--', 'm--', 'r', 'g', 'b', 'c', 'm']
with_attack = True
global_acc = global_fpr = global_fnr = 0
total = 0
clients = [2]
#clients = list(range(0, 1))
dirname = r'd:\temp\MNIST'
target_labels = [5, 4, 0, 6, 2, 8, 7, 9, 1, 3] if 'MNIST' in dirname else [12, 16, 1, 21, 19, 34, 27, 40, 23, 7]
print(target_labels)
TP = TN = FP = FN = 0

for malicious_client in clients:
    target_label = target_labels[malicious_client]
    target_label = 0
    for trial in range(0, 1):
        data1 = []
        filename = r'd:\temp\MNIST\weights_distance_with_attack_{}_{}_128_{}.pkl'.format(malicious_client, target_label, trial)
        #filename = r'd:\temp\GTSRB\Manhattan\weights_distance_without_attack_128_4.pkl'
        if not Path(filename).is_file():
            print('{} not exists'.format(filename))
            continue
        with open(filename, 'rb') as text:
            data1 = pkl.load(text)
        data = np.array(data1)    
        
        for i in range(clients_per_round):
            plt.plot(data[i], colors[i], label='client {}'.format(i))
        plt.title('parameter updates(Malicious client: {}, target label: {})'.format(malicious_client, target_label))
        plt.xlabel('round')
        plt.ylabel('parameter change rate')
        plt.legend()
        pic_name = 'weights_distance_with_attack_{}_{}_128_{}.png'.format(malicious_client, target_label, trial)
        #plt.savefig(dirname + '\\' + pic_name)
        plt.show()
        plt.close()

        suspects = search_suspects4(data)
        if not len(suspects):
            #pass
            suspects = search_suspects6(data)
        print('trial #{}, found {} suspects: {}, actual suspect: {}\n'.format(trial, len(suspects), ' '.join(map(str, suspects)), malicious_client))

        if malicious_client in suspects:
            TP += 1
            if len(suspects) != 1:
                FP += 1
        #if len(suspects) != 1 or suspects[0] != malicious_client:
        #    FP += 1
        
        total += 1

if total > 0:
    error = FP / total
    recall = TP / total
    print('total positive samples: {}, recall: {}, error rate: {}\n'.format(total, recall, error))
exit(0)

#trial_set = {10, 16, 37,40,52,63,65,76,78}
trial_set = set()

for trial in range(0, 100):
    data1 = []
    filename = r'd:\temp\GTSRB0\weights_distance_without_attack_128_{}.pkl'.format(trial)
    if not Path(filename).is_file():
        print('{} not exists'.format(filename))
        continue
    with open(filename, 'rb') as text:
        data1 = pkl.load(text)
    data = np.array(data1)    
    
    for i in range(clients_per_round):
        plt.plot(data[i], colors[i], label='client {}'.format(i))
    
    plt.title('parameter updates(No malicious participant)')
    plt.xlabel('round')
    plt.ylabel('parameter change rate')
    plt.legend()
    filename = 'weights_distance_without_attack_128_{}.png'.format(trial)
    plt.savefig(dirname + '\\' + filename)

    suspects = search_suspects4(data)
    if not len(suspects):
        #pass
        suspects = search_suspects6(data)        
    print('trial #{}, found {} suspects: {}\n'.format(trial, len(suspects), ' '.join(map(str, suspects))))

    if False and len(suspects):
        plt.show()
    plt.close()

    if len(suspects) > 0:
        FP += 1
    total += 1

fpr = FP / total
print('total negative samples: {}, false positive rate: {}\n'.format(total, fpr))







import os
from typing import Any
import pickle
import torch
import pandas as pd
import numpy as np
import csv
corruption = []
# DomainNum=np.zeros(16)
# Combine=None
def saveIBNMeanMaxAsCSV(conf, bn_idx, mean_diff_l2_mean, mean_diff_l2_max, acc):
    grad_dir = os.path.join(
            "ttab/exp_result",
            'class-cosINBNcsv',
            )
    if not os.path.exists(grad_dir):
        os.makedirs(grad_dir)
    headers = [
                "step",
                "domain",
                "bn_idx",
                "InstanceNorm",
                "BatchNorm",
                "cross_entropy",
                "acc",
               ]
    global corruption
    print("！！！！！domain：",corruption)
    dict={
            "step":conf.step,
            "domain":corruption,
            "bn_idx":bn_idx,
            "InstanceNorm":mean_diff_l2_mean,
            "BatchNorm":mean_diff_l2_max,
            "cross_entropy":acc["cross_entropy"],
            "acc":acc["accuracy_top1"],
    }
    # corruption = []
    csv_file = os.path.join(grad_dir,f"6_{conf.base_data_name}_{conf.inter_domain}.csv")
  
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)  
            writer.writeheader()
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writerow(dict)


def saveMeanVarAsCSV(conf,idx, mean,var):
    grad_dir = os.path.join(
            "/data/TTA-exp/ttab-w/TTA-exp/data/logs",
            '1_MeanVarcsv',
            )
    if not os.path.exists(grad_dir):
        os.makedirs(grad_dir)
    
    headers = [
                "idx",
                "mean",
                "var",
               ]
    dict={
            "idx":idx,
            "mean":mean,
            "var":var,
    }
    csv_file = os.path.join(grad_dir,f"{conf.base_data_name}_{conf.inter_domain}.csv")
  
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)  
            writer.writeheader()
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writerow(dict)
        

def saveAsCSV(conf: Any , state , batch):
    conf.grad_dir = os.path.join(
            # 可修改为其他路径
            # "E:/GITHUB/TTA-Exp/ttab/data/step_loss_grad",
            # conf.model_name,
            # conf.job_name,
            # f"{conf.step}_{conf.model_name}_{conf.base_data_name}_{conf.model_adaptation_method}_{conf.model_selection_method}"
            conf.root_path,
            conf.model_name,
            conf.base_data_name,
            'csvData',
            # f"{conf.model_name}_{conf.base_data_name}_{conf.model_adaptation_method}_{conf.model_selection_method}_{int(conf.timestamp if conf.timestamp is not None else time.time())}-seed{conf.seed}",
            # f"{conf.model_name}_{conf.base_data_name}_{conf.model_adaptation_method}_{conf.model_selection_method}_{str(time.time()).replace('.', '_')}-seed{conf.seed}",
            # f"{conf.job_id}_{conf.model_name}_{conf.base_data_name}_{conf.model_adaptation_method}_{conf.inter_domain}_{conf.corruption_num}",
            # f"{conf.job_id}_{conf.model_name}_{conf.base_data_name}_{conf.model_adaptation_method}_{conf.inter_domain}",
            )
    if not os.path.exists(conf.grad_dir):
        os.makedirs(conf.grad_dir)
    print(conf)
    if state["grads"] is not None:
        L2=getL2(state["grads"])
    else:
        L2=0
    Accuracy_Top1=accuracy_top1(
        target=batch._y, 
        output=state["yhat"]
        )
    Cross_Entropy=cross_entropy(    
        output=state["yhat"], 
        target=batch._y
        )
    # add
    # global DomainNum,Combines
    headers = ["step", 
               "loss", 
               "L2", 
               "Accuracy_Top1", 
               "Cross_Entropy", 
               "model_adaptation_method", 
               "dataset", 
               "batch_size", 
               #"data_names",
               "inter_domain",
               "model_name",
               "Combine"
               ]
    dict={
             "step":conf.step,
             "loss":state["loss"],
             "L2":L2,
             "Accuracy_Top1":Accuracy_Top1,
             "Cross_Entropy":Cross_Entropy,
             "model_adaptation_method":conf.model_adaptation_method,
             "dataset":conf.base_data_name,
             "batch_size":conf.batch_size,
             #"data_names":conf.data_names,
             "inter_domain":conf.inter_domain,
             "model_name":conf.model_name,
             "Combine":Combine,
        }
    corruptions = conf.data_names.split(';')
    #dict.update({"domain_length":corruptions.size()})
    #names=[]
    i=1
    for corr in corruptions:
        name = corr.split('-')[-2]
        headers.append(name)
        dict.update({f'{name}':DomainNum[i]})
        #dict.append(corr)
        #dict[corr]=DomainNum[i]
        i=i+1
        #names.append(corr)   
    # add
    #+ list(temp["grads"].keys())
    DomainNum = np.zeros(16)
    Combine = None
    csv_file = os.path.join(conf.grad_dir, f"{conf.job_id}_{conf.model_name}_{conf.base_data_name}_{conf.model_adaptation_method}_{conf.inter_domain}_{conf.corruption_num}.csv")
  
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)  
            writer.writeheader()
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writerow(dict)
        
def save(conf: Any,state):
    pass
        # conf.grad_dir = os.path.join(
        #     # #可修改为其他路径
        #     # "E:/GITHUB/TTA-Exp/ttab/data/step_loss_grad",
        #     # conf.model_name,
        #     # conf.job_name,
        #     # f"{conf.step}_{conf.model_name}_{conf.base_data_name}_{conf.model_adaptation_method}_{conf.model_selection_method}"
        #     conf.root_path,
        #     conf.model_name,
        #     conf.base_data_name,
        #     'loss_grad',
        #     # f"{conf.model_name}_{conf.base_data_name}_{conf.model_adaptation_method}_{conf.model_selection_method}_{int(conf.timestamp if conf.timestamp is not None else time.time())}-seed{conf.seed}",
        #     # f"{conf.model_name}_{conf.base_data_name}_{conf.model_adaptation_method}_{conf.model_selection_method}_{str(time.time()).replace('.', '_')}-seed{conf.seed}",
        #     f"{conf.job_id}_{conf.model_name}_{conf.base_data_name}_{conf.model_adaptation_method}_{conf.inter_domain}_{conf.corruption_num}",
        #     # f"{conf.job_id}_{conf.model_name}_{conf.base_data_name}_{conf.model_adaptation_method}_{conf.inter_domain}",
        #     )
        # if not os.path.exists(conf.grad_dir):
        #     os.makedirs(conf.grad_dir)
        # temp=state
        # loss = state["loss"]
        # # if conf.step in None:
        # #      print('conf.step is none')
        # # elif temp['loss'] is None:
        # #      print('state loss is none')
        # # elif temp['grads'] is None:
        # #      print('state grads is none')
        # # print("the loss is ", loss)
        # # print(conf)
        # dict={
        #      "step":conf.step,
        #      "loss":temp["loss"],
        #      **temp["grads"]
        #       }
        # pickle_file = os.path.join(conf.grad_dir, f'{conf.step}_save.pickle')
        # # print(f"file path:{pickle_file}")
        # with open(pickle_file, 'ab') as f:
        #     pickle.dump(dict, f)


def getL2(data):
    weight=[]
    for key in data.keys():
        if 'weight'in key:
            weight.append(data[key].cpu().numpy()[0])
    combined_data = np.concatenate(weight, axis=None)
    #L2范式（保存每一步的全部梯度的L2范式）
    l2_weight = np.linalg.norm(combined_data, ord=2)
    return l2_weight

# ACC
def _accuracy(target, output, topk):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)

    _, pred = output.topk(topk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_k = correct[:topk].reshape(-1).float().sum(0, keepdim=True)
    return correct_k.mul_(100.0 / batch_size).item()

def accuracy_top1(target, output, topk=1):
    """Computes the precision@k for the specified values of k"""
    return _accuracy(target, output, topk)

# CrossEntropy
cross_entropy_loss = torch.nn.CrossEntropyLoss()

def cross_entropy(target, output):
    """Cross entropy loss"""
    return cross_entropy_loss(output, target).item()


# def bn_hook(module, inputs, outputs):
#     input = inputs[0]
#     bn_dims = [0, 2, 3]  # 计算 batch 和空间维度上的均值和方差
#     in_dims = [2, 3]  # 计算空间维度上的均值和方差,对每个样本分别计算

#     bn_mean = input.mean(bn_dims, keepdim=True)
#     bn_var = input.var(bn_dims, keepdim=True, unbiased=False)

#     in_mean = input.mean(in_dims, keepdim=True)
#     in_var = input.var(in_dims, keepdim=True, unbiased=False)

#     # 计算 BN 与 IN 均值的差值
#     mean_diff = bn_mean - in_mean
#     # 计算 BN 与 IN 方差的差值
#     var_diff = bn_var - in_var

#     # 计算差值的 L2 范数
#     mean_diff_l2 = mean_diff.pow(2).sum(dim=[1, 2, 3]).pow(0.5)  # 对所有维度求和,再开平方根
#     var_diff_l2 = var_diff.pow(2).sum(dim=[1, 2, 3]).pow(0.5)

#     # 计算 L2 范数的均值和最大值
#     mean_diff_l2_mean = mean_diff_l2.mean()
#     mean_diff_l2_max = mean_diff_l2.max()
#     # var_diff_l2_mean = var_diff_l2.mean()
#     # var_diff_l2_max = var_diff_l2.max()

#     print(f"BN 与 IN 均值差值 L2 范数的均值: {mean_diff_l2_mean.item()}")
#     print(f"BN 与 IN 均值差值 L2 范数的最大值: {mean_diff_l2_max.item()}")
#     # print(f"BN 与 IN 方差差值 L2 范数的均值: {var_diff_l2_mean.item()}")
#     # print(f"BN 与 IN 方差差值 L2 范数的最大值: {var_diff_l2_max.item()}")
#     layer_idx = len(bn_stats)
#     bn_stats[layer_idx] = (mean_diff_l2_mean, mean_diff_l2_max)
    # print(bn_stats[layer_idx])
# def bn_hook(module, inputs, outputs):
#     input = inputs[0]ass
#     dims = [dim for dim in range(input.dim())]  # if dim != 1
#     mean = input.mean(dims, keepdim=True)
#     # print(f"Layer: {module.__class__.__name__}, Mean shape: {mean.size()}")
#     var = input.var(dims, keepdim=True, unbiased=False)
#     layer_name = f"{module.__class__.__name__}_{len(bn_stats)}"
#     bn_stats[layer_name] = (mean, var)
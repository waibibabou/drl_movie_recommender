import os
import torch
import torch.nn as nn
import numpy as np
from model import PCT_Seg
from repc_reader import readAllData
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def split_data(points):
    np_points = np.array(points)
    points_num = np_points.shape[0]
    start_x = np_points[0,1]
    end_x = np_points[points_num-1,1]
    diff_x = abs(end_x-start_x)
    k = int(diff_x // 10)
    part_num = int(points_num // k)
    start_index = 0
    section_points = []
    for i in range(int(k)):
        if i == k-1:
            part_points = np_points[start_index:,:]
            section_points.append(part_points)
        else:
            part_points = np_points[start_index:start_index+part_num, :]
            section_points.append(part_points)
            start_index += part_num
    return section_points



def split_data(points):
    points_num = len(points)
    left_part_points = []
    right_part_points = []
    section_points = []
    start_x = points[0][1]
    end_x = points[points_num-1][1]
    for i in range(points_num):
        point_id = points[i][0]
        x = points[i][1]
        y = points[i][2]
        z = points[i][3]
        if(abs(x-start_x)>=10 and abs(end_x-x)>5):
            section_points.append(left_part_points.copy())
            section_points.append(right_part_points.copy())
            left_part_points.clear()
            right_part_points.clear()
            start_x = points[i][1]
        elif(z>-20 and z< 20 and y >-2):
            if(z<0):
                left_part_points.append([point_id,x,abs(z),y])
            else:
                right_part_points.append([point_id,x,z,y])

    if len(left_part_points) > 0:
        section_points.append(left_part_points.copy())
        left_part_points.clear()
    if len(right_part_points) > 0:
        section_points.append(right_part_points.copy())
        right_part_points.clear()
    return section_points


def handle_data(points,num_sample):
    points_num = points.shape[0]
    points_index = np.random.permutation(points_num)
    random_points = points[points_index]
    if points_num>=num_sample:
        k = points_num//num_sample
        sample = []
        scope = torch.zeros((k+1,2))
        for i in range(k):
            data = random_points[i*num_sample:(i+1)*num_sample]
            # data = centralization(data)
            sample.append(data)
            scope[i,0] = 0
            scope[i,1] = num_sample
        data = random_points[points_num-num_sample:]
        data_clone = data.copy()
        # data = centralization(data)
        sample.append(data_clone)
        scope[k,0] = (k+1)*num_sample-points_num
        scope[k,1] = num_sample


        return sample,scope
    else:
        dup_index = np.random.choice(points_num, num_sample - points_num)
        dup_points = random_points[dup_index, ...]
        scope = torch.zeros((1,2))
        scope[0,0] = 0
        scope[0,1] = points_num
        sample = []
        sample.append(np.concatenate([points, dup_points], 0))
        return sample,scope

def handleAllData(all_points,num_sample):
    section_num = len(all_points)
    all_sample = []
    all_scope = []
    for i in range(section_num):
        points = np.array(all_points[i])
        sample,scope = handle_data(points, num_sample)
        all_sample.extend(sample)
        all_scope.extend(scope)

    # print('1111:',len(all_sample))
    # print('2222:',len(all_scope))
    return all_sample,all_scope

def centralization(points):
    points[:,0] -= np.mean(points[:,0])
#     points[:,1] /= np.std(points[:,1])
    return points

class MyDataset(Dataset):
    def __init__(self,all_points,num_sample):
        all_sample,all_scope = handleAllData(all_points,num_sample)
        self.all_sample = all_sample
        self.all_scope = all_scope
    def __getitem__(self, item):
        data = self.all_sample[item]
        data = centralization(data)
        scope = self.all_scope[item]
        return data, scope
    def __len__(self):
        return len(self.all_sample)

def save_result(root,num,data,pred):
    if os.path.exists(root):
        os.remove(root)

    handle = open(root,'a')
    for i in range(num):
        string = str(i) + ' ' + str(data[i,0]) + ' ' + str(data[i,1]) + ' ' + str(data[i,2]) + ' ' + str(pred[i]) +'\n'
        handle.write(string)
    handle.close()

class Model():
    def __init__(self,num_sample=8192,part_num=7,batch_size=4,device_id=0):
        print('模型初始化...\n')
        print('样本点数：',num_sample)
        print('样本类别数：', part_num)
        print('batch_size：', batch_size)
        print('设备序号：',device_id)
        self.num_sample = num_sample
        self.part_num = part_num
        self.pth_file = 'checkpoints/my_model.pth'
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.k = 0
        self.batch_size = batch_size
        self.device_id = device_id
        self.load_model()
    def load_model(self):
        print('开始加载模型...\n')
        model = PCT_Seg(part_num=self.part_num)
        print(str(model))
        model.load_state_dict(torch.load(self.pth_file))
        self.model = model

        if torch.cuda.device_count() > 1:
             self.model = nn.DataParallel(model)
        #    if self.device_id == 0:
        #        self.model = nn.DataParallel(model)
        #    else:
        #        self.model = nn.DataParallel(model,device_ids=self.device_id)
        self.model.to(self.device)
        self.model.eval()
        print('模型加载成功\n')
    def predict(self,all_points):
        print(len(all_points))
        print('batch_size',self.batch_size)
        print('开始预测...\n')
        if(self.model==None):
            self.load_model()
        model = self.model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        # print("3333")
        # model.to(device)
        # print("4444")
        # model.eval()
        # print("5555")

        my_data_loader = DataLoader(MyDataset(all_points=all_points, num_sample=self.num_sample),
                                    self.batch_size, shuffle=False, drop_last=False, pin_memory=True)
        all_result_list = []
        with torch.no_grad():
            for batch_id,(data,scope) in tqdm(enumerate(my_data_loader),total=len(my_data_loader)):
                data = data[:,:,1:].float()
                try:
                    data = data.to(device)
                except RuntimeError as e:
                    print(str(e))
                prediction = model(data)
                data = data.cpu()
                pred = prediction.max(dim=1)[1]
                # print(pred.shape)
                # print(pred)
                pred = pred.cpu()


                torch.cuda.empty_cache()
                for i in range(pred.shape[0]):
                    ##### 保存结果
                    root = os.path.abspath('result/%d.txt' % (self.k))
                    save_result(root, pred[i].shape[0], torch.squeeze(data[i]).numpy(), torch.squeeze(pred[i]).numpy())
                    # pcd_root = os.path.abspath('../only_result/pcd/%d.pcd' % (self.k))
                    # save_result_pcd(pcd_root, torch.squeeze(data[i]).numpy(), torch.squeeze(pred[i]).numpy())
                    self.k += 1
                    print(self.k)
                    ######

                    result = pred[i,int(scope[i,0]):int(scope[i,1])]
                    #result = changeResult(result)
                    all_result_list.append(result)
        all_result = torch.cat(all_result_list,dim=0)
        print(all_result.shape)
        return all_result.numpy().tolist()

def test(repc_root):
    points = readAllData(repc_root)
    section_points = split_data(points)
    my_model = Model()
    my_model.predict(section_points)

# test()
# model = PCT_Seg(7)
# print(str(model))
# model.load_state_dict(torch.load('checkpoints/my_model.pth'))
# model.to('cuda')
#
# test_data = torch.randn((1,100,3))
# print(test_data)
# test_data = test_data.to('cuda')
# prediction = model(test_data)
# print(prediction.shape)
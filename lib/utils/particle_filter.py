import os
import logging
import time
import torch
import numpy as np

class ParticleFilter():

    def __init__(self, cfg):

        self.filename = cfg.UUID.replace('-','_')
        self.filepath = self.filename + '.npy'
        self.points = 2000
        # control affect between previous particle and new predicted result
        self.prev_particle = 20000
        self.block_num = 8
        self.map = np.zeros(self.block_num)
        # initialize 
        # Randomly generate a bunch of particles
        for i in range(int(self.points)):
            self.map[np.random.randint(self.block_num, size=1)] += 1/self.points

        # save the map
        np.save(self.filename, self.map)

    def run(self, model_output):
        # model_output : a tensor save the probability of each block (e.g., 8 blocks)
        # shape : (batch, blocks)
        final_output = []
        model_output = model_output.tolist()
        weight_map = np.load(self.filepath)
        # update weights of each block 
        for _ , blocks in enumerate(model_output):
            # adjust the importance of previous particles
            weight_map = weight_map * self.prev_particle
            #print(blocks)
            for i, w in enumerate(blocks):
                # update by new estimated result
                # weight_map[i] += w * self.points
                # i-1 and i+1
                # magic number 0.7 and 0.5
                weight_map[i] +=  w * self.points 

                # first spread
                if i >= 1:
                    weight_map[i-1] += w * self.points * 0.7
                if i+1 < len(blocks):
                    weight_map[i+1] += w * self.points * 0.7
                # second spread
                if i >= 2:
                    weight_map[i-2] += w * self.points * 0.7 * 0.5
                if i+2 < len(blocks):
                    weight_map[i+2] += w * self.points * 0.7 * 0.5
                # third spread
                if i >= 3:
                    weight_map[i-3] += w * self.points * 0.7 * 0.5 * 0.5
                if i+3 < len(blocks):
                    weight_map[i+3] += w * self.points * 0.7 * 0.5 * 0.5

                # old version
                # weight_map[int(i-1)%len(blocks)] += w * self.points * 0.7
                # weight_map[int(i+1)%len(blocks)] += w * self.points * 0.7
                # weight_map[int(i-2)%len(blocks)] += w * self.points * 0.7 * 0.5
                # weight_map[int(i+2)%len(blocks)] += w * self.points * 0.7 * 0.5


            # print(blocks)
            #print(weight_map)
            # normalization
            _, topk = torch.topk(torch.tensor(weight_map), 8, dim=-1)
            #print(topk)
            weight_map = self.resample(weight_map)

            final_output.append(topk)
        final_output = torch.stack(final_output, dim=0)
        # save map
        np.save(self.filename, weight_map)

        return final_output


    def resample(self, map):
        # doing normalization
        weight_sum = map.sum(axis=0)
        return map/weight_sum

class ParticleFilter_v2():

    def __init__(self, cfg):

        self.filename = cfg.UUID.replace('-','_')
        self.filepath = self.filename + '.npy'
        # 總粒子數
        self.points = 500
        # RP數量
        self.block_num = 8
        self.map = np.zeros((self.points,2)) # 權重, x座標
        # 初始化 
        # 隨機產生粒子，權重一致
        for p in self.map:
            p[0] = 1/self.points
            p[1] = np.random.uniform(0, self.block_num-1)

        # save the map
        np.save(self.filename, self.map)

    def run(self, model_output):
        # model_output : a tensor save the probability of each block (e.g., 8 blocks)
        # shape : (batch, blocks)
        final_output = []
        model_output = model_output.tolist()

        self.map = np.load(self.filepath)

        for batch in model_output:
            # 更新狀態
            mu, sigma = 0 , 1.4  # 平均值和標準差
            self.update_positon(mu, sigma)
            # 更新權重
            self.update_weights(batch)
            # 重新產生particles
            self.map = self.resample(self.map)
            # 求出最終位置
            final_pos = torch.tensor([self.estimate_position()])
            final_output.append(final_pos)
        final_output = torch.stack(final_output, dim=0)

        # save map
        np.save(self.filename, self.map)
        return final_output

    def update_positon(self, mu, sigma):
        # 更新每個粒子的位置(隨機移動)
        s = np.random.normal(mu, sigma, self.points)  
        self.map[:,1] += s        

    def update_weights(self, prob_list):
        _ , topk = torch.topk(torch.tensor(prob_list), 1, dim=-1)
        for pos in topk:
        # 根據粒子與初步預測位置的距離誤差(高斯分布)來調整權重
            self.map[:,0] += np.exp(-(self.map[:,1] - pos.item())**2 / 2)*prob_list[pos.item()]

    def estimate_position(self):
        final_pos = np.sum(self.map[:,0]*self.map[:,1])/np.sum(self.map[:,0])
        final_pos = 0 if final_pos < 0 else final_pos
        final_pos = 7 if final_pos > 7 else final_pos
        return round(final_pos)

    def resample(self, map):            
        # 正規化權重
        map[:,0] /= np.sum(map[:,0])
        # 重新灑點
        indices = np.random.choice(self.points, self.points, p=map[:,0])
        # 每個particle權重都一致
        map[:,0] = 1/self.points
        map[:,1] = map[indices,1]
        return map
    
if __name__ == "__main__":
    import sys 
    sys.path.append("..") 
    from config import cfg

    x = torch.tensor([[0.4, 0.3, 0.2, 0.1, 0, 0, 0, 0], [0, 0.2, 0.2, 0.3, 0.3, 0, 0, 0]])
    pf = ParticleFilter_v2(cfg)
    pf.run(x)

    

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.cluster import KMeans

device = "cuda:1"


class Memory(nn.Module):
    def __init__(self, memory_size, feature_dim, key_dim, temp_update, temp_gather):
        super(Memory, self).__init__()
        # Constants
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.key_dim = key_dim
        self.temp_update = temp_update
        self.temp_gather = temp_gather
        self.momentum = 0.95

    def forward(self, query, keys, train=True, Defect=False, label=None, epoch=0, defect_memory=None):
        query = [F.normalize(q, dim=1) for q in query]
        query = [q.permute(0, 2, 3, 1) for q in query]

        # train
        if train:
            separateness_loss, compactness_loss = 0, 0
            for i in range(len(query)):
                s_loss, c_loss, keys, defect_memory = self.gather_loss(query[i], train, Defect, label[i], keys, epoch,
                                                                        defect_memory)
                separateness_loss += s_loss
                compactness_loss += c_loss
            updated_query = self.read(query[0])
            return updated_query, separateness_loss, compactness_loss, keys, defect_memory

        # test
        else:
            hotmaps = []
            for i in range(len(query)):
                hotmap = self.gather_loss(query[i], False, Defect, None, keys, epoch, None)
                hotmaps.append(hotmap)
            return hotmaps[0]

    def gather_loss(self, query, train, Defect, labels, keys, epoch, defect_bank):
        # query: feature map(b,h,w,d)--(2*64*64*512); keys: memory bank(20 * 512);
        batch_size, h, w, dims = query.size()  # b X h X w X d
        if train:
            loss = torch.nn.TripletMarginLoss(margin=0.8)  # default: 0.8
            loss_d = torch.nn.TripletMarginLoss(margin=1.0)  # judge the defect
            loss_mse = torch.nn.MSELoss()
            query_reshape = query.contiguous().view(batch_size * h * w, dims)
            '''if the query is not a background sample, it exists the defect area'''
            if Defect:
                gathering_loss = 0
                compact_Loss = 0
                labels = labels[:, 0, :, :]
                labels = labels.unsqueeze(-1)
                label_bs, label_h, label_w, label_dims = labels.size()
                label_reshape = labels.contiguous().view(label_bs * label_h * label_w, label_dims)
                pos_index = []
                neg_index = []
                for i in range(label_reshape.shape[0]):
                    if label_reshape[i] == True:
                        neg_index += [i]
                    elif label_reshape[i] == False:
                        pos_index += [i]
                pos = query_reshape[pos_index, :].to(device)
                neg = query_reshape[neg_index, :].to(device)
                bg_kmeans = KMeans(n_clusters=10, random_state=0).fit(pos.detach().cpu().numpy())
                bg_Centers = torch.tensor(bg_kmeans.cluster_centers_).to(device)

                if keys.shape[0] == 0:
                    keys = bg_Centers
                if epoch < 1:
                    keys = torch.cat((keys, bg_Centers), dim=0)

                if epoch >= 1:
                    keys, defect_bank = self.memory_items_operation(keys, bg_Centers, Defect, neg, defect_bank)

                for j in range(pos.shape[0]):
                    gathering_loss += loss_d(pos[[j], :],
                                             bg_Centers[[bg_kmeans.labels_[j]], :],
                                             neg.to(device))
                for k in range(neg.size()[0]):
                    centralPoint = torch.tensor(bg_Centers[[bg_kmeans.labels_[k]], :]).to(device)
                    compact_Loss += loss_mse(pos[[k], :], centralPoint)

                b = math.log(epoch + 2, 2)
                return b * gathering_loss, compact_Loss, keys, defect_bank
            # to cluster all background's the query
            kmeans = KMeans(n_clusters=10, random_state=0).fit(query_reshape.detach().cpu().numpy())
            compactLoss = 0
            center = torch.tensor(kmeans.cluster_centers_).to(device)
            # get the most relevant query with center
            gathering_loss = 0
            for i in range(query_reshape.shape[0]):
                compactLoss += loss_mse(query_reshape[[i], :], center[[kmeans.labels_[i]], :])
            # if keys exited, make key items close to background centers

            if epoch >= 1:
                temp_defect_memory = torch.empty(0, 512).to(device)
                keys, defect_bank = self.memory_items_operation(keys, center, Defect, None, temp_defect_memory)
            if keys.shape[0] == 0:
                keys = center
            if epoch < 1:
                keys = torch.cat((keys, center), dim=0)
            query_with_center = ((query_reshape.unsqueeze(1) - center.unsqueeze(0)) ** 2).mean(dim=2)
            _, indexOf_near_keys = torch.topk(query_with_center, 2, dim=1, largest=False)
            for i in range(query_reshape.shape[0]):
                gathering_loss += loss(query_reshape[[i], :],
                                       center[[indexOf_near_keys[i, 0]], :],
                                       center[[indexOf_near_keys[i, 1]], :])

            if defect_bank.shape[0] > 0:
                dis_of_query_defect = ((query_reshape.unsqueeze(1) - defect_bank.unsqueeze(0)) ** 2).mean(dim=2)
                _, indexOf_near_defect = torch.topk(dis_of_query_defect, 1, dim=1, largest=False)
                for i in range(query_reshape.shape[0]):
                    gathering_loss += loss(query_reshape[[i], :],
                                           center[[kmeans.labels_[i]], :],
                                           defect_bank[[indexOf_near_defect[i, 0]], :])
            return gathering_loss, compactLoss, keys, defect_bank

        else:
            query_reshape = query.contiguous().view(batch_size * h * w, dims)
            query_dis_keys = ((query_reshape.unsqueeze(1) - keys[0].unsqueeze(0)) ** 2).mean(dim=2)
            _, gathering_indices = torch.topk(query_dis_keys, 1, dim=1, largest=False)
            # threshold is defined as: 2, 4
            gathering_loss = torch.sum(torch.pow(query_reshape - keys[0][gathering_indices, :].squeeze(1).detach(), 4),
                                       dim=1)
            return gathering_loss.view(batch_size, h, w, 1)

    def read(self, query):  # , updated_memory updated_memory == keys
        batch_size, h, w, dims = query.size()  # b X h X w X d
        query_reshape = query.contiguous().view(batch_size * h * w, dims)

        updated_query = query_reshape.view(batch_size, h, w, 1 * dims)
        updated_query = updated_query.permute(0, 3, 1, 2)

        return updated_query

    def memory_items_operation(self, keys, bg_Centers, Defect, neg, defect_memory):
        mse_loss = torch.nn.MSELoss()
        if Defect:
            defect_memory = torch.cat((defect_memory, neg), dim=0)
            print("after add neg items, defectBank shape is : ", defect_memory.shape)

        if defect_memory.shape[0] > 0:
            new_items = torch.empty((0, 512)).to(device)
            keys_with_defectBank = ((keys.unsqueeze(1) - defect_memory.unsqueeze(0)) ** 2).mean(dim=2)
            _, indexOf_keys_nearDefect = torch.topk(keys_with_defectBank, 1, dim=0, largest=False)
            less01_keys_nearDefect = torch.zeros((1, 1)).to(device)

            # to delete near defect normal feature vector
            for i in range(indexOf_keys_nearDefect.shape[1]):
                if torch.sqrt(mse_loss(keys[indexOf_keys_nearDefect[0, i], :], defect_memory[i, :])) < 0.01:
                    less01_keys_nearDefect = torch.cat((less01_keys_nearDefect, indexOf_keys_nearDefect[:, [i]]), dim=1)
            less01_keys_nearDefect = less01_keys_nearDefect[:, 1:]
            print("need to dropout keys item: ", less01_keys_nearDefect.shape[1])
            if less01_keys_nearDefect.shape[1] != 0:
                keys = self.mask_Tensors(keys, less01_keys_nearDefect)

            dropout_centers = torch.zeros((1, 512)).to(device)
            for i in range(bg_Centers.shape[0]):
                for j in range(defect_memory.shape[0]):
                    if torch.sqrt(mse_loss(bg_Centers[[i], :], defect_memory[[j], :])) < 0.01:
                        dropout_centers = torch.cat((dropout_centers, bg_Centers[[i], :]), dim=0)
                        break
            dropout_centers = dropout_centers[1:, :]
            print("need to dropout centers: ", dropout_centers.shape[0])

            flag_neg = 0
            for i in range(bg_Centers.shape[0]):
                for j in range(dropout_centers.shape[0]):
                    if torch.equal(bg_Centers[[i], :], dropout_centers[[j], :]):
                        flag_neg = 1
                        break
                if flag_neg == 0:
                    new_items = torch.cat((new_items, bg_Centers[[i], :]))
                flag_neg = 0
            print("import new items: ", new_items.shape[0])
            keys = torch.cat((keys, new_items[0:, :]), dim=0)

            if keys.shape[0] > 150:
                keys = keys[50:]
            return keys, defect_memory
        # append new normal samples center
        if keys.shape[0] > 150:
            keys = keys[50:]
        keys = torch.cat((keys, bg_Centers), dim=0)
        return keys, defect_memory

    def mask_Tensors(self, keys, mask_index):
        mask = torch.ones(keys.size())
        new_keys = torch.zeros((1, 512)).to(device)
        mask_index = mask_index.int()
        for i in range(mask_index.shape[1]):
            mask[[mask_index[0, [i]]], :] = 0
        for i in range(keys.shape[0]):
            if mask[i, 0] == 1:
                new_keys = torch.cat((new_keys, keys[[i], :]), dim=0)
        new_keys = new_keys[1:, :]
        return new_keys

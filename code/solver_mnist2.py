from __future__ import print_function
import os, sys
sys.path.append('./C')
from scipy.spatial.distance import cdist
import numpy as np
from sklearn.metrics import euclidean_distances
import os
import pickle
import torch
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model.build_gen import *
from datasets.dataset_read import dataset_read
from einops import rearrange
from math import log,exp
import torch.nn.functional as F
# from show.show import plotlabels
from mmdloss import MMDLoss
# The solver for training and testing LtC-MSDA
class Solver(object):
    def __init__(self, args, batch_size=128,
                 target='mnistm', learning_rate=0.0001, interval=10, optimizer='adam',
                 checkpoint_dir=None, save_epoch=10):
        self.batch_size = batch_size
        self.target = target
        self.pretrain = False
        self.checkpoint_dir = checkpoint_dir
        self.save_epoch = save_epoch
        self.interval = interval
        self.lr = learning_rate
        self.best_correct = 0
        self.args = args
        self.mean_t_flag = False
        self.nclasses =self.args.nclasses
        self.nfeat = self.args.nfeat
        self.alpha = 10
        self.t = 1
        self.t1 = 100
        self.save_picture = False
        self.r =0.5
        if self.args.use_target:
            self.ndomain = self.args.ndomain
        else:
            self.ndomain = self.args.ndomain - 1
        
        # load source and target domains
        self.datasets, self.dataset_test, self.dataset_size = dataset_read(target, self.batch_size)
        
        self.niter = self.dataset_size / self.batch_size
        print('Dataset loaded!')

        # define the feature extractor and C-based classifier
        self.G = Generator(self.args.net)
        self.C = Classifier('resnet50',feat=args.nfeat, nclass=args.nclasses)#nfeat=2048 nclass=10
        self.G.cuda()
        self.C.cuda()
        if self.pretrain == True:
            self.state = torch.load(os.path.join('/SSD/xzf/msda/prototype/pretrain_model', 'best_model_mnistm.pth'))
            self.G.load_state_dict(self.state['G'])  
            self.C.load_state_dict(self.state['C'])
        print('Model initialized!')

        if self.args.load_checkpoint is not None:
            self.state = torch.load(self.args.load_checkpoint)
            self.G.load_state_dict(self.state['G'])
            self.C.load_state_dict(self.state['C'])
            print('Model load from: ', self.args.load_checkpoint)

        # initialize statistics (prototypes and adjacency matrix)
        if self.args.load_checkpoint is None:
            self.mean = torch.zeros(args.nclasses * (self.ndomain-1), args.nfeat).cuda()
            self.mean_t = torch.zeros(args.nclasses , args.nfeat).cuda()
            # self.adj = torch.zeros(args.nclasses * self.ndomain, args.nclasses * self.ndomain).cuda()
            print('Statistics initialized!')
        else:
            self.mean = self.state['mean'].cuda()
            self.adj = self.state['adj'].cuda()
            print('Statistics loaded!')

        # define the optimizer
        self.set_optimizer(which_opt=optimizer, lr=self.lr)
        print('Optimizer defined!')
    def a(self,outputs,label,batch_size):
        pred = outputs.max(1)[1].cpu()
                
        correct = pred.eq(label).cpu().sum()/batch_size
        return correct
    # optimizer definition
    def set_optimizer(self, which_opt='sgd', lr=0.001, momentum=0.9):
        if which_opt == 'sgd':
            self.opt_g = optim.SGD(self.G.parameters(),
                                   lr=lr, weight_decay=0.0005,
                                   momentum=momentum)
            self.opt_C = optim.SGD(self.C.parameters(),
                                     lr=lr, weight_decay=0.0005,
                                     momentum=momentum)
        elif which_opt == 'adam':
            self.opt_g = optim.Adam(self.G.parameters(),
                                    lr=lr, weight_decay=0.0005)
            self.opt_C = optim.Adam(self.C.parameters(),
                                      lr=lr, weight_decay=0.0005)

    # empty gradients
    def reset_grad(self):
        self.opt_g.zero_grad()
        self.opt_C.zero_grad()

    # compute the discrepancy between two probabilities
    def discrepancy(self, out1, out2):
        return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))

    # compute the Euclidean distance between two tensors
    def euclid_dist(self, x, y):
        x_sq = (x ** 2).mean(-1)
        x_sq_ = torch.stack([x_sq] * y.size(0), dim = 1)
        y_sq = (y ** 2).mean(-1)
        y_sq_ = torch.stack([y_sq] * x.size(0), dim = 0)
        xy = torch.mm(x, y.t()) / x.size(-1)
        dist = x_sq_ + y_sq_ - 2 * xy

        return dist

    # construct the extended adjacency matrix
    def construct_adj(self, feats):
        dist = self.euclid_dist(self.mean, feats)
        # self.mean是5*10,2048
        sim = torch.exp(-dist / (2 * self.args.sigma ** 2))
        E = torch.eye(feats.shape[0]).float().cuda()

        A = torch.cat([self.adj, sim], dim = 1)
        B = torch.cat([sim.t(), E], dim = 1)
        C_adj = torch.cat([A, B], dim = 0)

        return C_adj

    # assign pseudo labels to target samples
    # def pseudo_label(self, logit, feat):
    #     pred = F.softmax(logit, dim=1)
    #     entropy = (-pred * torch.log(pred)).sum(-1)
    #     label = torch.argmax(logit, dim=-1).long()

    #     mask = (entropy < self.args.entropy_thr).float()
    #     index = torch.nonzero(mask).squeeze(-1)
    #     feat_ = torch.index_select(feat, 0, index)
    #     label_ = torch.index_select(label, 0, index)

    #     return feat_, label_
    def w(self):#[nclass,ndomain,nfeat]
        w = list()
        m = rearrange(self.mean,'(b a) c -> a b c',b =self.ndomain-1,a=self.nclasses)
        for i in range(self.nclasses):
            w.append((self.mean_t[i]@torch.pinverse(m[i])).unsqueeze(0))
        w = torch.cat(w,dim=0)    
        return w    
    # update prototypes and adjacency matrix
    def update_statistics(self, feats, labels, epsilon=1e-5):
        curr_mean = list()
        num_labels = 0

        for domain_idx in range(self.ndomain-1):
            tmp_feat = feats[domain_idx]
            tmp_label = labels[domain_idx]
            num_labels += tmp_label.shape[0]

            if tmp_label.shape[0] == 0:
                curr_mean.append(torch.zeros((self.args.nclasses, self.args.nfeat)).cuda())
            else:
                onehot_label = torch.zeros((tmp_label.shape[0], self.args.nclasses)).scatter_(1, tmp_label.unsqueeze(
                    -1).cpu(), 1).float().cuda()#在属于的类别上标1,[1024,10]
                domain_feature = tmp_feat.unsqueeze(1) * onehot_label.unsqueeze(-1)#[1024,1,2048]*[1024,10,1]=[1024,10,2048]
                tmp_mean = domain_feature.sum(0) / (onehot_label.unsqueeze(-1).sum(0) + epsilon)#[10,[10,1]

                curr_mean.append(tmp_mean)
        mask = (self.mean.sum(-1) != 0).float().unsqueeze(-1)
        curr_mean = torch.cat(curr_mean, dim = 0)
        curr_mask = (curr_mean.sum(-1) != 0).float().unsqueeze(-1)#防止出现全0特征，导致原来原型×0.7，后续+0×0.3
        self.mean = self.mean.detach() * (1 - curr_mask) + (self.mean.detach() * self.args.beta + curr_mean * (1 - self.args.beta)) * curr_mask +curr_mean * self.args.beta * (1-mask)
        curr_dist = self.euclid_dist(self.mean, self.mean)
        self.adj = torch.exp(-curr_dist / (2 * self.args.sigma ** 2))

        # compute local relation alignment loss
        loss_local = ((((curr_mean - self.mean) * curr_mask) ** 2).mean(-1)).sum() / num_labels

        return loss_local
    def update_statistics_t(self, feats, labels, epsilon=1e-5):
        curr_mean = list()
        num_labels = 0

        for domain_idx in range(1):
            tmp_feat = feats[domain_idx]
            tmp_label = labels[domain_idx]
            num_labels += tmp_label.shape[0]

            if tmp_label.shape[0] == 0:
                curr_mean.append(torch.zeros((self.args.nclasses, self.args.nfeat)).cuda())
            else:
                onehot_label = torch.zeros((tmp_label.shape[0], self.args.nclasses)).scatter_(1, tmp_label.unsqueeze(
                    -1).cpu(), 1).float().cuda()#在属于的类别上标1,[1024,10]
                domain_feature = tmp_feat.unsqueeze(1) * onehot_label.unsqueeze(-1)#[1024,1,2048]*[1024,10,1]=[1024,10,2048]
                tmp_mean = domain_feature.sum(0) / (onehot_label.unsqueeze(-1).sum(0) + epsilon)#[10,[10,1]

                curr_mean.append(tmp_mean)

        curr_mean = torch.cat(curr_mean, dim = 0)
        curr_mask = (curr_mean.sum(-1) != 0).float().unsqueeze(-1)#防止出现全0特征，导致原来原型×0.7，后续+0×0.3
        mask = (self.mean_t.sum(-1) != 0).float().unsqueeze(-1)
        self.mean_t = self.mean_t.detach() * (1 - curr_mask) + (self.mean_t.detach() * self.args.beta + curr_mean * (1 - self.args.beta)) * curr_mask+ curr_mean * self.args.beta * (1-mask)
        curr_dist = self.euclid_dist(self.mean_t, self.mean_t)
        self.adj = torch.exp(-curr_dist / (2 * self.args.sigma ** 2))

        # compute local relation alignment loss
        loss_local = ((((curr_mean - self.mean_t) * curr_mask) ** 2).mean(-1)).sum() / num_labels

        return loss_local
        # return loss_local

    # compute global relation alignment loss
    def adj_loss(self):
        adj_loss = 0

        for i in range(self.ndomain):
            for j in range(self.ndomain):
                adj_ii = self.adj[i * self.args.nclasses:(i + 1) * self.args.nclasses,
                         i * self.args.nclasses:(i + 1) * self.args.nclasses]
                adj_jj = self.adj[j * self.args.nclasses:(j + 1) * self.args.nclasses,
                         j * self.args.nclasses:(j + 1) * self.args.nclasses]
                adj_ij = self.adj[i * self.args.nclasses:(i + 1) * self.args.nclasses,
                         j * self.args.nclasses:(j + 1) * self.args.nclasses]

                adj_loss += ((adj_ii - adj_jj) ** 2).mean()
                adj_loss += ((adj_ij - adj_ii) ** 2).mean()
                adj_loss += ((adj_ij - adj_jj) ** 2).mean()

        adj_loss /= (self.ndomain * (self.ndomain - 1) / 2 * 3)

        return adj_loss

    def sim(self,c_zu,feat):
        # f = feat.unsqueeze(1).repeat([1,nclass,1])
        d = torch.div(torch.mul(c_zu,feat).sum(-1),torch.sqrt(torch.mul((c_zu**2).sum(-1),(feat**2).sum(-1)))+1e-8)
        return d
    def dis(self,c_zu,f,nclass):
        f = f.unsqueeze(1).repeat([1,nclass,1])
        d = torch.sum((c_zu-f)**2,dim=-1)
        return d
    # per epoch training in a Domain Generalization setting
    
    def g_loss(self,zuhe):
        d = torch.cdist(zuhe,self.mean_t)**2
        d = torch.exp(d/self.t1)
        index = torch.argmin(d,dim=1)
        mask = F.one_hot(index,self.nclasses)
        min_d =(d*mask).sum(-1)
        return torch.div(min_d,(d.sum(-1))).mean()
  
    def d(self,c_zu,f):
        # f = f.unsqueeze(1).repeat([1,nclass,1])
        c_zu_d = c_zu/((c_zu**2).sum(-1).unsqueeze(-1).repeat([1,1,1,self.nfeat]))
        f_d = f/((f**2).sum(-1).unsqueeze(-1).repeat([1,1,1,self.nfeat]))
        d = torch.sqrt(torch.sum((c_zu_d-f_d)**2,dim=-1))
        d = torch.sqrt(torch.sum((c_zu-f)**2,dim=-1))
        return d
    def psedo_labels(self,feat_t,logit,tc):
        pred = F.softmax(logit, dim=1)
        mask=(pred.max(1)[0]>tc).float()
        label = torch.argmax(logit, dim=-1).long()
        index = torch.nonzero(mask).squeeze(-1)
        feat_ = torch.index_select(feat_t, 0, index)
        label_ = torch.index_select(label, 0, index)
        index_p = torch.nonzero(1-mask).squeeze(-1)
        feat_p = torch.index_select(feat_t, 0, index_p)
        label_p = torch.index_select(pred, 0, index_p)
        return feat_, label_, feat_p
    def train_C_baseline(self, epoch, record_file=None):
        
        criterion = nn.CrossEntropyLoss().cuda()
        self.G.train()
        self.C.train()

        for batch_idx, data in enumerate(self.datasets):
            # get the source batches
            img_s = list()
            label_s = list()
            stop_iter = False
            for domain_idx in range(self.ndomain - 1):
                tmp_img = data['S' + str(domain_idx + 1)].cuda()
                tmp_label = data['S' + str(domain_idx + 1) + '_label'].long().cuda()
                img_s.append(tmp_img)
                label_s.append(tmp_label)

                if tmp_img.size()[0] < self.batch_size:
                    stop_iter = True

            if stop_iter:
                break

            # get the target batch
            img_t = data['T'].cuda()
            label_t = data['T_label']
            if img_t.size()[0] < self.batch_size:
                break

            self.reset_grad()

            # get feature embeddings
            feat_list = list()
            for domain_idx in range(self.ndomain - 1):
                tmp_img = img_s[domain_idx]
                t_clone=tmp_img.data
                t_clone.requires_grad=True
                tmp_feat = self.G(tmp_img)
                feat_list.append(tmp_feat)
            i_clone=img_t.data
            i_clone.requires_grad=True
            feat_t = self.G(i_clone)
            local_src = self.update_statistics(feat_list, label_s)
            # feat_list.append(feat_t)
            src_domain_label = torch.cat([torch.arange(self.args.nclasses)] * (self.ndomain-1), dim=0).long().cuda()
            feats = torch.cat(feat_list, dim=0)#4个源域+1个目标域叠在一起，每1024个为一个域，2048是特征维度，feats的torch.Size([5120, 2048])
            labels = torch.cat(label_s, dim=0)#4个源域的数据标签4096
            # if self.save_picture ==True:
            #     my_dict = {'S': feats, 'T': feat_t}
            #     with open(r'show/mnistm_only'+ str(batch_idx)+'.pkl', 'wb') as f:
            #         pickle.dump(my_dict, f)
            
            C_list = torch.cat([feats,feat_t,self.mean],dim = 0)
            C_logit = self.C(C_list)
            loss_cls_dom = criterion(C_logit[feats.shape[0]+feat_t.shape[0]:,:], src_domain_label)
            output = C_logit[:feats.shape[0], :]
            pred = output.max(1)[1]
            A = pred.eq(labels).sum()/feats.shape[0]
            tc = 1/(1 + torch.exp(-3*A))
            feat_t_, label_t_ ,feat_p= self.psedo_labels(feat_t,C_logit[feats.shape[0]:(feats.shape[0]+feat_t.shape[0])],tc)
            # _,_,_,feat_t_,label_t_,zu= self.psedo_labels(feat_t,self.batch_size)
            
            feat_t_list = list()
            label_t_list = list()
            feat_t_list.append(feat_t_)
            # feat_t_and_zu.append(zu)
            label_t_list.append(label_t_)
            # label_t_and_zu.append(label_t_)
            local_tgt = self.update_statistics_t(feat_t_list, label_t_list)
            loss_cls_src = criterion(C_logit[:feats.shape[0],:], labels)
            
            
            
            if not (self.mean_t_flag):
                if (True not in (self.mean_t.sum(1)==0)):
                    self.mean_t_flag = True
            if (not self.mean_t_flag):
                loss_cls =  loss_cls_src + loss_cls_dom 
               
                if (True not in (self.mean.sum(1)==0)):
                    loss = loss_cls_src + loss_cls_dom
                
                loss = loss_cls_src
            else:
                target_logit = C_logit[feats.shape[0]:feats.shape[0]+feat_t.shape[0],:]
                target_prob = F.softmax(target_logit, dim=1)
                loss_cls_tgt = (-target_prob * torch.log(target_prob + 1e-8)).mean()
                # loss_local = local_src + local_tgt
                m = torch.cat([self.mean,self.mean_t],dim=0)
                m1 = m.view([self.ndomain,self.nclasses,self.nfeat])
                # m = rearrange(m, 'a b c -> (a b) c')
                f = torch.cat([feats,feat_t_],dim=0)
                l = torch.cat([labels,label_t_],dim=0)

                weight = self.w()
                ms = rearrange(self.mean,'(b a) c -> a b c',b =self.ndomain-1,a=self.nclasses)
                zu_s = ((weight.unsqueeze(-1).repeat([1,1,self.nfeat])*ms).sum(1))
                L_s = -torch.log((torch.softmax(self.sim(feats.unsqueeze(1).repeat([1,self.nclasses,1]),zu_s.unsqueeze(0).repeat([feats.shape[0],1,1]))/self.t,dim=-1)*F.one_hot(labels,self.nclasses)).sum(-1)).mean()
                L_t = -torch.log((torch.softmax(self.sim(feat_t_.unsqueeze(1).repeat([1,self.nclasses,1]),self.mean_t.unsqueeze(0).repeat([feat_t_.shape[0],1,1]))/self.t,dim=-1)*F.one_hot(label_t_,self.nclasses)).sum(-1)).mean()
                L_t_s = -torch.log((torch.softmax(self.sim(feat_t_.unsqueeze(1).repeat([1,self.nclasses,1]),zu_s.unsqueeze(0).repeat([feat_t_.shape[0],1,1]))/self.t,dim=-1)*F.one_hot(label_t_,self.nclasses)).sum(-1)).mean()
                L_s_t = -torch.log((torch.softmax(self.sim(feats.unsqueeze(1).repeat([1,self.nclasses,1]),self.mean_t.unsqueeze(0).repeat([feats.shape[0],1,1]))/self.t,dim=-1)*F.one_hot(labels,self.nclasses)).sum(-1)).mean()
                
                L_p = -torch.log((torch.softmax(self.sim(zu_s.unsqueeze(1).repeat([1,self.nclasses,1]),self.mean_t.unsqueeze(0).repeat([zu_s.shape[0],1,1]))/self.t,dim=-1)*F.one_hot(torch.arange(self.args.nclasses).long().cuda(),self.nclasses)).sum(-1)).mean()
                
                
                tgt_dom_label = torch.arange(self.args.nclasses).long().cuda()
                tgt_logit = self.C(self.mean_t)
                loss_cls_dom_tgt = criterion(tgt_logit,tgt_dom_label)
                loss_combined = criterion(self.C(zu_s),tgt_dom_label)
                loss_cls = loss_cls_src  +(loss_cls_dom + loss_cls_dom_tgt + loss_combined)
                loss_relation =  L_s + L_t + L_t_s + L_s_t + L_p
                # loss_local = local_src + local_tgt
                # loss_relation = loss_local + loss_global
                
                
                mmdloss = MMDLoss()
                
                if feat_p.shape[0]!=0:
                    lambd = 2 / (1 + torch.exp(-10 * (torch.tensor([epoch])+1) / 200)) - 1 
                    loss = loss_cls + loss_relation+ (mmdloss(zu_s,feat_p))*lambd.item()
                      
                else:    
                    loss = loss_cls + loss_relation
                # assert not torch.isnan(loss_global).item()
            loss = loss_cls_src
            loss.backward(retain_graph = True)
            self.opt_C.step()
            self.opt_g.step()
            self.mean = self.mean.data
            self.mean.requires_grad=True
            self.mean_t = self.mean_t.data
            self.mean_t.requires_grad=True
            # record training information
            if epoch ==0 and batch_idx==0:
                record = open(record_file, 'a')
                record.write(str(self.args)+'\n')
                record.close()

            if (self.mean_t_flag):
                print(
                    'Train Epoch: {:>3} [{:>3}/{} ({:.2f}%)]\tLoss_cls_domain: {:.5f}\tLoss_cls_source: {:.5f}'
                    '\tLoss_cls_dom_tgt: {:.5f}\tLoss_combined: {:.5f}\tLoss_cls: {:.5f}\tloss_relation: {:.5f}\tloss_MMD: {:.5f}'.format(
                        epoch, batch_idx + 1, self.niter, (batch_idx + 1.) / self.niter * 100,
                        loss_cls_dom.item(), loss_cls_src.item(), loss_cls_dom_tgt.item(),
                        loss_combined.item(), loss_cls.item(), loss_relation.item(),(loss-loss_cls-loss_relation).item()))
                if record_file:
                    record = open(record_file, 'a')
                    record.write(
                        '\nTrain Epoch: {:>3} [{:>3}/{} ({:.2f}%)]\tLoss_cls_domain: {:.5f}\tLoss_cls_source: {:.5f}'
                        '\tLoss_cls_dom_tgt: {:.5f}\tLoss_combined: {:.5f}\tLoss_cls: {:.5f}\tloss_relation: {:.5f}\tloss_MMD: {:.5f}'.format(
                        epoch, batch_idx + 1, self.niter, (batch_idx + 1.) / self.niter * 100,
                        loss_cls_dom.item(), loss_cls_src.item(), loss_cls_dom_tgt.item(),
                        loss_combined.item(), loss_cls.item(), loss_relation.item(),(loss-loss_cls-loss_relation).item()))
                    record.close()
            else:
                print(
                    'Train Epoch: {:>3} [{:>3}/{} ({:.2f}%)]\tLoss_cls_domain: {:.5f}\tLoss_cls_source: {:.5f}'
                    ''.format(
                        epoch, batch_idx + 1, self.niter, (batch_idx + 1.) / self.niter * 100,
                        loss_cls_dom.item(), loss_cls_src.item()))
                if record_file:
                    record = open(record_file, 'a')
                    record.write(
                        '\nTrain Epoch: {:>3} [{:>3}/{} ({:.2f}%)]\tLoss_cls_domain: {:.5f}\tLoss_cls_source: {:.5f}'
                        ''.format(
                        epoch, batch_idx + 1, self.niter, (batch_idx + 1.) / self.niter * 100,
                        loss_cls_dom.item(), loss_cls_src.item()))
                    record.close()
        return batch_idx

    # per epoch training in a Multi-Source Domain Adaptation setting
    def train_C_adapt(self, epoch, record_file=None):
        
        criterion = nn.CrossEntropyLoss().cuda()
        self.G.train()
        self.C.train()

        for batch_idx, data in enumerate(self.datasets):
            # get the source batches
            img_s = list()
            label_s = list()
            stop_iter = False
            for domain_idx in range(self.ndomain - 1):
                tmp_img = data['S' + str(domain_idx + 1)].cuda()
                tmp_label = data['S' + str(domain_idx + 1) + '_label'].long().cuda()
                img_s.append(tmp_img)
                label_s.append(tmp_label)

                if tmp_img.size()[0] < self.batch_size:
                    stop_iter = True

            if stop_iter:
                break

            # get the target batch
            img_t = data['T'].cuda()
            label_t = data['T_label']
            if img_t.size()[0] < self.batch_size:
                break

            self.reset_grad()

            # get feature embeddings
            feat_list = list()
            for domain_idx in range(self.ndomain - 1):
                tmp_img = img_s[domain_idx]
                t_clone=tmp_img.data
                t_clone.requires_grad=True
                tmp_feat = self.G(tmp_img)
                feat_list.append(tmp_feat)
            i_clone=img_t.data
            i_clone.requires_grad=True
            feat_t = self.G(i_clone)
            local_src = self.update_statistics(feat_list, label_s)
            # feat_list.append(feat_t)
            src_domain_label = torch.cat([torch.arange(self.args.nclasses)] * (self.ndomain-1), dim=0).long().cuda()
            feats = torch.cat(feat_list, dim=0)#4个源域+1个目标域叠在一起，每1024个为一个域，2048是特征维度，feats的torch.Size([5120, 2048])
            labels = torch.cat(label_s, dim=0)#4个源域的数据标签4096
            # if self.save_picture ==True:
            #     my_dict = {'S': feats, 'T': feat_t}
            #     with open(r'show/mnistm_only'+ str(batch_idx)+'.pkl', 'wb') as f:
            #         pickle.dump(my_dict, f)
            
            C_list = torch.cat([feats,feat_t,self.mean],dim = 0)
            C_logit = self.C(C_list)
            loss_cls_dom = criterion(C_logit[feats.shape[0]+feat_t.shape[0]:,:], src_domain_label)
            output = C_logit[:feats.shape[0], :]
            pred = output.max(1)[1]
            A = pred.eq(labels).sum()/feats.shape[0]
            tc = 1/(1 + torch.exp(-3*A))
            feat_t_, label_t_ ,feat_p= self.psedo_labels(feat_t,C_logit[feats.shape[0]:(feats.shape[0]+feat_t.shape[0])],tc)
            # _,_,_,feat_t_,label_t_,zu= self.psedo_labels(feat_t,self.batch_size)
            
            feat_t_list = list()
            label_t_list = list()
            feat_t_list.append(feat_t_)
            # feat_t_and_zu.append(zu)
            label_t_list.append(label_t_)
            # label_t_and_zu.append(label_t_)
            local_tgt = self.update_statistics_t(feat_t_list, label_t_list)
            loss_cls_src = criterion(C_logit[:feats.shape[0],:], labels)
            
            
            
            if not (self.mean_t_flag):
                if (True not in (self.mean_t.sum(1)==0)):
                    self.mean_t_flag = True
            if (not self.mean_t_flag):
                loss_cls =  loss_cls_src + loss_cls_dom 
               
                if (True not in (self.mean.sum(1)==0)):
                    loss = loss_cls_src + loss_cls_dom
                
                loss = loss_cls_src
            else:
                target_logit = C_logit[feats.shape[0]:feats.shape[0]+feat_t.shape[0],:]
                target_prob = F.softmax(target_logit, dim=1)
                loss_cls_tgt = (-target_prob * torch.log(target_prob + 1e-8)).mean()
                # loss_local = local_src + local_tgt
                m = torch.cat([self.mean,self.mean_t],dim=0)
                m1 = m.view([self.ndomain,self.nclasses,self.nfeat])
                # m = rearrange(m, 'a b c -> (a b) c')
                f = torch.cat([feats,feat_t_],dim=0)
                l = torch.cat([labels,label_t_],dim=0)

                weight = self.w()
                ms = rearrange(self.mean,'(b a) c -> a b c',b =self.ndomain-1,a=self.nclasses)
                zu_s = ((weight.unsqueeze(-1).repeat([1,1,self.nfeat])*ms).sum(1))
                L_s = -torch.log((torch.softmax(self.sim(feats.unsqueeze(1).repeat([1,self.nclasses,1]),zu_s.unsqueeze(0).repeat([feats.shape[0],1,1]))/self.t,dim=-1)*F.one_hot(labels,self.nclasses)).sum(-1)).mean()
                L_t = -torch.log((torch.softmax(self.sim(feat_t_.unsqueeze(1).repeat([1,self.nclasses,1]),self.mean_t.unsqueeze(0).repeat([feat_t_.shape[0],1,1]))/self.t,dim=-1)*F.one_hot(label_t_,self.nclasses)).sum(-1)).mean()
                L_t_s = -torch.log((torch.softmax(self.sim(feat_t_.unsqueeze(1).repeat([1,self.nclasses,1]),zu_s.unsqueeze(0).repeat([feat_t_.shape[0],1,1]))/self.t,dim=-1)*F.one_hot(label_t_,self.nclasses)).sum(-1)).mean()
                L_s_t = -torch.log((torch.softmax(self.sim(feats.unsqueeze(1).repeat([1,self.nclasses,1]),self.mean_t.unsqueeze(0).repeat([feats.shape[0],1,1]))/self.t,dim=-1)*F.one_hot(labels,self.nclasses)).sum(-1)).mean()
                
                L_p = -torch.log((torch.softmax(self.sim(zu_s.unsqueeze(1).repeat([1,self.nclasses,1]),self.mean_t.unsqueeze(0).repeat([zu_s.shape[0],1,1]))/self.t,dim=-1)*F.one_hot(torch.arange(self.args.nclasses).long().cuda(),self.nclasses)).sum(-1)).mean()
                
                
                tgt_dom_label = torch.arange(self.args.nclasses).long().cuda()
                tgt_logit = self.C(self.mean_t)
                loss_cls_dom_tgt = criterion(tgt_logit,tgt_dom_label)
                loss_combined = criterion(self.C(zu_s),tgt_dom_label)
                loss_cls = loss_cls_src  +(loss_cls_dom + loss_cls_dom_tgt + loss_combined)
                loss_relation =  L_s + L_t + L_t_s + L_s_t + L_p
                # loss_local = local_src + local_tgt
                # loss_relation = loss_local + loss_global
                
                
                mmdloss = MMDLoss()
                
                if feat_p.shape[0]!=0:
                    lambd = 2 / (1 + torch.exp(-10 * (torch.tensor([epoch])+1) / 200)) - 1 
                    loss = loss_cls + loss_relation+ (mmdloss(zu_s,feat_p))*lambd.item()
                      
                else:    
                    loss = loss_cls + loss_relation
                # assert not torch.isnan(loss_global).item()
           
            loss.backward(retain_graph = True)
            self.opt_C.step()
            self.opt_g.step()
            self.mean = self.mean.data
            self.mean.requires_grad=True
            self.mean_t = self.mean_t.data
            self.mean_t.requires_grad=True
            # record training information
            if epoch ==0 and batch_idx==0:
                record = open(record_file, 'a')
                record.write(str(self.args)+'\n')
                record.close()

            if (self.mean_t_flag):
                print(
                    'Train Epoch: {:>3} [{:>3}/{} ({:.2f}%)]\tLoss_cls_domain: {:.5f}\tLoss_cls_source: {:.5f}'
                    '\tLoss_cls_dom_tgt: {:.5f}\tLoss_combined: {:.5f}\tLoss_cls: {:.5f}\tloss_relation: {:.5f}\tloss_MMD: {:.5f}'.format(
                        epoch, batch_idx + 1, self.niter, (batch_idx + 1.) / self.niter * 100,
                        loss_cls_dom.item(), loss_cls_src.item(), loss_cls_dom_tgt.item(),
                        loss_combined.item(), loss_cls.item(), loss_relation.item(),(loss-loss_cls-loss_relation).item()))
                if record_file:
                    record = open(record_file, 'a')
                    record.write(
                        '\nTrain Epoch: {:>3} [{:>3}/{} ({:.2f}%)]\tLoss_cls_domain: {:.5f}\tLoss_cls_source: {:.5f}'
                        '\tLoss_cls_dom_tgt: {:.5f}\tLoss_combined: {:.5f}\tLoss_cls: {:.5f}\tloss_relation: {:.5f}\tloss_MMD: {:.5f}'.format(
                        epoch, batch_idx + 1, self.niter, (batch_idx + 1.) / self.niter * 100,
                        loss_cls_dom.item(), loss_cls_src.item(), loss_cls_dom_tgt.item(),
                        loss_combined.item(), loss_cls.item(), loss_relation.item(),(loss-loss_cls-loss_relation).item()))
                    record.close()
            else:
                print(
                    'Train Epoch: {:>3} [{:>3}/{} ({:.2f}%)]\tLoss_cls_domain: {:.5f}\tLoss_cls_source: {:.5f}'
                    ''.format(
                        epoch, batch_idx + 1, self.niter, (batch_idx + 1.) / self.niter * 100,
                        loss_cls_dom.item(), loss_cls_src.item()))
                if record_file:
                    record = open(record_file, 'a')
                    record.write(
                        '\nTrain Epoch: {:>3} [{:>3}/{} ({:.2f}%)]\tLoss_cls_domain: {:.5f}\tLoss_cls_source: {:.5f}'
                        ''.format(
                        epoch, batch_idx + 1, self.niter, (batch_idx + 1.) / self.niter * 100,
                        loss_cls_dom.item(), loss_cls_src.item()))
                    record.close()
        return batch_idx
  
    # per epoch test on target domain
    def test(self, epoch, record_file=None, save_model=False):
        self.G.eval()
        self.C.eval()

        test_loss = 0
        correct = 0
        size = 0

        for batch_idx, data in enumerate(self.dataset_test):
            img = data['T']
            label = data['T_label']
            img, label = img.cuda(), label.long().cuda()

            feat = self.G(img)
            # plotlabels(feat,label)
            C_feats = torch.cat([self.mean, feat], dim=0)
            # C_adj = self.construct_adj(feat)
            C_logit = self.C(C_feats)
            output = C_logit[self.mean.shape[0]:, :]

            test_loss += -F.nll_loss(output, label).item()
            pred = output.max(1)[1]
            k = label.size()[0]
            correct += pred.eq(label).cpu().sum()
            size += k

        test_loss = test_loss / size

        if correct > self.best_correct:
            self.best_correct = correct
            if save_model:
                best_state = {'G': self.G.state_dict(), 'C': self.C.state_dict(), 'mean': self.mean.cpu(),
                              'adj': self.adj.cpu(), 'epoch': epoch}
                torch.save(best_state, os.path.join(self.checkpoint_dir, 'best_model.pth'))

        # save checkpoint
        if save_model and epoch % self.save_epoch == 0:
            state = {'G': self.G.state_dict(), 'C': self.C.state_dict()}
            torch.save(state, os.path.join(self.checkpoint_dir, 'epoch_' + str(epoch) + '.pth'))

        # record test information
        print(
            '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%), Best Accuracy: {}/{} ({:.4f}%)  \n'.format(
                test_loss, correct, size, 100. * float(correct) / size, self.best_correct, size,
                                          100. * float(self.best_correct) / size))

        if record_file:
            if epoch == 0:
                record = open(record_file, 'a')
                record.write(str(self.args))
                record.close()

            record = open(record_file, 'a')
            print('recording %s', record_file)
            record.write(
                '\nEpoch {:>3} Average loss: {:.5f}, Accuracy: {:.5f}, Best Accuracy: {:.5f}'.format(
                    epoch, test_loss, 100. * float(correct) / size, 100. * float(self.best_correct) / size))
            record.close()
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, ReduceLROnPlateau

import time
import math
import numpy as np
from tqdm import tqdm, trange
from itertools import repeat
from sklearn.mixture import GaussianMixture

from .layer import Encoder, Decoder, build_mlp, DeterministicWarmup
from .loss import elbo, elbo_SCALE

import scanpy as sc
import episcanpy.api as epi


class VAE(nn.Module):
    def __init__(self, dims, bn=False, dropout=0, binary=True):
        """
        Variational Autoencoder [Kingma 2013] model
        consisting of an encoder/decoder pair for which
        a variational distribution is fitted to the
        encoder. Also known as the M1 model in [Kingma 2014].

        :param dims: x, z and hidden dimensions of the networks
        """
        super(VAE, self).__init__()
        [x_dim, z_dim, encode_dim, decode_dim] = dims
        self.binary = binary
        if binary:
            decode_activation = nn.Sigmoid()
        else:
            decode_activation = None

        self.encoder = Encoder([x_dim, encode_dim, z_dim], bn=bn, dropout=dropout)
        self.decoder = Decoder([z_dim, decode_dim, x_dim], bn=bn, dropout=dropout, output_activation=decode_activation)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize weights
        """
        for m in self.modules():
            # 这个条件语句判断当前模块 m 是否为线性层(nn.Linear)，只有当模块是线性层时才会进行参数初始化操作。
            if isinstance(m, nn.Linear):
                # 使用 Xavier 初始化方法对当前线性层的权重(m.weight)进行初始化。
                # Xavier 初始化是一种常用的权重初始化方法，旨在使输入和输出的方差保持一致，以便在训练过程中更好地进行梯度传播。
                init.xavier_normal_(m.weight.data)
                # 如果线性层有偏置参数，则将偏置参数初始化为零。
                # 这里的 m.bias.data 是偏置参数的数据，zero_() 方法将其所有元素设置为零
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y=None):
        """
        Runs a data point through the model in order
        to provide its reconstruction and q distribution
        parameters.

        :param x: input data
        :return: reconstructed input
        """
        z, mu, logvar = self.encoder(x)
        recon_x = self.decoder(z)

        return recon_x

    def loss_function(self, x):
        z, mu, logvar = self.encoder(x)
        recon_x = self.decoder(z)
        likelihood, kl_loss = elbo(recon_x, x, (mu, logvar), binary=self.binary)

        return (-likelihood, kl_loss)


    def predict(self, dataloader, device='cpu', method='kmeans'):
        """
        Predict assignments applying k-means on latent feature

        Input: 
            x, data matrix
        Return:
            predicted cluster assignments
        """

        if method == 'kmeans':
            from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
            feature = self.encodeBatch(dataloader, device)
            kmeans = KMeans(n_clusters=self.n_centroids, n_init=20, random_state=0)
            pred = kmeans.fit_predict(feature)
        elif method == 'gmm':
            logits = self.encodeBatch(dataloader, device, out='logit')
            pred = np.argmax(logits, axis=1)

        return pred

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

    def fit(self, dataloader,
            lr=0.002, 
            weight_decay=5e-4,
            device='cpu',
            beta = 1,
            n = 200,
            max_iter=30000,
            verbose=True,
            patience=100,
            outdir=None,
       ):

        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay) 
        Beta = DeterministicWarmup(n=n, t_max=beta)
        
        iteration = 0
        # n_epoch = int(np.ceil(max_iter/len(dataloader)))
        n_epoch = 1000
        early_stopping = EarlyStopping(patience=patience, outdir=outdir)
        with tqdm(range(n_epoch), total=n_epoch, desc='Epochs') as tq:
            for epoch in tq:
#                 epoch_loss = 0
                epoch_recon_loss, epoch_kl_loss = 0, 0
                tk0 = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, desc='Iterations')
                for i, x in tk0:
#                     epoch_lr = adjust_learning_rate(lr, optimizer, iteration)
                    x = x.float().to(device)
                    optimizer.zero_grad()
                    
                    recon_loss, kl_loss = self.loss_function(x)
#                     loss = (recon_loss + next(Beta) * kl_loss)/len(x);
                    loss = (recon_loss + kl_loss)/len(x)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(self.parameters(), 10) # clip
                    optimizer.step()
                    
                    epoch_kl_loss += kl_loss.item()
                    epoch_recon_loss += recon_loss.item()

                    tk0.set_postfix_str('loss={:.3f} recon_loss={:.3f} kl_loss={:.3f}'.format(
                            loss, recon_loss/len(x), kl_loss/len(x)))
                    tk0.update(1)
                    
                    iteration+=1
                tq.set_postfix_str('recon_loss {:.3f} kl_loss {:.3f}'.format(
                    epoch_recon_loss/((i+1)*len(x)), epoch_kl_loss/((i+1)*len(x))))

    def fit_clear(self, adata, dataloader, testloader, k,
            lr=0.002,
            weight_decay=5e-4,
            device='cpu',
            beta=1,
            n=200,
            max_iter=30000,
            verbose=True,
            patience=100,
            outdir=None,
            ):

        self.to(device)
        start_time = time.time()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        # Beta = DeterministicWarmup(n=n, t_max=beta)

        iteration = 0
        # n_epoch = int(np.ceil(max_iter/len(dataloader)))
        n_epoch = 1000
        # early_stopping = EarlyStopping(patience=patience, outdir=outdir)
        # ari_max = 0.1

        ari_louvain_max = 0
        ari_leiden_then = 0
        ari_kmeans_then = 0
        epoch_max = 0
        nmi_max = 0
        f1_max = 0
        with tqdm(range(n_epoch), total=n_epoch, desc='Epochs') as tq:
            for epoch in tq:
                #                 epoch_loss = 0
                epoch_recon_loss, epoch_kl_loss = 0, 0
                tk0 = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, desc='Iterations')
                for i, x in tk0:
                                        # epoch_lr = adjust_learning_rate(lr, optimizer, iteration)
                    x = x.float().to(device)
                    optimizer.zero_grad()

                    recon_loss, kl_loss = self.loss_function(x)
                    #                     loss = (recon_loss + next(Beta) * kl_loss)/len(x);
                    loss = (recon_loss + kl_loss) / len(x)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(self.parameters(), 10)  # clip
                    optimizer.step()

                    epoch_kl_loss += kl_loss.item()
                    epoch_recon_loss += recon_loss.item()

                    tk0.set_postfix_str('loss={:.3f} recon_loss={:.3f} kl_loss={:.3f}'.format(
                        loss, recon_loss / len(x), kl_loss / len(x)))
                    tk0.update(1)

                    iteration += 1
                tq.set_postfix_str('recon_loss {:.3f} kl_loss {:.3f}'.format(
                    epoch_recon_loss / ((i + 1) * len(x)), epoch_kl_loss / ((i + 1) * len(x))))

                print('\n')
                if epoch % 10 == 0:
                    # # 1. latent feature
                    adata.obsm['latent'] = self.encodeBatch(testloader, device=device, out='z')
                    # 2. cluster
                    sc.pp.neighbors(adata, n_neighbors=30, use_rep='latent')
                    # louvain算法
                    epi.tl.louvain(adata)
                    ari_louvain = epi.tl.ARI(adata, 'louvain', 'cell_type')
                    print(f'louvain_ari: {ari_louvain}.\n')

                    # leiden算法
                    epi.tl.leiden(adata)
                    ari_leiden = epi.tl.ARI(adata, 'leiden', 'cell_type')
                    print(f'leiden_ari: {ari_leiden}.\n')

                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
                    adata.obs['kmeans'] = kmeans.fit_predict(adata.obsm['latent']).astype(str)
                    ari_kmeans = epi.tl.ARI(adata, 'kmeans', 'cell_type')
                    # 结束计时
                    end_time = time.time()
                    # 计算运行时间
                    elapsed_time = end_time - start_time
                    print(f'kmeans_ari: {ari_kmeans}, with time:{elapsed_time} s.')
                    # if ari_louvain > ari_louvain_max:
                    if ari_kmeans > ari_kmeans_then:
                        ari_louvain_max = ari_louvain
                        ari_leiden_then = ari_leiden
                        ari_kmeans_then = ari_kmeans
                        # nmi_max = nmi
                        # f1_max = f1
                        epoch_max = epoch
                    # if ari > ari_max:
                    #     ari_max = ari
                    #     epoch_max = epoch
                        if outdir:
                            sc.settings.figdir = outdir
                            torch.save(self.state_dict(), os.path.join(outdir, 'model.pt'))  # save model
                            # print(adata)
                            # ----------------------------UMAP---------------------------------
                            sc.tl.umap(adata, min_dist=0.1)
                            # print(adata)
                            color = [c for c in ['louvain', 'cell_type'] if c in adata.obs]
                            sc.pl.umap(adata, color=color, save='_louvain.png', show=False, wspace=0.4, ncols=4)

                            color = [c for c in ['leiden', 'cell_type'] if c in adata.obs]
                            sc.pl.umap(adata, color=color, save='_leiden.png', show=False, wspace=0.4,
                                       ncols=4)
                            color = [c for c in ['kmeans', 'cell_type'] if c in adata.obs]
                            sc.pl.umap(adata, color=color, save='_kmeans.png', show=False, wspace=0.4,
                                       ncols=4)
                            # ----------------------------TSNE---------------------------------
                            sc.tl.tsne(adata, use_rep='latent')
                            color = [c for c in ['louvain', 'cell_type'] if c in adata.obs]
                            sc.pl.tsne(adata, color=color, save='_louvain.png', show=False, wspace=0.4, ncols=4)
                            color = [c for c in ['leiden', 'cell_type'] if c in adata.obs]
                            sc.pl.tsne(adata, color=color, save='_leiden.png', show=False, wspace=0.4,
                                       ncols=4)
                            color = [c for c in ['kmeans', 'cell_type'] if c in adata.obs]
                            sc.pl.tsne(adata, color=color, save='_kmeans.png', show=False, wspace=0.4,
                                       ncols=4)
                print(
                    f'ari_louvain_then:{ari_louvain_max}, then ari_leiden:{ari_leiden_then}, ari_kmeans_max:{ari_kmeans_then}, epoch:{epoch_max}.\n')

    # 这是一个用于批量编码数据的函数 encodeBatch。它通过给定的数据加载器(dataloader)，
    # 将输入数据转换为潜在向量 z，重建数据 x 或者聚类分配 logit。这个函数可以在模型训练后使用，用于对新的数据进行编码或者解码操作。
    def encodeBatch(self, dataloader, device='cpu', out='z', transforms=None):
        output = []
        for x in dataloader:
            # 将当前批次 x 展平为二维张量，并转换为浮点数类型，并将其放置到设备 (device) 上进行计算，通常是 GPU。
            x = x.view(x.size(0), -1).float().to(device)
            # 使用模型的编码器(self.encoder)将输入数据 x
            # 编码为潜在向量 z，以及潜在向量 z 对应的均值 mu 和对数方差 logvar。
            z, mu, logvar = self.encoder(x)
            # 如果 out 参数是 'z'，表示需要返回潜在向量 z，则将 z 添加到 output 列表中，
            # 同时使用 detach().cpu() 将结果从计算设备中分离，并转移到 CPU 上。
            if out == 'z':
                output.append(z.detach().cpu())
            # 如果 out 参数是 'x'，表示需要返回解码后的数据 x，则使用模型的解码器(self.decoder)对潜在向量 z 进行解码，
            # 并将解码结果添加到 output 列表中。
            elif out == 'x':
                recon_x = self.decoder(z)
                output.append(recon_x.detach().cpu().data)
            # 如果 out 参数是 'logit'，表示需要返回聚类分配，这里使用了一个 self.get_gamma(z)[0] 方法，
            # 该方法似乎与聚类相关，返回一个与 z 相关的聚类分配。将聚类分配结果添加到 output 列表中。
            elif out == 'logit':
                output.append(self.get_gamma(z)[0].cpu().detach().data)
        # 将 output 列表中的所有结果拼接成一个大的张量，并将其转换为 NumPy 数组形式。
        output = torch.cat(output).numpy()

        return output



class SCALE(VAE):
    def __init__(self, dims, n_centroids):
        super(SCALE, self).__init__(dims)
        self.n_centroids = n_centroids
        z_dim = dims[1]

        # init c_params
        self.pi = nn.Parameter(torch.ones(n_centroids)/n_centroids)  # pc
        self.mu_c = nn.Parameter(torch.zeros(z_dim, n_centroids)) # mu
        self.var_c = nn.Parameter(torch.ones(z_dim, n_centroids)) # sigma^2

    def loss_function(self, x):
        # 其中，z是潜在空间(latent space)中的样本，mu是潜在空间的均值，logvar是潜在空间的对数方差。
        # 编码器的目的是将输入数据 x 映射到潜在空间中。
        # x:(32,11285),z, mu, logvar:(32,10)
        z, mu, logvar = self.encoder(x)
        # 这是一个解码器(decoder)函数，它接受潜在变量 z，并尝试将其还原为与原始输入数据相匹配的重建数据 recon_x。
        # 解码器的目标是从潜在空间中重建原始数据。
        # recon_x:(32,11285)
        recon_x = self.decoder(z)
        # gamma 是一个矩阵，用于衡量输入样本与潜在聚类中心的相关性，
        # mu_c 和 var_c 是聚类中心的均值和方差， pi 是聚类中心的先验概率。
        gamma, mu_c, var_c, pi = self.get_gamma(z) #, self.n_centroids, c_params)
        # ELBO是用于训练变分自编码器的常见损失函数，有助于保持重建的质量同时确保潜在空间的充分表示。
        # 这里的 recon_x 是重建的数据，x 是原始输入数据，gamma 是用于聚类的矩阵，(mu_c, var_c, pi) 是聚类中心的统计信息，
        # (mu, logvar) 是潜在空间的统计信息。-likelihood 是负对数似然，用于衡量重建数据与原始数据之间的差异。
        # kl_loss 是KL散度损失，用于测量潜在空间中的分布与先验分布之间的差异。
        likelihood, kl_loss = elbo_SCALE(recon_x, x, gamma, (mu_c, var_c, pi), (mu, logvar), binary=self.binary)

        return -likelihood, kl_loss

    def get_gamma(self, z):
        """
        Inference c from z

        gamma is q(c|x)
        q(c|x) = p(c|z) = p(c)p(c|z)/p(z)
        """
        n_centroids = self.n_centroids

        N = z.size(0)
        z = z.unsqueeze(2).expand(z.size(0), z.size(1), n_centroids)
        pi = self.pi.repeat(N, 1) # NxK
#         pi = torch.clamp(self.pi.repeat(N,1), 1e-10, 1) # NxK
        mu_c = self.mu_c.repeat(N,1,1) # NxDxK
        var_c = self.var_c.repeat(N,1,1) + 1e-8 # NxDxK

        # p(c,z) = p(c)*p(z|c) as p_c_z
        p_c_z = torch.exp(torch.log(pi) - torch.sum(0.5*torch.log(2*math.pi*var_c) + (z-mu_c)**2/(2*var_c), dim=1)) + 1e-10
        gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)

        return gamma, mu_c, var_c, pi

    def init_gmm_params(self, dataloader, device='cpu'):
        """
        Init SCALE model with GMM model parameters
        """
        # n_components是GMM的成分数（即高斯分量的个数），covariance_type='diag'表示每个高斯分量的协方差矩阵是对角矩阵。
        gmm = GaussianMixture(n_components=self.n_centroids, covariance_type='diag')
        # 这里调用了self对象的另一个方法encodeBatch()，它接受dataloader和device作为输入，可能用于对输入数据进行编码，
        # 并返回编码后的结果z。
        z = self.encodeBatch(dataloader, device)
        # 这一步是对编码后的数据z进行GMM拟合，即使用GMM对数据进行聚类，估计各个高斯分量的参数（均值和协方差矩阵）
        gmm.fit(z)
        # 这里将拟合得到的GMM模型的均值参数（gmm.means_）转换为PyTorch张量，并将其复制到SCALE模型的参数mu_c中。
        self.mu_c.data.copy_(torch.from_numpy(gmm.means_.T.astype(np.float32)))
        # 这里将拟合得到的GMM模型的协方差矩阵参数（gmm.covariances_）转换为PyTorch张量，并将其复制到SCALE模型的参数var_c中。
        self.var_c.data.copy_(torch.from_numpy(gmm.covariances_.T.astype(np.float32)))


def adjust_learning_rate(init_lr, optimizer, iteration):
    lr = max(init_lr * (0.9 ** (iteration//10)), 0.0002)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr	


import os
class EarlyStopping:
    """Early stops the training if loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, outdir=None):
        """
        Args:
            patience (int): How long to wait after last time loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf
        self.model_file = os.path.join(outdir, 'model.pt') if outdir else None

    def __call__(self, loss, model):
        if np.isnan(loss):
            self.early_stop = True
        score = -loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_model(self.model_file)
        else:
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        '''Saves model when loss decrease.'''
        if self.verbose:
            print(f'Loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving model ...')
        if self.model_file:
            torch.save(model.state_dict(), self.model_file)
        self.loss_min = loss

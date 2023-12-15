#!/usr/bin/env python

from .layer import *
from .model import *
from .loss import *


#!/usr/bin/env python


from .layer import *
from .model import *
from .loss import *
from .dataset import load_dataset
from .utils import estimate_k, binarization

from .dataset import *


import time
import torch

import numpy as np
import pandas as pd
import os
import scanpy as sc

import matplotlib
matplotlib.use('Agg')
# import scanpy as sc
sc.settings.autoshow = False

from anndata import AnnData
from typing import Union, List

#from .ED_model import Autoencoder
from .plot import *

from sklearn.metrics import f1_score
#from .forebrain import ForeBrain
from ResNetAE_pytorch_00 import ResNetVAE
import episcanpy.api as epi

import anndata as ad
from torch.utils.data import DataLoader

from .forebrain import ForeBrain
from labels_statistic import *

from TFIDF import *


def get_score(true_labels, kmeans_labels):
    # 假设 kmeans_labels 是 K-Means 聚类的结果
    # 假设 true_labels 是真实的标签（如果有的话）
    # 如果没有真实标签，可以使用其他方法来评估聚类结果
    # 计算调整后的兰德指数（ARI）
    ari = adjusted_rand_score(true_labels, kmeans_labels)
    print("Adjusted Rand Index (ARI):", ari)
    # 计算归一化互信息（NMI）
    nmi = normalized_mutual_info_score(true_labels, kmeans_labels)
    print("Normalized Mutual Information (NMI):", nmi)
    # 计算 F1 分数
    f1 = f1_score(true_labels, kmeans_labels, average='weighted')
    print("F1 Score:", f1)
    return ari, nmi, f1

def some_function(
        data_list:Union[str, List], 
        n_centroids:int = 30,
        outdir:bool = None,
        verbose:bool = False,
        pretrain:str = None,
        lr:float = 0.0002,
        batch_size:int = 64,
        gpu:int = 0,
        seed:int = 18,
        encode_dim:List = [1024, 128],
        decode_dim:List = [],
        latent:int = 10,
        min_peaks:int = 100,
        min_cells:Union[float, int] = 3,
        n_feature:int = 100000,
        log_transform:bool = False,
        max_iter:int = 30000,
        weight_decay:float = 5e-4,
        impute:bool = False,
        binary:bool = False,
        embed:str = 'UMAP',
        reference:str = 'cell_type',
        cluster_method:str = 'leiden',
    )->AnnData:

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available(): # cuda device
        device='cuda'
        torch.cuda.set_device(gpu)
    else:
        device='cpu'
    # device = 'cpu'
    
    print("\n**********************************************************************")
    print(" Ressac: Resnet based single-cell ATAC-seq clustering")
    print("**********************************************************************\n")


    # data = LabelsFile()
    # print(len(data.labels))
    # print(data.cluster_num)

    # # ------------------------------------------------------------------------------------------------
    # data = ForeBrain()
    # # -----------------------------------------------------------------------------------------------------

    adata, trainloader, testloader = load_dataset_new(
        data_list,
        min_genes=min_peaks,
        min_cells=min_cells,
        n_top_genes=n_feature,
        batch_size=batch_size,
        log=None,
    )

    n_obs, n_vars = adata.shape
    print(n_obs, n_vars)
    s = math.floor(math.sqrt(n_vars))

    cell_num = adata.shape[0]
    input_dim = adata.shape[1]

    k = n_centroids
    # k = data.cluster_num

    if outdir:
        outdir = outdir + '/'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        print('outdir: {}'.format(outdir))

    print("\n======== Parameters ========")
    print(
        'Cell number: {}\nPeak number: {}\ndevice: {}\nlr: {}\nbatch_size: {}\ncell filter by peaks: {}\npeak filter by cells: {}'.format(
            cell_num, input_dim, device, lr, batch_size, min_peaks, min_cells))
    print("============================")

    latent = 10
    encode_dim = [1024, 128]
    decode_dim = []
    dims = [input_dim, latent, encode_dim, decode_dim]

    # model = SCALE(dims, n_centroids=k)
    model = ResNetVAE(input_shape=(s, s, 1), n_centroids=k, dims=dims).to(device)

    model = model.to(torch.float)

    if not pretrain:
        print('\n## Training Model ##')
        model.init_gmm_params(testloader, device)

        model.fit_res_sc_b(adata, trainloader, testloader, batch_size, k,
                         lr=lr,
                         verbose=verbose,
                         device=device,
                         max_iter=max_iter,
                         outdir=outdir
                         )

    else:
        model_path ='output/model.pt'
        print('\n## Loading Model: {}\n'.format(model_path))
        # 使用 torch.load() 加载模型状态字典，并映射到CPU
        #state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        # state_dict = torch.load(model_path, map_location=torch.device('cuda'))
        state_dict = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(state_dict)
        model.to(device)


    ### output ###

    # 1. latent feature
    # adata.obsm['latent'] = model.encodeBatch(testloader, device=device, out='z')
    adata.obsm['latent'] = model.encodeBatch(testloader, device=device, out='z')
    # print(adata.obsm['latent'])
    # 2. cluster
    sc.pp.neighbors(adata, n_neighbors=30, use_rep='latent')
    if cluster_method == 'leiden':
        sc.tl.leiden(adata)
    elif cluster_method == 'kmeans':
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
        adata.obs['kmeans'] = kmeans.fit_predict(adata.obsm['latent']).astype(str)

        # result = kmeans.labels_
        # ari, nmi, f1 = get_score(data.labels, result)
    print('kmeans:\t', epi.tl.ARI(adata, 'kmeans', 'cell_type'))
    #
    # sc.set_figure_params(dpi=80, figsize=(6,6), fontsize=10)
    # if outdir:
    #     sc.settings.figdir = outdir
    #     save = '.png'
    # else:
    #     save = None
    # if embed == 'UMAP':
    #     sc.tl.umap(adata, min_dist=0.1)
    #     color = [c for c in ['celltype',  'kmeans', 'leiden', 'cell_type'] if c in adata.obs]
    #     sc.pl.umap(adata, color=color, save=save, show=False, wspace=0.4, ncols=4)
    # elif  embed == 'tSNE':
    #     sc.tl.tsne(adata, use_rep='latent')
    #     color = [c for c in ['celltype',  'kmeans', 'leiden', 'cell_type'] if c in adata.obs]
    #     sc.pl.tsne(adata, color=color, save=save, show=False, wspace=0.4, ncols=4)
    #
    # if  impute:
    #     print("Imputation")
    #     adata.obsm['impute'] = model.encodeBatch(testloader, device=device, out='x')
    #     adata.obsm['binary'] = binarization(adata.obsm['impute'], adata.X)
    #
    # # if outdir:
    # #     adata.write(outdir+'adata.h5ad', compression='gzip')
    print("over....")
    
    return adata
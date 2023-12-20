# import scanpy as sc
import anndata as ad
# import scanpy.external as sce
import numpy as np
import pandas as pd
import episcanpy.api as epi

# from ressac.dataset import *
from scipy.io import mmread
from .TFIDF import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from .labels_statistic import *
from scipy.sparse import issparse, csr_matrix

# 需要自行判断输入前是否需要对矩阵进行转置
def process_sparse_matrix(matrix):
    # 检查是否是稀疏矩阵
    if issparse(matrix) and matrix.getformat() == 'csr':
        # 将稀疏矩阵的所有非零元素替换为1
        matrix.data = np.ones_like(matrix.data)
        tfidf_matrix_data = TFIDF_csc(matrix.data)
    else:
        # 如果不是稀疏矩阵，可以选择进行其他处理
        print("Input is not a sparse CSR matrix.")
        tfidf_matrix_data = TFIDF(matrix.data)

    return tfidf_matrix_data


def Clustering_algorithms(adata, savename):
    epi.pp.lazy(adata)
    # knn算法
    print(adata)
    epi.tl.kmeans(adata, num_clusters=40)
    epi.pl.umap(adata, color=['kmeans', 'cell_type'], wspace=0.4)
    plt.savefig("epi_kmeans_"+str(savename)+".png", dpi=300, bbox_inches="tight")
    plt.close()
    print('kmeans:\t', epi.tl.ARI(adata, 'kmeans', 'cell_type'))

    # epi.pp.lazy(adata)
    print(adata)
    # louvain算法
    epi.tl.louvain(adata)
    epi.pl.umap(adata, color=['louvain', 'cell_type'], wspace=0.4)
    plt.savefig("epi_louvain_"+str(savename)+".png", dpi=300, bbox_inches="tight")
    plt.close()
    print('louvain:\t', epi.tl.ARI(adata, 'louvain', 'cell_type'))

    # leiden算法
    # epi.pp.lazy(adata)
    print(adata)
    epi.tl.leiden(adata)
    epi.pl.umap(adata, color=['leiden', 'cell_type'], wspace=0.4)
    plt.savefig("epi_leiden_"+str(savename)+".png", dpi=300, bbox_inches="tight")
    plt.close()
    print('leiden:\t', epi.tl.ARI(adata, 'leiden', 'cell_type'))



if __name__=='__main__':
    # mt_file = 'F:/Ressac/mouse_atlas/atac_matrix.binary.qc_filtered.mm'
    mt_file = 'Mouseatlas_all.mtx'
    # mt_file = 'Mouseatlas_matrix_test.mtx'
    # 读取 .mtx 文件
    print('read .mtx ......')
    matrix = mmread(mt_file)
    print('read .mtx   over......')
    # 转置稀疏矩阵
    transposed_matrix = matrix.transpose()
    data_frame_matrix = pd.DataFrame.sparse.from_spmatrix(transposed_matrix)
    # data_frame_matrix = matrix.toarray()
    # # 将稀疏矩阵转换为 Python 的列表对象
    print('data_frame_matrix  over......')

    # # 数据类型转换
    # matrix_data = data_frame_matrix.astype('int')  # 将数据类型转换为float32
    # print('to float32  over......')
    # 读取数据为 DataFrame
    file_path = 'mouse_atlas/barcodes.txt'  # 文件路径
    data_df = pd.read_csv(file_path, header=None)
    # 获取第一列数据
    first_column = data_df.iloc[:, 0].values
    obs = pd.DataFrame()
    obs['cell_name'] = first_column

    # 读取数据为 DataFrame
    peaks_path = 'mouse_atlas/peaks.txt'  # 文件路径
    peaks_df = pd.read_csv(peaks_path, header=None)
    # # 获取第一列数据
    var_names = peaks_df.iloc[:, 0].values
    var = pd.DataFrame(index=var_names)

    label_path = 'mouse_atlas/labels.txt'  # 文件路径
    # data_df = pd.read_csv(label_path, header=None)
    data_df = pd.read_csv(label_path, sep='\t', header=None, names=['column1', 'column2'])
    # 获取第一列数据
    labels_column = data_df['column1']
    cell_type_column = data_df['column2']
    print(labels_column)
    print(cell_type_column)
    # obs = pd.DataFrame()
    obs['labels'] = labels_column
    obs['cell_type'] = cell_type_column

    min_features = 1000
    min_cells = 5
    batch_size = 32

    adata = ad.AnnData(X = data_frame_matrix, obs=obs, var=var)
    # adata = ad.AnnData(X=matrix.T)
    # adata = ad.AnnData(X=data_frame_matrix)
    print(adata.obs)
    print(adata.var_names)
    print(adata)

    Clustering_algorithms(adata, 436206)
    # --------基于min_features，min_cells-----------进一步筛选----------------------
    # sc.pp.filter_cells(adata, min_genes=min_features)
    # print(adata)
    # sc.pp.filter_genes(adata, min_cells=min_cells)
    # print(adata)

    epi.pp.filter_cells(adata, min_features=min_features)
    print(adata)
    epi.pp.filter_features(adata, min_cells=min_cells)
    print(adata)

    # 筛选特征数量（通常是基因数量）大于阈值 min_features 的细胞，以及至少在 min_cells 个细胞中被检测到的特征（通常是基因）
    adata_Mouseatlas_1000_5 = 'find_genes_adata_Mouseatlas_1000_5_epi.h5ad'
    adata.write_h5ad(adata_Mouseatlas_1000_5)
    print(f'AnnData 已保存到 {adata_Mouseatlas_1000_5}')

    # 要重新读取保存的 AnnData 文件，可以使用 read_h5ad 方法
    adata = ad.read_h5ad(adata_Mouseatlas_1000_5)
    print(adata)
    #-------------控制数据集大小-----------------
    # # # 436206, %30
    # # nb_feature_selected = 130000
    # # 436206, %10
    nb_feature_selected = 43600
    adata = epi.pp.select_var_feature(adata,
                                      nb_features=nb_feature_selected,
                                      show=False,
                                      copy=True)
    print(adata)

    # 保留能够开方的最大特征数，便于送入模型
    n_obs, n_vars = adata.shape
    print(n_obs, n_vars)
    s = math.floor(math.sqrt(n_vars))
    num_features_to_keep = s*s
    print(num_features_to_keep)

    # 随机生成一个索引数组，表示要保留的特征
    indices_to_keep = np.random.choice(n_vars, num_features_to_keep, replace=False)
    # 使用索引数组来选择要保留的特征
    adata = adata[:, indices_to_keep]
    print(adata)

    adata_Mouseatlas_43600 = 'find_genes_adata_Mouseatlas_'+str(num_features_to_keep)+'.h5ad'
    # adata_Mouseatlas_43600 = 'find_genes_adata_Mouseatlas_43264.h5ad'
    adata.write_h5ad(adata_Mouseatlas_43600)
    print(f'AnnData 已保存到 {adata_Mouseatlas_43600}')
    # 要重新读取保存的 AnnData 文件，可以使用 read_h5ad 方法
    adata = ad.read_h5ad(adata_Mouseatlas_43600)
    print(adata)


    print('over..........')



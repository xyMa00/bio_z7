# import scanpy as sc
import anndata as ad
# import scanpy.external as sce
import numpy as np
import pandas as pd
import episcanpy.api as epi

# from scale.dataset import *
from scipy.io import mmread
from TFIDF import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
# from labels_statistic import *
from scipy.sparse import issparse, csr_matrix
import argparse

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

def add_celltype(file_path, label_path, save_name):
    # 读取数据为 DataFrame
    data_frame = pd.read_csv(file_path, sep='\t', header=0, index_col=0)
    data_matrix = data_frame.T
    # 创建 anndata 对象
    adata = ad.AnnData(data_matrix)
    # print(adata)

    # # -----------------------add cell_type-----------------------------------
    data_frame = pd.read_csv(label_path, sep='\t', header=0)
    cell_type_column = data_frame['cell_type']
    obs = pd.DataFrame()
    adata.obs['cell_type'] = cell_type_column.tolist()
    print(adata)
    # print('cell_type: ', adata.obs['cell_type'])


    adata_h5ad = 'find_genes_'+save_name+'.h5ad'
    adata.write_h5ad(adata_h5ad)
    print(f'AnnData 已保存到 {adata_h5ad}')
    # adata = ad.read_h5ad(adata_GSM_34051)
    # print(adata)



    return adata, adata_h5ad

def clusters(h5ad_path):
    # adata_h5ad = 'find_genes_' + save_name + '.h5ad'
    # adata.write_h5ad(adata_h5ad)
    # print(f'AnnData 已保存到 {adata_h5ad}')
    adata = ad.read_h5ad(h5ad_path)
    print(adata)
    # ----------------------------聚类算法----------------------------------------
    epi.pp.lazy(adata)
    # ----------------------------umap---------------------------------
    # knn算法
    # print(adata)
    epi.tl.kmeans(adata, num_clusters=4)
    epi.pl.umap(adata, color=['kmeans', 'cell_type'], wspace=0.4)
    plt.savefig("epi_kmeans_umap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print('kmeans:\t', epi.tl.ARI(adata, 'kmeans', 'cell_type'))

    # epi.pp.lazy(adata)
    # print(adata)
    # louvain算法
    epi.tl.louvain(adata)
    epi.pl.umap(adata, color=['louvain', 'cell_type'], wspace=0.4)
    plt.savefig("epi_louvain_umap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print('louvain:\t', epi.tl.ARI(adata, 'louvain', 'cell_type'))

    # leiden算法
    # epi.pp.lazy(adata)
    # print(adata)
    epi.tl.leiden(adata)
    epi.pl.umap(adata, color=['leiden', 'cell_type'], wspace=0.4)
    plt.savefig("epi_leiden_umap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print('leiden:\t', epi.tl.ARI(adata, 'leiden', 'cell_type'))

    # ----------------------------TSNE---------------------------------

    # knn算法
    # print(adata)
    epi.tl.kmeans(adata, num_clusters=4)
    epi.pl.tsne(adata, color=['kmeans', 'cell_type'], wspace=0.4)
    plt.savefig("epi_kmeans_tsne.png", dpi=300, bbox_inches="tight")
    plt.close()

    # epi.pp.lazy(adata)
    # print(adata)
    # louvain算法
    epi.tl.louvain(adata)
    epi.pl.tsne(adata, color=['louvain', 'cell_type'], wspace=0.4)
    plt.savefig("epi_louvain_tsne.png", dpi=300, bbox_inches="tight")
    plt.close()

    # leiden算法
    # epi.pp.lazy(adata)
    # print(adata)
    epi.tl.leiden(adata)
    epi.pl.tsne(adata, color=['leiden', 'cell_type'], wspace=0.4)
    plt.savefig("epi_leiden_tsne.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ----------------------------------------------------------------------

def doFilters(h5ad_path, min_features, min_cells, nb_feature_selected,save_name):

    adata = ad.read_h5ad(h5ad_path)
    print(adata)

    epi.pp.filter_cells(adata, min_features=min_features)
    # (18863,34051)
    print(adata)
    epi.pp.filter_features(adata, min_cells=min_cells)
    # 18863 脳 33667
    print(adata)

    # # -----------------------select_var_feature-----------------------------------
    n_obs, n_vars = adata.shape
    print(n_obs, n_vars)
    # 33667, %30
    # nb_feature_selected = 23835
    nb_feature_selected = n_vars * nb_feature_selected
    adata = epi.pp.select_var_feature(adata,
                                      nb_features=nb_feature_selected,
                                      show=False,
                                      copy=True)
    print(adata)


    n_obs, n_vars = adata.shape
    print(n_obs, n_vars)
    # 33667, %30
    # nb_feature_selected = 23835
    nb_feature_selected = n_vars * nb_feature_selected
    adata = epi.pp.select_var_feature(adata,
                                      nb_features=nb_feature_selected,
                                      show=False,
                                      copy=True)
    print(adata)
    # #
    # 保留能够开方的最大特征数
    n_obs, n_vars = adata.shape
    print(n_obs, n_vars)
    s = math.floor(math.sqrt(n_vars))
    num_features_to_keep = s*s
    print(num_features_to_keep)

    # # 指定要保留的特征数量
    # num_features_to_keep = 43264  # 例如，要保留1000个特征
    # # 获取 X 的形状
    # n_obs, n_vars = adata.shape
    # 随机生成一个索引数组，表示要保留的特征
    indices_to_keep = np.random.choice(n_vars, num_features_to_keep, replace=False)
    # 使用索引数组来选择要保留的特征
    adata = adata[:, indices_to_keep]
    print(adata)


    adata_filtered = 'find_genes_'+save_name +'_'+str(num_features_to_keep)+'.h5ad'
    adata.write_h5ad(adata_filtered)
    print(f'AnnData 已保存到 {adata_filtered}')
    return adata_filtered


if __name__=='__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='add cell_type')
    # 添加参数
    parser.add_argument('file_path', type=str, help='The origin h5ad file path.')
    parser.add_argument('label_path', type=str, help='The path to the file containing cell_type.')
    parser.add_argument('save_name', type=str, help='the final file name.')
    parser.add_argument('filters', type=bool, help='Choose the suitable cells and peaks.', default=False)
    parser.add_argument('min_features', type=int,
                        help='The cell contains at least min_features.', default=1000)
    parser.add_argument('min_cells', type=int,
                        help='Features that are detected in at least min_cells.', default=5)
    parser.add_argument('nb_feature_selected', type=float,
                        help='The ratio of the most specific peaks you want to keep.', default=1)

    # 解析命令行参数
    args = parser.parse_args()

    # 访问参数
    print("file_path:", args.file_path)
    print("label_path:", args.label_path)
    print("save_name:", args.save_name)

    print("filter:", args.file_path)
    print("min_features:", args.label_path)
    print("min_cells:", args.save_name)
    print("nb_feature_selected:", args.nb_feature_selected)

    file_path = args.file_path
    label_path = args.label_path
    save_name = args.save_name

    filter = args.filters
    min_features = args.min_features
    min_cells = args.min_cells
    nb_feature_selected = args.nb_feature_selected

    adata, h5ad_path = add_celltype(file_path, label_path, save_name)
    if filter:
        h5ad_path = doFilters(h5ad_path, min_features, min_cells, nb_feature_selected, save_name)

    clusters(h5ad_path)
    print('over..........')



import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix, diags
import anndata as ad


def TFIDF_csc(object=[]):

    # object = transposed_df
    # 使用 NumPy 的 sum 函数计算列和
    # npeaks = np.sum(object, axis=0)
    npeaks = object.sum(axis=0)  # 计算每列的和
    print('npeaks...', npeaks.shape)
    print(type(npeaks))
    # 创建对角矩阵 Diagonal(x = 1 / npeaks)
    # diagonal_matrix = np.diag(1 / npeaks)
    # diagonal_matrix = diags(1 / npeaks, format='csc')  # 创建对角矩阵
    # diagonal_matrix = diags(1 / npeaks, offsets=0, format='csc')
    # 从列和创建对角矩阵
    diagonal_matrix = diags(1/npeaks.A.ravel(), format='csc')
    # 计算转置乘积
    # tf = np.dot(object.T, diagonal_matrix)
    print('diagonal_matrix...')
    diagonal_matrix_t = diagonal_matrix.transpose()
    print('diagonal_matrix_t....')
    # tf = np.dot(object, diagonal_matrix.T)
    # tf = np.dot(object, diagonal_matrix_t)
    tf = object.dot(diagonal_matrix_t)
    # tf_m = tf.toarray()
    print('tf...')
    # 使用 NumPy 的 sum 函数计算行和
    # rsums = np.sum(object, axis=1)
    # rsums = np.sum(object, axis=1)
    rsums = object.sum(axis=1)
    # 计算稀疏矩阵的行和
    # rsums = np.array(object.sum(axis=1))
    print('rsums...')
    # 计算 object 列数（ncol）
    ncol_object = object.shape[1]
    print('ncol_object...')
    # 计算 idf
    idf = ncol_object / rsums
    print('idf...')
    # 创建对角矩阵 Diagonal(n = length(x = idf), x = idf)
    # diagonal_matrix_idf = np.diag(idf)
    # diagonal_matrix_idf = diags(idf, format='csc')  # 创建对角矩阵
    diagonal_matrix_idf = diags(idf.A.ravel(), format='csc')
    # diagonal_matrix_idf = diags(idf, format='csr')
    print('diagonal_matrix_idf...')
    # 矩阵相乘
    norm_data = diagonal_matrix_idf @ tf  # 或者使用 np.dot(diagonal_matrix, tf)
    # norm_data_m=norm_data.toarray()
    # norm_data = diagonal_matrix_idf.dot(tf)
    print('norm_data...')
    # 设置放缩比例
    scale_factor = 1e4
    # 计算R: slot(object = norm.data, name = "x") * scale.factor
    result = np.log1p(norm_data * scale_factor)
    # result_m = result.toarray()
    print('result...')

    tfidf_matrix_data = result.transpose()
    # tfidf_matrix_data = result
    print(type(tfidf_matrix_data))
    print(tfidf_matrix_data.shape)

    # df_1 = pd.DataFrame(data=result)
    # print('df_1...')
    # # # 将 df 的行索引赋值给 df2
    # # df_1 = df_1.set_index(object.index)
    # # # 将 df 的列索引赋值给 df2
    # # df_1 = df_1.set_axis(object.columns, axis=1)
    # tfidf_matrix_data = df_1.T
    # # tfidf_matrix_data = df_1

    return tfidf_matrix_data

def TFIDF(object=[]):

    # object = transposed_df
    # 使用 NumPy 的 sum 函数计算列和
    npeaks = np.sum(object, axis=0)
    # 创建对角矩阵 Diagonal(x = 1 / npeaks)
    diagonal_matrix = np.diag(1 / npeaks)
    # 计算转置乘积
    # tf = np.dot(object.T, diagonal_matrix)
    tf = np.dot(object, diagonal_matrix.T)
    # 使用 NumPy 的 sum 函数计算行和
    rsums = np.sum(object, axis=1)
    # 计算 object 列数（ncol）
    ncol_object = object.shape[1]
    # 计算 idf
    idf = ncol_object / rsums
    # 创建对角矩阵 Diagonal(n = length(x = idf), x = idf)
    diagonal_matrix_idf = np.diag(idf)
    # 矩阵相乘
    norm_data = diagonal_matrix_idf @ tf  # 或者使用 np.dot(diagonal_matrix, tf)
    # 设置放缩比例
    scale_factor = 1e4
    # 计算R: slot(object = norm.data, name = "x") * scale.factor
    result = np.log1p(norm_data * scale_factor)

    df_1 = pd.DataFrame(data=result)
    # # 将 df 的行索引赋值给 df2
    # df_1 = df_1.set_index(object.index)
    # # 将 df 的列索引赋值给 df2
    # df_1 = df_1.set_axis(object.columns, axis=1)
    # tfidf_matrix_data = df_1.T
    tfidf_matrix_data = df_1


    return tfidf_matrix_data




if __name__=='__main__':
    # adata_forebrain_new = 'find_genes_adata_101124.h5ad'
    adata_forebrain_new = 'find_genes_adata_forebrain_11236.h5ad'
    # adata.write_h5ad(adata_forebrain_new)
    print(f'AnnData 已保存到 {adata_forebrain_new}')

    adata = ad.read_h5ad(adata_forebrain_new)
    print(adata)

    # object = adata
    object = adata.X.transpose()
    print('run tf-idf ......')
    tfidf_matrix_data = TFIDF_csc(object)
    # tfidf_matrix_data = TFIDF(object)
    # tfidf_matrix_data_m=tfidf_matrix_data.toarray()
    print('tf-idf over ......')

    adata.X = tfidf_matrix_data
    print(adata)

    # # 转换为 Pandas 数据框
    df = adata.to_df()
    # # 指定 CSV 文件路径并保存数据
    csv_filename = "find_genes_adata_forebrain_11236_tfidf_csc.txt"
    df.to_csv(csv_filename)  # 如果不想保存行索引，请设置 index=False

    # adata_101124_tfidf = 'find_genes_adata_101124_tfidf_csc.h5ad'
    adata_101124_tfidf = 'find_genes_adata_forebrain_11236_tfidf_csc.h5ad'
    adata.write_h5ad(adata_101124_tfidf)

    print(f'AnnData 已保存到 {adata_101124_tfidf}')

    # adata = ad.read_h5ad(adata_Mouseatlas_43264_tfidf)
    print(adata)

    print('over.........')


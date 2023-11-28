import pandas as pd
import anndata as ad


adata_43264 = 'find_genes_adata_Mouseatlas_43264.h5ad'
# # adata.write_h5ad(adata_310249)
# # print(f'AnnData 已保存到 {adata_310249}')

# # 要重新读取保存的 AnnData 文件，可以使用 read_h5ad 方法
adata = ad.read_h5ad(adata_43264)

# 转换为 Pandas 数据框
df = adata.to_df()

# df.to_csv('find_genes_adata_forebrain_11236_TFIDF_new.csv', index=False,header=False)
df.to_csv('matrix_data_43264.csv')

adata.write_csvs('datas_43264')
print(f'AnnData 已保存到 {adata_43264}')



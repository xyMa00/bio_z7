# Ressac: Resnet based single-cell ATAC-seq clustering

#![](https://github.com/xyMa00/bio_z7/wiki/png/RES_ATACseq_model.png)


## Installation  

Ressac neural network is implemented in [Pytorch](https://pytorch.org/) framework.  
Running Ressac on CUDA is recommended if available.   

#### install from PyPI

    pip install Ressac

#### install latest develop version from GitHub
    pip install https://github.com/xyMa00/bio_z7.git
or download and install

	git clone git@github.com:xyMa00/bio_z7.git
	cd Ressac
	python setup.py install
    
Installation only requires a few minutes.  

## Data preprocessing
* First you need to convert the input data to a .h5ad file(this file should contain 'cell_type' in its obs).
* Second, you can screen based on cells and peaks, at the same time, you can choose to keep the most variable features.
* Third, the final number of retained peaks should be the square of some number.

You can refer to **epi_h5ad.py** for the whole process.


## Quick Start

#### Input
* h5ad file(should contain 'cell_type' in its obs).

#### Run 

    Ressac.py -d [input]

#### Output
Output will be saved in the output folder including:
* **model.pt**:  saved model to reproduce results cooperated with option --pretrain
* **tsne_louvain.png**:  clustering result of louvain by tsne.
* **umap_louvain.png**:  clustering result of louvain by umap.
* **tsne_leiden.png**:  clustering result of leiden by tsne.
* **umap_leiden.png**:  clustering result of leiden by umap.

#### Imputation  
Get binary imputed data in adata.h5ad file using scanpy **adata.obsm['binary']** with option **--binary** (recommended for saving storage)

    Ressac.py -d [input] --binary  
    
or get numerical imputed data in adata.h5ad file using scanpy **adata.obsm['imputed']** with option **--impute**

    Ressac.py -d [input] --impute
     
#### Useful options  
* save results in a specific folder: [-o] or [--outdir] 
* modify the initial learning rate, default is 0.001: [--lr]  
* change random seed for parameter initialization, default is 18: [--seed]


#### Help
Look for more usage of Ressac

	Ressac.py --help 

Use functions in Ressac packages.

	import ressac
	from ressac import *
	from ressac.plot import *
	from ressac.utils import *
	

## Tutorial
**[Tutorial Forebrain](https://github.com/xyMa00/bio_z7/wiki/Forebrain)**   Run Ressac on dense matrix **Forebrain** dataset (k=8, 2088 cells)

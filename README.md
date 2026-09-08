# HGRL

## Requirements  需求

```
torch 2.0.0+cu118

python==3.8.1

torch-geometric==1.6.3

In our experiment setting: cuda 11.8
```

> Note: If there is a problem with the torch-sparse installation, please use the link
> https://pytorch-geometric.com/whl/torch-2.0.0%2Bcu118.html
> to download the appropriate torch-sparse version.
>
> 注意：如果 torch-sparse 安装存在任何问题，请使用链接
> https://pytorch-geometric.com/whl/torch-2.0.0%2Bcu118.html
> 下载相应的 torch-sparse 版本。

## How to use  如何使用

```bash
python main.py
```

Drug data should be unzipped and placed in the code directory (e.g., under `drug_data/`).
药物数据应解压后放置在代码目录下（例如 `drug_data/` 目录）。
# 基于 LAS 点云初始化与 CUDA 深度筛选的 3D Gaussian Splatting 实验项目

本项目基于 [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) 修改，目标是研究在真实采集场景中，地理坐标 LAS 点云、XML/AT 相机元数据、COLMAP 格式相机文件以及预生成深度图如何接入 3D Gaussian Splatting。

本项目主要回答两个问题：

1. 能否使用地理 LAS 点云替代 COLMAP 稀疏点云来初始化 3DGS 高斯？
2. 基于预生成深度图的 CUDA 侧硬深度筛选是否能提升收敛速度或最终渲染质量？

本项目使用的数据来自手持激光雷达采集流程。一个完整场景通常包含：

- `*.las`：手持激光雷达导出的彩色点云或几何点云。
- `metadata.xml`：记录坐标原点、相机或工程元数据，其中 `SRSOrigin` 用于坐标对齐。
- RGB 图片：与 XML/相机位姿对应的训练图像。
- mask 图片：用于屏蔽行人等动态物体干扰。
- COLMAP 格式 `sparse/0/`：由相机元数据或 COLMAP 流程生成的相机、图片和稀疏点文件。
- depth maps：由 LAS、相机和图片关系预先生成的每像素深度范围图。

## 核心结论

从当前量化结果看，**COLMAP 初始化 + 无深度筛选是图像质量最优选择**。在 HLSRedHouse/HLSGY 数据集上，它取得最高 PSNR、最高 SSIM 和最低 LPIPS：

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 训练时间趋势 |
| --- | ---: | ---: | ---: | --- |
| COLMAP + 无深度筛选 | **20.635** | **0.658** | **0.445** | 约 58 分钟 |
| COLMAP + 自适应深度筛选 | 20.014 | 0.651 | 0.447 | 约 68 分钟 |
| LAS + 无深度筛选 | 19.541 | 0.605 | 0.500 | 约 48 分钟 |
| LAS + 自适应深度筛选 | 18.832 | 0.595 | 0.504 | 约 49 分钟 |

因此，如果目标是**最终渲染质量**，推荐使用：

```text
COLMAP 初始化 + 无深度筛选
```

如果目标是**更快完成训练或快速实验验证**，推荐使用：

```text
LAS 初始化 + 无深度筛选
```

本项目的实验结论是：**COLMAP + 无深度筛选质量最优；LAS + 无深度筛选速度最快；自适应深度筛选在当前实验中没有带来收益。**

## 目录

- [项目特点](#项目特点)
- [代码结构](#代码结构)
- [环境配置](#环境配置)
- [输入数据结构](#输入数据结构)
- [数据准备说明](#数据准备说明)
- [训练命令](#训练命令)
- [输出日志说明](#输出日志说明)
- [主要代码改动](#主要代码改动)
- [实验设计](#实验设计)
- [实验结果](#实验结果)
- [当前限制与后续工作](#当前限制与后续工作)
- [致谢](#致谢)

## 项目特点

- 支持使用 LAS 点云初始化 3DGS 高斯。
- 使用 `metadata.xml` 中的 `SRSOrigin` 将 LAS 地理坐标对齐到 XML2Colmap 使用的局部坐标系。
- 支持 LAS 点云预处理缓存，包括体素下采样、颜色读取、法线处理和初始高斯尺度估计。
- 支持从 COLMAP 稀疏点云初始化，保留原始 3DGS 基线。
- 支持预生成深度图输入，每张图对应 `_min.npy` 和 `_max.npy` 深度范围。
- 支持 CUDA 侧临时深度筛选，不永久删除高斯。
- 支持自适应困难图片深度筛选，根据每张图片的近期 PSNR EMA 选择低质量视角启用深度筛选。
- 支持固定验证机制，在训练过程中定期用固定相机集合、关闭深度筛选进行对比。
- 支持图片 lazy loading，并在小数据集上自动缓存图片 tensor。
- 支持动态物体 mask，mask 参与训练 loss 和 PSNR 计算，用于减弱行人等动态干扰。

## 算法优点

相比原始 3DGS，本项目的主要优点不是单纯追求更高 PSNR，而是扩展了 3DGS 在手持激光雷达数据上的输入能力和实验验证能力：

- **真实测量点云初始化**：可以直接使用手持激光雷达生成的 LAS 点云初始化高斯，不完全依赖 COLMAP 稀疏点云。
- **统一坐标对齐链路**：通过 `metadata.xml` 中的 `SRSOrigin` 将 LAS、XML/相机、深度图和 COLMAP 格式输入放到同一局部坐标系。
- **更快的实验初始化路线**：在当前 HLS 实验中，LAS 初始化路线整体训练时间约 48-49 分钟，短于 COLMAP 初始化的约 58-68 分钟。
- **动态物体鲁棒监督**：mask 可以屏蔽行人等动态干扰区域，使 loss 和 PSNR 只在有效静态区域计算。
- **深度筛选可控对照**：深度图不会永久删除高斯，而是在 CUDA rasterizer 中按视角临时筛选，便于打开/关闭做严格对照。
- **固定验证机制**：训练过程中的固定验证始终关闭深度筛选，从而比较模型本身质量，避免评估阶段引入额外过滤。
- **大规模真实图片友好**：lazy loading 和小数据集自动缓存降低了大图像集合训练时的内存压力。

## 代码结构

```text
.
├── train.py                         # 训练入口
├── render.py                        # 渲染训练/测试视角
├── metrics.py                       # 计算 PSNR / SSIM / LPIPS
├── arguments/                       # 训练参数定义
├── gaussian_renderer/               # Python 渲染入口
├── scene/                           # 场景、相机、高斯模型
├── utils/                           # 图像、loss、LAS、深度图工具
├── submodules/
│   ├── diff-gaussian-rasterization/ # 修改后的 CUDA rasterizer
│   └── simple-knn/                  # KNN CUDA 扩展
├── depth_map_generate.py            # 深度图生成相关脚本
├── preprocess_depth_pooling.py      # 深度图离线 pooling 预处理
├── filter_images_by_las_coverage.py # 根据 LAS 覆盖筛选图片
├── filter_cubicba_xml.py            # XML/AT 数据处理辅助脚本
├── results.md                       # 原始结果记录或补充材料
└── README.md
```

关键修改文件：

| 文件 | 作用 |
| --- | --- |
| `train.py` | 增加 LAS 初始化、CUDA 深度筛选、自适应深度调度、固定验证、mask loss、图片缓存控制和实验日志。 |
| `scene/gaussian_model.py` | 增加 `create_from_las_data()`，从 LAS 预处理后的数组直接初始化高斯参数。 |
| `utils/point_cloud_utils.py` | 读取 LAS，解析 `SRSOrigin`，完成坐标对齐、下采样、颜色/法线处理、初始尺度估计和缓存。 |
| `gaussian_renderer/__init__.py` | 在 Python 渲染入口中按视角读取深度图，并把深度范围传入 CUDA rasterizer。 |
| `utils/depth_utils.py` | 按需读取 `_min.npy` / `_max.npy` 深度范围，并做 CPU LRU 缓存。 |
| `scene/cameras.py` | 实现 lazy image loading 和可选图片缓存，避免一次性加载大量图像。 |
| `utils/camera_utils.py` | 加载并缩放动态物体 mask，保证不同 `-r` 分辨率下 mask 仍能参与训练。 |
| `metrics.py` | 支持 `train` / `test` split，并逐图计算指标，避免大数据集一次性加载导致 CUDA OOM。 |

## 环境配置

本项目优先沿用原始 3DGS 的环境配置方式，然后额外安装 LAS 和点云处理依赖。

### 1. 克隆项目

```bash
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>
```

如果你使用 Git submodule 管理 CUDA 扩展：

```bash
git clone --recursive https://github.com/<your-user>/<your-repo>.git
cd <your-repo>
```

### 2. 创建 Conda 环境

```bash
conda env create --file environment.yml
conda activate gaussian_splatting
```

Windows 上编译 CUDA 扩展前，建议按原始 3DGS 的方式设置：

```bat
SET DISTUTILS_USE_SDK=1
```

### 3. 安装 CUDA 扩展

```bash
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

如果修改了 `submodules/diff-gaussian-rasterization/` 中的 CUDA/C++ 文件，需要重新安装：

```bash
pip install --force-reinstall submodules/diff-gaussian-rasterization
```

### 4. 安装新增依赖

```bash
pip install laspy open3d scikit-learn
```

这些库用于 LAS 读取、点云下采样、最近邻搜索和初始尺度估计。

## 输入数据结构

推荐使用如下结构：

```text
data/
  <dataset_name>/
    images/
      <image_name>.jpg
    masks/
      <image_name>.jpg.png
    sparse/
      0/
        cameras.bin
        images.bin
        points3D.bin
    metadata.xml
    <scene>.las
    depth_maps/
      <image_base>_min.npy
      <image_base>_max.npy
```

其中 `*.las`、`metadata.xml`、`sparse/`、`depth_maps/` 与图片目录处于同一数据集目录下。`masks/` 为可选目录，但对于手持激光雷达和现场拍摄数据，建议保留，因为照片中可能存在行人或其他动态干扰。

参数对应关系：

| 参数 | 含义 |
| --- | --- |
| `-s`, `--source_path` | 数据集目录，内部应包含 `images/` 和 `sparse/0/`。 |
| `--las_file` | `<dataset_root>/<scene>.las`。提供该参数时使用 LAS 初始化高斯。 |
| `--metadata_path` | `<dataset_root>/metadata.xml`，其中应包含 `SRSOrigin`。 |
| `--depth_dir` | `<dataset_root>/depth_maps`，内部包含 `_min.npy` 和 `_max.npy`。 |
| `--disable_cuda_depth_filter` | 即使提供深度图，也关闭 CUDA 深度筛选，用作无筛选基线。 |
| `--adaptive_depth_filter` | 开启自适应困难图片深度筛选。 |
| `--adaptive_depth_ratio` | 每次选择近期 PSNR 较低的图片比例，默认实验使用 `0.2`。 |

## 数据准备说明

### COLMAP sparse 文件

`sparse/0/` 中的 `cameras.bin`、`images.bin`、`points3D.bin` 是 3DGS 读取相机和初始点云的标准输入。

在本项目中，COLMAP sparse 可以来自两种方式：

1. 直接由 COLMAP SfM 生成。
2. 由 XML/AT 相机元数据转换成 COLMAP 格式相机，再使用 COLMAP 生成或补充 `points3D`。

无论使用哪种方式，图片文件名、相机姿态、LAS 坐标和深度图必须使用同一套坐标约定，否则 LAS 初始化和深度筛选都会失效。

### LAS 与 `metadata.xml`

LAS 原始坐标通常是地理坐标或工程坐标，数值很大。代码会读取 `metadata.xml` 中的 `SRSOrigin`，并执行：

```text
LAS local xyz = LAS raw xyz - SRSOrigin
```

这个局部坐标必须与 XML2Colmap 输出的相机坐标一致。也就是说，生成 COLMAP 相机、生成深度图、读取 LAS 初始化时都应使用同一个 `SRSOrigin`。

### 深度图

训练阶段不会再做 depth pooling。当前代码直接读取预处理好的深度图：

```text
<image_base>_min.npy
<image_base>_max.npy
```

如果需要扩大深度容差或做邻域 pooling，应在训练前通过深度图生成或预处理脚本完成，而不是通过训练参数临时调整 pooling kernel。

### mask

`masks/` 目录是可选的。mask 用于排除照片中的动态人物或其他不希望参与监督的区域。

当前训练中，mask 会参与：

- `masked_l1_loss`
- `masked_psnr`
- 固定验证中的 loss 和 PSNR

mask 不会直接修改 LAS 点云或删除高斯，它只控制哪些像素参与图像监督。

## 训练命令

下面给出四组核心对照实验命令。所有路径均使用通用占位符：

```text
<dataset_root>          数据集根目录，包含 images/、sparse/、metadata.xml、LAS 和 depth_maps/
<scene_las>             LAS 文件名
<depth_dir>             深度图目录
```

### 1. COLMAP 初始化 + 无深度筛选

```bat
python train.py ^
  -s <dataset_root> ^
  --depth_dir <depth_dir> ^
  --iterations 30000 ^
  --disable_cuda_depth_filter ^
  --eval_interval 500 ^
  --eval_count 100 ^
  --experiment_log_interval 1000 ^
  --checkpoint_iterations 10000 20000 30000
```

### 2. COLMAP 初始化 + 自适应深度筛选

```bat
python train.py ^
  -s <dataset_root> ^
  --depth_dir <depth_dir> ^
  --iterations 30000 ^
  --adaptive_depth_filter ^
  --adaptive_depth_ratio 0.2 ^
  --eval_interval 500 ^
  --eval_count 100 ^
  --experiment_log_interval 1000 ^
  --checkpoint_iterations 10000 20000 30000
```

### 3. LAS 初始化 + 无深度筛选

```bat
python train.py ^
  -s <dataset_root> ^
  --las_file <dataset_root>\<scene_las> ^
  --metadata_path <dataset_root>\metadata.xml ^
  --depth_dir <depth_dir> ^
  --iterations 30000 ^
  --disable_cuda_depth_filter ^
  --eval_interval 500 ^
  --eval_count 100 ^
  --experiment_log_interval 1000 ^
  --checkpoint_iterations 10000 20000 30000
```

### 4. LAS 初始化 + 自适应深度筛选

```bat
python train.py ^
  -s <dataset_root> ^
  --las_file <dataset_root>\<scene_las> ^
  --metadata_path <dataset_root>\metadata.xml ^
  --depth_dir <depth_dir> ^
  --iterations 30000 ^
  --adaptive_depth_filter ^
  --adaptive_depth_ratio 0.2 ^
  --eval_interval 500 ^
  --eval_count 100 ^
  --experiment_log_interval 1000 ^
  --checkpoint_iterations 10000 20000 30000
```

说明：无深度筛选实验中仍然可以传入 `--depth_dir`，但必须加 `--disable_cuda_depth_filter`，这样日志中可以明确记录“有深度图但关闭筛选”的对照条件。

## 输出日志说明

训练过程中会输出两类主要日志。

### 实验日志

```text
[实验验证机制][ITER 10000] mode=unfiltered, adaptive_depth=off, ...
```

常见字段：

| 字段 | 含义 |
| --- | --- |
| `mode` | 当前训练视角是否实际启用 CUDA 深度筛选。 |
| `adaptive_depth` | 是否开启自适应深度筛选策略。 |
| `depth_decision` | 当前图片的深度筛选决策，如 `disabled`、`normal`、`hard_psnr`。 |
| `hard_images` | 当前被判定为困难图片的数量。 |
| `depth_status` | 深度图读取状态，常见为 `enabled`、`disabled`、`missing`。 |
| `depth_source` | 深度来源，常见为 `depth_file` 或 `none`。 |
| `selected_count` | 当前视角 rasterizer 后可见或保留的高斯数量。 |
| `total_gaussians` | 当前模型总高斯数量。 |
| `selected_ratio` | `selected_count / total_gaussians`。 |
| `valid_depth_ratio` | 当前深度图中有效深度像素比例。 |
| `valid_mask_ratio` | 当前监督中 mask 保留的有效像素比例。 |
| `psnr` | 当前训练图片的 masked PSNR。 |
| `loss` | 当前训练图片 loss。 |

### 固定验证日志

```text
[固定验证机制][ITER 30000] eval_count=100, eval_depth_filter=off, ...
```

固定验证始终关闭深度筛选，目的是比较训练出来的模型本身质量，而不是评估时再用深度图过滤结果。

## 主要代码改动

### LAS 初始化

原始 3DGS 使用 COLMAP 稀疏点云初始化高斯。本项目在 `train.py` 中增加 `--las_file` 和 `--metadata_path`。当提供 LAS 文件时，训练流程会：

1. 读取 `metadata.xml` 中的 `SRSOrigin`。
2. 将 LAS 原始坐标转换到局部坐标。
3. 使用 Open3D 进行体素下采样。
4. 保留或匹配颜色、法线。
5. 使用 KNN 距离估计初始高斯尺度。
6. 调用 `GaussianModel.create_from_las_data()` 初始化高斯。

### CUDA 深度筛选

深度筛选不是在 Python 中永久删除高斯，而是在每次 render 时把当前视角的深度范围传入 CUDA rasterizer。CUDA 侧会根据每个高斯投影位置与深度范围判断是否保留。

这样做的优点是：

- 不破坏全局高斯集合。
- 不同视角可以使用不同深度约束。
- 可以关闭筛选进行公平验证。

### 自适应困难图片筛选

全局深度筛选对所有图片都施加硬约束，容易造成有效梯度减少。自适应策略只对近期 PSNR 较低的图片启用深度筛选：

```text
每张图片维护 EMA PSNR -> 按 PSNR 从低到高排序 -> 选择最低的 adaptive_depth_ratio 比例作为 hard images
```

默认实验中：

```text
adaptive_depth_ratio = 0.2
```

### 固定验证

训练过程中每隔 `--eval_interval` 次迭代，固定选取 `--eval_count` 张相机进行验证。验证阶段关闭深度筛选：

```text
eval_depth_filter=off
```

这样可以避免“训练时用了深度筛选，验证时也用深度筛选”导致评价不公平。

### mask

mask 用于控制训练监督像素。当前代码会在不同 `-r` 分辨率下正确缩放 mask，并在 L1、SSIM 构造、PSNR 和固定验证中使用。

## 实验设计

核心实验为 2 x 2 对照：

| 编号 | 初始化方式 | 深度筛选 |
| --- | --- | --- |
| A | COLMAP sparse point cloud | 关闭 |
| B | COLMAP sparse point cloud | 自适应 CUDA 深度筛选 |
| C | LAS point cloud | 关闭 |
| D | LAS point cloud | 自适应 CUDA 深度筛选 |

评价指标包括：

- 训练过程固定验证 PSNR / loss / render time
- 离线渲染后的 PSNR
- SSIM
- LPIPS
- 最终高斯数量和训练时间趋势

## 实验结果

### HLSRedHouse / HLSGY 数据集

离线渲染后使用 `metrics.py` 计算的四组图像指标如下：

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 训练时间趋势 | 结论 |
| --- | ---: | ---: | ---: | --- | --- |
| COLMAP + 无深度筛选 | **20.635** | **0.658** | **0.445** | 约 58 分钟 | 图像质量最优 |
| COLMAP + 自适应深度筛选 | 20.014 | 0.651 | 0.447 | 约 68 分钟 | 深度筛选降低质量且变慢 |
| LAS + 无深度筛选 | 19.541 | 0.605 | 0.500 | 约 48 分钟 | LAS 路线更快，但质量低于 COLMAP |
| LAS + 自适应深度筛选 | 18.832 | 0.595 | 0.504 | 约 49 分钟 | 接近最快，但质量最差 |

训练过程固定验证的最终趋势也一致：

| 方法 | 最终固定验证 PSNR ↑ | 最终固定验证 loss ↓ | 观察 |
| --- | ---: | ---: | --- |
| COLMAP + 无深度筛选 | 约 20.5 | 约 0.119 | 收敛最好 |
| COLMAP + 自适应深度筛选 | 约 20.1 | 约 0.124 | 比无筛选差 |
| LAS + 无深度筛选 | 约 19.5-19.7 | 约 0.136 | 低于 COLMAP |
| LAS + 自适应深度筛选 | 约 19.1 | 约 0.143 | 深度筛选进一步降低质量 |

## 当前限制与后续工作

当前项目仍有几个限制：

- 尚未引入专门的几何指标，例如深度 RMSE、点云到 mesh/scan 的距离。
- 当前结论主要基于 HLSRedHouse/HLSGY 数据集，数据类型仍有限。
- LAS 初始化只使用简单体素下采样和 KNN 尺度估计，尚未针对建筑边缘、墙面、薄结构做优化。
- CUDA 深度筛选为硬筛选，缺少软权重或可微深度损失。
- 深度图质量和坐标对齐精度对结果影响很大，需要更系统的对齐验证工具。

后续可以尝试：

- 用软深度 loss 替代硬筛选。
- 对 LAS 初始化的尺度、opacity、颜色和法线做更细致设计。
- 加入深度误差和几何一致性指标。
- 用可视化误差热力图展示 COLMAP 与 LAS 的差异。
- 在 COLMAP 明显失败、弱纹理更严重、低分辨率图像更多的数据集上继续验证 LAS 的潜在优势。

## 致谢

本项目基于以下开源工作：

- [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [graphdeco-inria/diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [graphdeco-inria/simple-knn](https://gitlab.inria.fr/bkerbl/simple-knn)

如果使用本项目，请同时引用原始 3D Gaussian Splatting 工作。

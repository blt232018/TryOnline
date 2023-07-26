# &#x20;TryOn

## &#x20;项目简介

*   本项目基于几个 `Python` 深度学习的项目（详见引用部分）以及 [fastapi](https://fastapi.tiangolo.com/) 相关框架
*   由于训练模型数据仅包含上衣，所以当前仅支持试穿上衣
*   当前不支持自定义模特（仅允许随机切换现有模特数据）
*   由于个人开发环境仅有集成显卡，所以涉及 `pytorch` 相关未使用 `cuda`

### &#x20;为什么会有这个项目

前段时间参加一个面试，当时那个面试官希望基于 <https://github.com/sangyun884/HR-VITON> 开发一套服务商用。由于各种原因最后没有合作，但我利用闲暇时间整理了这个项目。一是最近离职找工作，确实时间相对充沛；二是上一份工作主要做RPA和后端开发，在后端中集成工具也比较实用；三是对深度学习知识了解很少，可以趁机学习一点相关知识。

### &#x20;项目涉及主要内容

*   fastapi 框架的基本使用
*   opencv、PIL、Array、bytes 对象之间的转换
*   pytorch 框架相关
*   命令行脚本工具转化为工具类（主要工作）

## &#x20;项目环境

目前仅集成试穿上衣，未集成自定义模特，环境相对简单。
*未集成自定义模特的原因见下方 **题外话***

### &#x20;基本环境

*   Linux 可选
*   python 3.10
    *   fastapi 0.100
    *   torch 2.0
    *   opencv-python 4.8
*   更多详见 env.yaml

### &#x20;关于 env.yaml

如果动手能力强可以手动安装所有依赖。如果只是想体验一下，推荐使用 conda 创建新环境（测试）。

```bash
conda env create -f env.yaml  # 使用 env.yaml 创建环境

```

##

## &#x20;项目目录结构

## &#x20;题外话

> 选择在`Linux`系统下开发，主要是因为 `detectron2` 不支持 `Windows OS`，如果后续集成相关功能减少不必要的麻烦

> 自定义试穿模特（比如使用个人照片）需要做很多额外的工作，正如 [HR-VITON](https://github.com/sangyun884/HR-VITON) 开发者在 [Preprocessing for HR-VITON](https://github.com/sangyun884/HR-VITON/blob/main/Preprocessing.md) 提到的，比如使用 [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) 生成 `key_points` json 数据（在linux环境下需要自己编译 Openpose）；然后是 `Human Parsing`，原文中基于 [CIHP\_PGN](https://github.com/Engineering-Course/CIHP_PGN) 实现，但是这个项目基于 `tensorflow` 实现的，所以如果你需要集成这个项目，你不得不同时安装两个大型框架，即使它们的功能在某种程度上是一样的。当然，你也可以参考 [TryYours-Virtual-Try-On](https://github.com/lastdefiance20/TryYours-Virtual-Try-On) 开发者的方案，使用 [Graphonomy](https://github.com/Gaoyiminggithub/Graphonomy) 实现 `Human Parsing`，我在测试过程中采用了这个方案，因为 `Graphonomy` 是基于 `pytorch` 实现的；还有接下来基于 `detectron2` 子模块 `densepose` 获取`Densepose` json数据（`detectron2`可能需要在本地编译安装，取决于 `pytorch` 版本，由于 `2.0` 版本官方未提供预编译包，只能手动编译安装）。除此之外，你需要把这些项目命令行接口封装成工具类，下载项目所需的模型、数据集（在大陆实现这些还是相当麻烦的）。


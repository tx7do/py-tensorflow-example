# py-tensorflow-example

## 安装TensorFlow

CPU版本：

```bash
pip3 install --upgrade tensorflow
```

GPU版本：

```bash
pip3 install --upgrade tensorflow-gpu
```

## 安装GPU依赖包

* [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)
* [cuDNN SDK](https://developer.nvidia.com/cudnn)

CUDA工具包默认安装在：`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6`

偷个懒，直接把 `cuDNN SDK` 和 TensorFlow C库都解压放到这下面去。

## 参考资料

* [TensorFlow GPU 支持](https://www.tensorflow.org/install/gpu)

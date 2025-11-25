from setuptools import setup
import os

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 手动指定CUDA架构，支持8.9（RTX 40系列）
# 设置环境变量以支持8.9架构
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.9')

setup(
    name='SOFTGROUP_OP',
    ext_modules=[
        CUDAExtension(
            'SOFTGROUP_OP', ['src/softgroup_api.cpp', 'src/softgroup_ops.cpp', 'src/cuda.cu'],
            extra_compile_args={
                'cxx': ['-g'],
                'nvcc': ['-O2', '-gencode', 'arch=compute_89,code=sm_89']  # 手动添加8.9架构支持
            })
    ],
    cmdclass={'build_ext': BuildExtension})

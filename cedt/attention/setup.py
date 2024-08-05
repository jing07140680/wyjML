from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='attention_cuda',
    ext_modules=[
        CUDAExtension(
            'attention_cuda', 
            ['attention.cu','attention_kernel.cu'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
})

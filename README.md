# tf-faster-rcnn-cpu
Xinlei Chen's tf-faster-rcnn work with cpu.
and changed by kidtic . 

## CPU运行方法
将tf-faster-rcnn 改成cpu运行需要更改3个地方：
1. 将lib/model/config.py 270行 “__C.USE_GPU_NMS = True” 改成“__C.USE_GPU_NMS = False”
2. 将lib/setup.py 注释掉CUDA = locate_cuda()
3. 将lib/setup.py 中以下代码段注释调
```
    Extension('nms.gpu_nms',
        ['nms/nms_kernel.cu', 'nms/gpu_nms.pyx'],
        library_dirs=[CUDA['lib64']],
        libraries=['cudart'],
        language='c++',
        runtime_library_dirs=[CUDA['lib64']],
        # this syntax is specific to this build system
        # we're only going to use certain compiler args with nvcc and not with gcc
        # the implementation of this trick is in customize_compiler() below
        extra_compile_args={'gcc': ["-Wno-unused-function"],
                            'nvcc': ['-arch=sm_52',
                                     '--ptxas-options=-v',
                                     '-c',
                                     '--compiler-options',
                                     "'-fPIC'"]},
        include_dirs = [numpy_include, CUDA['include']]
    )
```


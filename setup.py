import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Our example needs CUTLASS. Luckily it is header-only library, so all we need to do is include
# cutlass_dir = os.environ.get("CUTLASS_DIR", "")
cutlass_dir = "/root/cutlass_gemm/cutlass"
if not os.path.isdir(cutlass_dir):
  raise Exception("Environment variable CUTLASS_DIR must point to the CUTLASS installation") 
_cutlass_include_dirs = ["tools/util/include","include"]
cutlass_include_dirs = [os.path.join(cutlass_dir, d) for d in _cutlass_include_dirs]

# Set additional flags needed for compilation here
nvcc_flags=["-O3","-DNDEBUG","-std=c++17"]
ld_flags=[]

# For the hopper example, we need to specify the architecture. It also needs to be linked to CUDA library.
nvcc_flags += ["--generate-code=arch=compute_100a,code=[sm_100a]"] 
ld_flags += ["cuda"]

setup(
    name='cutlass_gemm',
    ext_modules=[
        CUDAExtension(
                name="cutlass_gemm",  
                sources=["cutlass_gemm.cu"],
                include_dirs=cutlass_include_dirs,
                extra_compile_args={'nvcc': nvcc_flags},
                libraries=ld_flags)
   ],
    cmdclass={
        'build_ext': BuildExtension
    })

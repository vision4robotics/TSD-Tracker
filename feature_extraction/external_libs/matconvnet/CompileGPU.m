addpath matlab
vl_compilenn('enableGpu',true,... 
             'cudaRoot', 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0', ...%自己安装的CUDA的路径
             'cudaMethod', 'nvcc');
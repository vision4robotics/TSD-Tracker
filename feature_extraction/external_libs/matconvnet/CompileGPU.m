addpath matlab
vl_compilenn('enableGpu',true,... 
             'cudaRoot', 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0', ...%�Լ���װ��CUDA��·��
             'cudaMethod', 'nvcc');
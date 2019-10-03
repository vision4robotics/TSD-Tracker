function TSD_Demo(save_dir)
close all;
clear;
clc;
setup_paths();
% vl_setupnn();
%%% Note that the default setting is CPU. TO ENABLE GPU, please recompile the MatConvNet toolbox
% vl_compilenn('enableGpu',true);
%     vl_compilenn('enableGpu',true,...
%                  'cudaRoot', 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0' ...%自己安装的CUDA的路径
%                   );
% global enableGPU;
% enableGPU = true;
% Load video information
seq = load_video_information('UAV123_10fps');% 'DTB70'
result = run_TSD(seq);
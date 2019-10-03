%   This function runs the BACF tracker on the video specified in "seq".
%   This function borrowed from BACF paper.
%   details of some parameters are not presented in the paper, you can
%   refer to BACF paper for more details.

function results = run_TSD(seq)

params.name = seq.name;
params.video_path = seq.video_path;

%   Grayscale feature parameters
grayscale_params.colorspace='gray';
grayscale_params.nDim = 1;

%%
%HOG feature parameters
hog_params.nDim  = 31;
%   CN feature parameters
cn_params.nDim = 11;
%   Gray feature parameters
gray_params.nDim = 1;
%   Saliency feature parameters
saliency_params.nDim = 3;

%   handcrafted parameters
Feat1 = 'cn';
switch Feat1
    case 'conv3'
        params.layerInd{1} = 3;
        params.nDim{1} = 256;
    case 'conv4'
        params.layerInd{1} = 2;
        params.nDim{1} = 512;
    case 'conv5'
        params.layerInd{1} = 1;
        params.nDim{1} = 512;
    case 'fhog'
        params.layerInd{1} = 0;
        params.nDim{1} = 31;
    case 'cn'
        params.layerInd{1} = 0;
        params.nDim{1} = 11;
    otherwise
        params.layerInd{1} = 0;
end
params.feat_type = Feat1;

params.t_global.type_assem = 'fhog_cn';
switch params.t_global.type_assem
    case 'fhog_cn_gray_saliency'
        handcrafted_params.nDim = hog_params.nDim + cn_params.nDim + gray_params.nDim + saliency_params.nDim;
    case 'fhog_cn_gray'
        handcrafted_params.nDim = hog_params.nDim + cn_params.nDim + gray_params.nDim;
    case 'fhog_gray_saliency'
        handcrafted_params.nDim = hog_params.nDim + gray_params.nDim + saliency_params.nDim;
    case 'fhog_gray'
        handcrafted_params.nDim = hog_params.nDim + gray_params.nDim;
    case 'fhog_cn'
        handcrafted_params.nDim = hog_params.nDim + cn_params.nDim;
end

params.t_features = {struct('getFeature_fhog',@get_fhog,...
    'getFeature_cn',@get_cn,...
    'getFeature_gray',@get_gray,...
    'getFeature_saliency',@get_saliency,...
    'getFeature_deep',@get_deep,...
    'getFeature_handcrafted',@get_handcrafted,...
    'hog_params',hog_params,...
    'cn_params',cn_params,...
    'gray_params',gray_params,...
    'saliency_params',saliency_params,...
    'handcrafted_params',handcrafted_params)};

params.t_global.w2c_mat = load('w2c.mat');
params.t_global.factor = 0.2; % for saliency
params.t_global.cell_size = 4;
params.t_global.cell_selection_thresh = 0.75^2;

%%

%   Search region + extended background parameters
params.search_area_shape = 'square';    % the shape of the training/detection window: 'proportional', 'square' or 'fix_padding'
params.search_area_scale = 5.2;           % the size of the training/detection area proportional to the target size
params.filter_max_area   = 50^2;        % the size of the training/detection area in feature grid cells

%   Learning parameters
params.learning_rate       = 0.03;        % learning rate
params.output_sigma_factor = 1/16;		% standard deviation of the desired correlation output (proportional to target)

%   Detection parameters
params.interpolate_response  = 4;
params.newton_iterations     = 50;

%   Scale parameters
params.number_of_scales =  5;
params.scale_step       = 1.015;

%   size, position, frames initialization
params.wsize    = [seq.init_rect(1,4), seq.init_rect(1,3)];
params.init_pos = [seq.init_rect(1,2), seq.init_rect(1,1)] + floor(params.wsize/2);
params.s_frames = seq.s_frames;
params.no_fram  = seq.en_frame - seq.st_frame + 1;
params.seq_st_frame = seq.st_frame;
params.seq_en_frame = seq.en_frame;

%   ADMM parameters, # of iteration, and lambda- mu and betha are set in
%   the main function.
params.admm_iterations = 2;
params.admm_lambda = 0.01;

%   Debug and visualization
params.visualization = 1;


params.nSamples = 50;                  % Maximal number of samples in memory 200
params.sample_reg = 4.8;                % Weights regularization (mu)
params.sample_burnin = 10;              % Number of frames before weight optimization starts
params.num_acs_iter = 1;                % Number of Alternate Convex Search iterations
params.sample_replace_strategy = 'constant_tail';
params.lt_size = 10;
params.nu = 0.18;
params.single_DPMR = 14;
params.rate = 0.225;

%   Run the main function
results = tracker(params);


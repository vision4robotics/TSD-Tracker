% This function implements the BACF tracker.
function [results] = tracker(params)
%%

% nDim = p.nDim;
feat_type = params.feat_type;
layerInd = params.layerInd;
rate = params.rate;

%   Setting parameters for local use.
search_area_scale   = params.search_area_scale;
output_sigma_factor = params.output_sigma_factor;
learning_rate       = params.learning_rate;
filter_max_area     = params.filter_max_area;
nScales             = params.number_of_scales;
scale_step          = params.scale_step;
interpolate_response = params.interpolate_response;

features    = params.t_features;
video_path  = params.video_path;
s_frames    = params.s_frames;
pos         = floor(params.init_pos);
target_sz   = floor(params.wsize);

visualization  = params.visualization;
num_frames     = params.no_fram;
init_target_sz = target_sz;

%set the feature ratio to the feature-cell size
featureRatio = params.t_global.cell_size;
search_area = prod(init_target_sz / featureRatio * search_area_scale);

% when the number of cells are small, choose a smaller cell size
if isfield(params.t_global, 'cell_selection_thresh')
    if search_area < params.t_global.cell_selection_thresh * filter_max_area
        params.t_global.cell_size = min(featureRatio, max(1, ceil(sqrt(prod(init_target_sz * search_area_scale)/(params.t_global.cell_selection_thresh * filter_max_area)))));
        
        featureRatio = params.t_global.cell_size;
        search_area = prod(init_target_sz / featureRatio * search_area_scale);
    end
end

global_feat_params = params.t_global;

if search_area > filter_max_area
    currentScaleFactor = sqrt(search_area / filter_max_area);
else
    currentScaleFactor = 1.0;
end

% target size at the initial scale
base_target_sz = target_sz / currentScaleFactor;

% window size, taking padding into account
switch params.search_area_shape
    case 'proportional'
        sz = floor( base_target_sz * search_area_scale);     % proportional area, same aspect ratio as the target
    case 'square'
        sz = repmat(sqrt(prod(base_target_sz * search_area_scale)), 1, 2); % square area, ignores the target aspect ratio
    case 'fix_padding'
        sz = base_target_sz + sqrt(prod(base_target_sz * search_area_scale) + (base_target_sz(1) - base_target_sz(2))/4) - sum(base_target_sz)/2; % const padding
    otherwise
        error('Unknown "params.search_area_shape". Must be ''proportional'', ''square'' or ''fix_padding''');
end
% set the size to exactly match the cell size
sz = round(sz / featureRatio) * featureRatio;
use_sz = floor(sz/featureRatio);

% construct the label function- correlation output, 2D gaussian function,
% with a peak located upon the target
output_sigma = sqrt(prod(floor(base_target_sz/featureRatio))) * output_sigma_factor;
rg           = circshift(-floor((use_sz(1)-1)/2):ceil((use_sz(1)-1)/2), [0 -floor((use_sz(1)-1)/2)]);
cg           = circshift(-floor((use_sz(2)-1)/2):ceil((use_sz(2)-1)/2), [0 -floor((use_sz(2)-1)/2)]);
[rs, cs]     = ndgrid( rg,cg);
y            = exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
yf           = fft2(y); %   FFT of y.

if interpolate_response == 1
    interp_sz = use_sz * featureRatio;
else
    interp_sz = use_sz;
end

% construct cosine window
cos_window = single(hann(use_sz(1))*hann(use_sz(2))');

% Calculate feature dimension
try
    im = imread([video_path '/img/' s_frames{1}]);
catch
    try
        im = imread(s_frames{1});
    catch
        im = imread([video_path '/' s_frames{1}]);
    end
end
if size(im,3) == 3
    if all(all(im(:,:,1) == im(:,:,2)))
        colorImage = false;
    else
        colorImage = true;
    end
else
    colorImage = false;
end

% compute feature dimensionality
feature_dim = 0;

if size(im,3) > 1 && colorImage == false
    im = im(:,:,1);
end

if nScales > 0
    scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2));
    scaleFactors = scale_step .^ scale_exp;
    min_scale_factor = scale_step ^ ceil(log(max(5 ./ sz)) / log(scale_step));
    max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));
end

if interpolate_response >= 3
    % Pre-computes the grid that is used for socre optimization
    ky = circshift(-floor((use_sz(1) - 1)/2) : ceil((use_sz(1) - 1)/2), [1, -floor((use_sz(1) - 1)/2)]);
    kx = circshift(-floor((use_sz(2) - 1)/2) : ceil((use_sz(2) - 1)/2), [1, -floor((use_sz(2) - 1)/2)])';
    newton_iterations = params.newton_iterations;
end

rect_positions = zeros(num_frames, 4);
time = 0;

% allocate memory for multi-scale tracking
multires_pixel_template = zeros(sz(1), sz(2), size(im,3), nScales, 'uint8');
small_filter_sz = floor(base_target_sz/featureRatio);

prior_weights = [];
sample_weights = [];
latest_ind = [];
sample_frame = nan(params.nSamples,1);
samplesf = 1i*zeros(params.nSamples,use_sz(1),use_sz(2),features{1}.cn_params.nDim,'single');
sampleyf = permute(repmat(yf,[1,1,params.nSamples]),[3,1,2]);

all_currentScaleFactor = zeros(params.seq_en_frame - params.seq_st_frame + 1,1);
all_pos = zeros(params.seq_en_frame - params.seq_st_frame + 1,2);
all_replace_frame = zeros(params.seq_en_frame - params.seq_st_frame + 1,1);

frame = 1;
single_DPMR = 0;
single_MAX = 0;
num_of_keyframe = 1;
keyframe(1,1) = 1;
DPMR = zeros(params.nSamples,1);
all_DPMR = zeros(numel(s_frames),1);
all_MAX = zeros(numel(s_frames),1);

%%
for total_frame = 1:numel(s_frames)
    
    %load image
    try
        im = imread([video_path '/img/' s_frames{total_frame}]);
    catch
        try
            im = imread([s_frames{total_frame}]);
        catch
            im = imread([video_path '/' s_frames{total_frame}]);
        end
    end
    if size(im,3) > 1 && colorImage == false
        im = im(:,:,1);
    end
    
    tic();
    %%
    if frame > 1
        for scale_ind = 1:nScales  %pixels = get_pixels(im,pos,round(sz*currentScaleFactor),sz);
            multires_pixel_template(:,:,:,scale_ind) = ...
                get_pixels(im, pos, round(sz*currentScaleFactor*scaleFactors(scale_ind)), sz);
        end
        xtf = fft2(bsxfun(@times,get_features(multires_pixel_template,features,global_feat_params,feat_type, layerInd),cos_window));
        responsef = permute(sum(bsxfun(@times, conj(g_f), xtf), 3), [1 2 4 3]);
        
        % if we undersampled features, we want to interpolate the
        % response so it has the same size as the image patch
        if interpolate_response == 2
            % use dynamic interp size
            interp_sz = floor(size(y) * featureRatio * currentScaleFactor);
        end
        responsef_padded = resizeDFT2(responsef, interp_sz);
        
        % response in the spatial domain
        response = ifft2(responsef_padded, 'symmetric');
        % find maximum peak
        if interpolate_response == 3
            error('Invalid parameter value for interpolate_response');
        elseif interpolate_response == 4
            [disp_row, disp_col, sind] = resp_newton(response, responsef_padded, newton_iterations, ky, kx, use_sz);
        else
            [row, col, sind] = ind2sub(size(response), find(response == max(response(:)), 1));
            disp_row = mod(row - 1 + floor((interp_sz(1)-1)/2), interp_sz(1)) - floor((interp_sz(1)-1)/2);
            disp_col = mod(col - 1 + floor((interp_sz(2)-1)/2), interp_sz(2)) - floor((interp_sz(2)-1)/2);
        end
        % calculate translation
        switch interpolate_response
            case 0
                translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor * scaleFactors(sind));
            case 1
                translation_vec = round([disp_row, disp_col] * currentScaleFactor * scaleFactors(sind));
            case 2
                translation_vec = round([disp_row, disp_col] * scaleFactors(sind));
            case 3
                translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor * scaleFactors(sind));
            case 4
                translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor * scaleFactors(sind));
        end
        
        % set the scale
        currentScaleFactor = currentScaleFactor * scaleFactors(sind);
        % adjust to make sure we are not to large or to small
        if currentScaleFactor < min_scale_factor
            currentScaleFactor = min_scale_factor;
        elseif currentScaleFactor > max_scale_factor
            currentScaleFactor = max_scale_factor;
        end
        
        % update position
        %         old_pos = pos;
        pos = pos + translation_vec;
        time = time + toc();
        %%
        %load image
        if frame > params.nSamples+1
            try
                replace_im = imread([video_path '/img/' s_frames{replace_frame}]);
            catch
                try
                    replace_im = imread([s_frames{replace_frame}]);
                catch
                    replace_im = imread([video_path '/' s_frames{replace_frame}]);
                end
            end
            if size(replace_im,3) > 1 && colorImage == false
                replace_im = replace_im(:,:,1);
            end
            replace_pixels = get_pixels(replace_im,all_pos(replace_frame,:),round(sz*all_currentScaleFactor(replace_frame)),sz);
            replace_xf = fft2(bsxfun(@times,get_features(replace_pixels,features,global_feat_params,feat_type, layerInd),cos_window));
            replace_responsef = sum(bsxfun(@times, conj(g_f), replace_xf), 3);
            replace_responsef_padded = resizeDFT2(replace_responsef, interp_sz);
            replace_response = ifft2(replace_responsef_padded, 'symmetric');
        end
        if visualization == 1
            if frame == 2
                figure(2);
                set(gcf,'unit','normalized','position',[0,0,1,1]);
                subplot(3,4,1);im_handle1 = imshow(pixels);title('Patch');
                subplot(3,4,2);im_handle2 = surf(fftshift(response(:,:,sind)), 'FaceColor','interp','EdgeColor','none');title('Response map');colormap('jet');
            else
                set(im_handle1, 'CData', pixels);
                set(im_handle2, 'zdata', fftshift(response(:,:,sind)));
                if frame > params.nSamples+1
                    subplot(3,4,3);im_handle3 = imshow(replace_pixels);title('Discarded sample');
                    text_replace_frame = text(10, 10,['Frame : ' int2str(replace_frame)]);
                    set(text_replace_frame, 'color', [0 1 1]);
                    subplot(3,4,4);im_handle4 = surf(fftshift(replace_response), 'FaceColor','interp','EdgeColor','none');title('Response map of discarded sample');colormap('jet');
                end
                if frame > params.nSamples++2
                    set(im_handle3, 'CData', replace_pixels);
                    set(im_handle4, 'zdata', fftshift(replace_response));
                    set(text_replace_frame, 'string', ['Frame : ' int2str(replace_frame)]);
                end
            end
            drawnow
        end
        
        %%
        tic();
        A = response(:,:,sind);
        single_DPMR = dpmr(A,rate);
        all_DPMR(total_frame,1) = single_DPMR;
        all_MAX(total_frame,1) = max(max(A));
    end
    %%
    % Update the prior weights
    [prior_weights, replace_ind] = update_prior_weights(prior_weights, sample_weights, latest_ind, frame, params);
    latest_ind = replace_ind;
    replace_frame = sample_frame(replace_ind);
    sample_frame(replace_ind) = total_frame;
    
    % Initialize the weight for the new sample
    if frame == 1
        sample_weights = prior_weights;
    else
        % ensure that the new sample always get its current prior weight
        new_sample_weight = learning_rate;
        sample_weights = sample_weights * (1 - new_sample_weight) / (1 - sample_weights(replace_ind));
        sample_weights(replace_ind) = new_sample_weight;
        sample_weights = sample_weights / sum(sample_weights);
    end
    
    % extract training sample image region
    pixels = get_pixels(im,pos,round(sz*currentScaleFactor),sz);
    
    % extract features and do windowing
    xf = fft2(bsxfun(@times,get_features(pixels,features,global_feat_params,feat_type, layerInd),cos_window));
    
    samplesf(replace_ind,:,:,:) = xf;
    %%
    if frame<=70
        standard_DPMR = params.single_DPMR;
    elseif frame>70 && frame<=75
        standard_DPMR = params.single_DPMR-1;
    elseif frame>75 && frame<=80
        standard_DPMR = params.single_DPMR-2;
    elseif frame>80 && frame<=85
        standard_DPMR = params.single_DPMR-3;
    elseif frame>85 && frame<=90
        standard_DPMR = params.single_DPMR-4;
    end
    
    if ((single_DPMR>=standard_DPMR) || total_frame ==1 || frame>90) && total_frame - max(keyframe) >30
        sample_frame = nan(params.nSamples,1);
        keyframe(num_of_keyframe,1) = total_frame;
        num_of_keyframe = num_of_keyframe + 1;
        single_keyframe = permute(sum(sample_weights .* samplesf,1),[2,3,4,1]);
        samplesf = 1i*zeros(params.nSamples,use_sz(1),use_sz(2),features{1}.cn_params.nDim,'single');
        samplesf(1,:,:,:) = permute(single_keyframe,[4,1,2,3]);
        [prior_weights, replace_ind] = update_prior_weights([], [], [], 1, params);
        sample_weights = prior_weights;
        latest_ind = replace_ind;
        replace_frame = sample_frame(replace_ind);
        sample_frame(replace_ind) = total_frame;
        DPMR = [];
        frame =1;
    end
    %%
    for acs_iter = 1 : params.num_acs_iter
        g_f = single(zeros(size(xf)));
        h_f = g_f;
        l_f = g_f;
        mu    = 1;
        betha = 10;
        mumax = 10000;
        num_caculate = min(frame, params.nSamples);
        A = bsxfun(@times, sample_weights(1:num_caculate,:,:,:), samplesf(1:num_caculate,:,:,:));
        
        for i = 1:params.admm_iterations
            g_f = (1./(permute(sum(bsxfun(@times, A, conj(samplesf(1:num_caculate,:,:,:))),1),[2 3 4 1]) + 0.5 * mu)) ...
                .* (permute(sum(bsxfun(@times, A , conj(sampleyf(1:num_caculate,:,:,:))),1),[2 3 4 1]) - l_f + 0.5 * mu .* h_f);
            
            h_f = bsxfun(@times,(1/(mu + params.admm_lambda)), (2 .* l_f + mu .* g_f));
            
            h = ifft2(h_f);
            [sx,sy,h] = get_subwindow_no_window(h, floor(use_sz/2) , small_filter_sz);
            t = single(zeros(use_sz(1), use_sz(2), size(h,3)));
            t(sx,sy,:) = h;
            h_f = fft2(t);
            
            %   update L
            l_f = l_f + (mu * (g_f - h_f));
            
            %   update mu- betha = 10.
            mu = min(betha * mu, mumax);
            %             i = i+1;
        end
        
        if frame > params.sample_burnin
            sample_loss = compute_loss(g_f, samplesf, sampleyf, params);
            responsef = sum(bsxfun(@times,permute(conj(g_f),[4 1 2 3]), samplesf), 4);
            if interpolate_response == 2
                interp_sz = floor(size(y) * featureRatio * currentScaleFactor);
            end
            for k = 1:params.nSamples
                responsef_padded = resizeDFT2(permute(responsef(k,:,:),[2,3,1]), interp_sz);
                response = ifft2(responsef_padded, 'symmetric');
                DPMR(k,1) = dpmr(response,rate);
            end
            sample_weights = update_weights_v2(sample_loss, prior_weights, frame, params, params.sample_reg, DPMR);
        else
            sample_weights = prior_weights;
        end
    end
    target_sz = floor(base_target_sz * currentScaleFactor);
    
    rect_position = [pos([2,1]) - floor(target_sz([2,1])/2), target_sz([2,1])];
    rect_positions(total_frame,:) = rect_position;
    
    all_currentScaleFactor(total_frame,:) = currentScaleFactor;
    all_pos(total_frame,:) = pos;
    all_replace_frame(total_frame,:) = replace_frame;
    time = time + toc();
    %%
    if visualization == 1
        plot_frames = nan(num_frames,1); plot_sample = nan(num_frames,1); plot_prior = nan(num_frames,1);
        max_ind = min(frame, params.nSamples);
        [sorted_frames, ind] = sort(sample_frame(1:max_ind));
        plot_frames(sorted_frames) = sorted_frames;
        plot_sample(sorted_frames) = sample_weights(ind);
        plot_prior(sorted_frames) = prior_weights(ind);
        
        % Sample weights
        if frame == 1
            figure(2);
            subplot(3,4,[7,8,11,12]); im_handle6 = plot(plot_frames, plot_sample, 'xb-', 'linewidth',1.5, 'markersize', 3);
            hold on;
            subplot(3,4,[7,8,11,12]); im_handle7 = plot(keyframe,zeros(size(keyframe)),'g*','linewidth',2);
            hold off;
            title('Sample scores');
            legend({ 'Sample scores','keyframe'}, 'location', 'northwest');
            axis([1 num_frames 0 1.3*max([sample_weights; prior_weights])]);
        else
            subplot(3,4,[7,8,11,12]); im_handle6 = plot(plot_frames, plot_sample, 'xb-', 'linewidth',1.5, 'markersize', 3);
            hold on;
            subplot(3,4,[7,8,11,12]); im_handle7 = plot(keyframe,zeros(size(keyframe)),'g*','linewidth',2);
            hold off;
            title('Sample scores');
            legend({ 'Sample scores','keyframe'}, 'location', 'northwest');
            axis([1 num_frames 0 1.3*max([sample_weights; prior_weights])]);
        end
        
        plot_frames = nan(num_frames,1); plot_sample = nan(num_frames,1); plot_prior = nan(num_frames,1);
        max_ind = min(frame, params.nSamples);
        [sorted_frames, ind] = sort(sample_frame(1:max_ind));
        plot_frames(sorted_frames) = sorted_frames;
        plot_sample(sorted_frames) = sample_weights(ind);
        plot_prior(sorted_frames) = prior_weights(ind);
        
        if frame == 1   %first frame, create GUI
            figure(2);
            subplot(3, 4, [5,6,9,10]);
            im_handle = imshow(uint8(im), 'Border','tight', 'InitialMag', 100 + 100 * (length(im) < 500));
            rect_handle = rectangle('Position',rect_position, 'EdgeColor','g', 'LineWidth',2);
            text_handle = text(10, 20,['Frame : ' int2str(total_frame) ' / ' int2str(num_frames)]);
            set(text_handle, 'color', [0 1 1]);
            text_fps = text(10, 60, ['FPS : ' num2str(1/(time/total_frame))]);
            set(text_fps, 'color', [0 1 1]);
            title('Result');
        else
            try  %subsequent frames, update GUI
                set(im_handle, 'CData', im);
                set(rect_handle, 'Position', rect_position);
                %                 set(rect_handle2, 'Position', rect_position_padded);
                set(text_handle, 'string', ['Frame : ' int2str(total_frame) ' / ' int2str(num_frames)]);
                set(text_fps, 'string', ['FPS : ' num2str(total_frame / time)]);
            catch
                return
            end
        end
        
        drawnow
    end
    %%
    frame = frame + 1;
end
%   save resutls.
fps = total_frame / time;
results.type = 'rect';
results.res = rect_positions;
results.fps = fps;
results.keyframe = keyframe;
results.DPMR = all_DPMR;
results.MAX = all_MAX;
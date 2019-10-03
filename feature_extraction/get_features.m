function [feature_pixels, support_sz] = get_features(image, features, gparams, feat, layerInd)

if ~ iscell(features)
    features = {features};
end;

[im_height, im_width, num_im_chan, num_images] = size(image);

colorImage = num_im_chan == 3;


% %compute total dimension of all features
% tot_feature_dim = 0;
% for n = 1:length(features)
%     
%     if ~isfield(features{n}.fparams,'useForColor')
%         features{n}.fparams.useForColor = true;
%     end;
%     
%     if ~isfield(features{n}.fparams,'useForGray')
%         features{n}.fparams.useForGray = true;
%     end;
%     
%     if (features{n}.fparams.useForColor && colorImage) || (features{n}.fparams.useForGray && ~colorImage)
%         tot_feature_dim = tot_feature_dim + features{n}.fparams.nDim;
%     end;
%     
% end;
% 
% if nargin < 4 || isempty(fg_size)
%     if gparams.cell_size == -1
%         fg_size = size(features{1}.getFeature(image,features{1}.fparams,gparams));
%     else
%         fg_size = [floor(im_height/gparams.cell_size), floor(im_width/gparams.cell_size)];
%     end
% end
% 
% % temporary hack for fixing deep features
% if gparams.cell_size == -1
%     cf = features{1};
%     if (cf.fparams.useForColor && colorImage) || (cf.fparams.useForGray && ~colorImage)
%         [feature_pixels, support_sz] = cf.getFeature(image,cf.fparams,gparams);
%     end;
% else
%     %compute the feature set
%     feature_pixels = zeros(fg_size(1),fg_size(2),tot_feature_dim, num_images, 'single');
%     
%     currDim = 1;
%     for n = 1:length(features)
%         cf = features{n};
%         if (cf.fparams.useForColor && colorImage) || (cf.fparams.useForGray && ~colorImage)
%             feature_pixels(:,:,currDim:(currDim+cf.fparams.nDim-1),:) = cf.getFeature(image,cf.fparams,gparams);
%             currDim = currDim + cf.fparams.nDim;
%         end;
%     end;
%     support_sz = [im_height, im_width];
% end
switch feat
    case 'fhog'
        fg_size = [floor(im_height/gparams.cell_size), floor(im_width/gparams.cell_size)];
        feature_pixels = zeros(fg_size(1),fg_size(2),features{1}.hog_params.nDim, num_images, 'single');
        feature_pixels(:,:,1:features{1}.hog_params.nDim,:) = features{1}.getFeature_fhog(image,features{1}.hog_params,gparams);
    case 'cn'
        fg_size = [floor(im_height/gparams.cell_size), floor(im_width/gparams.cell_size)];
        feature_pixels = zeros(fg_size(1),fg_size(2),features{1}.cn_params.nDim, num_images, 'single');
        feature_pixels(:,:,1:features{1}.cn_params.nDim,:) = features{1}.getFeature_cn(image,features{1}.cn_params,gparams);
    case 'gray'
        fg_size = [floor(im_height/gparams.cell_size), floor(im_width/gparams.cell_size)];
        feature_pixels = zeros(fg_size(1),fg_size(2),features{1}.gray_params.nDim, num_images, 'single');
        feature_pixels(:,:,1,:) = features{1}.getFeature_gray(image,features{1}.gray_params,gparams);
    case 'saliency'
        fg_size = [floor(im_height/gparams.cell_size), floor(im_width/gparams.cell_size)];
        feature_pixels = zeros(fg_size(1),fg_size(2),features{1}.saliency_params.nDim, num_images, 'single');
        feature_pixels(:,:,1:features{1}.saliency_params.nDim,:) = features{1}.getFeature_saliency(image,features{1}.saliency_params,gparams);
    case 'handcrafted_assem'
        fg_size = [floor(im_height/gparams.cell_size), floor(im_width/gparams.cell_size)];
        feature_pixels = zeros(fg_size(1),fg_size(2),features{1}.handcrafted_params.nDim, num_images, 'single');
        feature_pixels(:,:,1:features{1}.handcrafted_params.nDim,:) = features{1}.getFeature_handcrafted(image,features{1}.handcrafted_params,gparams);
    case 'conv3'
        fg_size = [floor(im_height/gparams.cell_size), floor(im_width/gparams.cell_size)];
        feature_pixels = zeros(fg_size(1),fg_size(2),features{1}.deep_params.nDim(layerInd), num_images, 'single');
        feature_pixels(:,:,1:features{1}.deep_params.nDim(layerInd),:) = features{1}.getFeature_deep(image,features{1}.deep_params,gparams,layerInd);
    case 'conv4'
        fg_size = [floor(im_height/gparams.cell_size), floor(im_width/gparams.cell_size)];
        feature_pixels = zeros(fg_size(1),fg_size(2),features{1}.deep_params.nDim(layerInd), num_images, 'single');
        feature_pixels(:,:,1:features{1}.deep_params.nDim(layerInd),:) = features{1}.getFeature_deep(image,features{1}.deep_params,gparams,layerInd);
    case 'conv5'
        fg_size = [floor(im_height/gparams.cell_size), floor(im_width/gparams.cell_size)];
        feature_pixels = zeros(fg_size(1),fg_size(2),features{1}.deep_params.nDim(layerInd), num_images, 'single');
        feature_pixels(:,:,1:features{1}.deep_params.nDim(layerInd),:) = features{1}.getFeature_deep(image,features{1}.deep_params,gparams,layerInd);
    case 'deep_assem'
        fg_size = [floor(im_height/gparams.cell_size), floor(im_width/gparams.cell_size)];
        Dim = features{1}.deep_params.nDim(1) + features{1}.deep_params.nDim(2) + features{1}.deep_params.nDim(3);
        feature_pixels = zeros(fg_size(1),fg_size(2), Dim, num_images, 'single');
        A = cell(3,1);
        for ii = 1:3
            temp = features{1}.getFeature_deep(image,features{1}.deep_params,gparams,ii);
            A{ii} = temp;
        end
        temp2 = cat(3,A{1},A{2});
        feature_pixels(:,:,1:Dim,:) = cat(3,temp2,A{3});
end
support_sz = [im_height, im_width];
end
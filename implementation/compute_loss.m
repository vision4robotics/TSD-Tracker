function [sample_loss] = compute_loss(g_f, samplesf, yf, params)

% Compute the training loss for each sample
nSamples = params.nSamples;
support_sz = numel(g_f(:,:,1));

corr_train = sum(permute(conj(g_f),[4 1 2 3]).*samplesf,4);

corr_error = yf - corr_train;

% error_temp = reshape(corr_error,[nSamples, 1, support_sz]);
% L = 1/support_sz * real(sum(error_temp .* conj(error_temp), 3));
% sample_loss = L;

L = zeros(nSamples,1);
for i = 1:nSamples
    L(i,:) = (norm(permute(corr_error(i,:,:),[2,3,1]),'fro'))^2;
end
% sample_loss = 1 / size(L,1) * L;
sample_loss = 1/support_sz * L;

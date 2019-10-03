% function [ sample_weights ] = update_weights_v2(sample_loss, prior_weights, frame, params, sample_reg,PSR, PPSR)
function [ sample_weights ] = update_weights_v2(sample_loss, prior_weights, frame, params, sample_reg,PPSR)
% Update the sample weights by solving the quadratic programming problem

% compute number of existing samples
num_samples = min(frame, params.nSamples);

% Set up the QP problem
% H = diag(2./(prior_weights(1:num_samples) * sample_reg + params.nu * PSR(1:num_samples) .* (PPSR(1:num_samples)/sum(PPSR(1:num_samples)))));
H = diag(2./(prior_weights(1:num_samples) * sample_reg + params.nu * 1./PPSR(1:num_samples)));

% H = diag(2./(prior_weights(1:num_samples) * sample_reg));
% H = diag(2./(prior_weights(1:num_samples) * sample_reg .* (APCE(1:num_samples)/sum(APCE(1:num_samples)))));
sample_loss = sample_loss(1:num_samples);
constraint = -eye(num_samples);
b = zeros(num_samples,1);
Aeq = ones(1,num_samples);
Beq = 1;
options.Display = 'off';
% sample_loss = sample_loss./sum(sample_loss);
% Do the QP optimization
sample_weights = quadprog(double(H),double(sample_loss),constraint,b,Aeq,Beq,[],[],[],options);

if frame < params.nSamples
    sample_weights = cat(1, sample_weights, zeros(params.nSamples - frame,1));
end;

end


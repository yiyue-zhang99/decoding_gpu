function [cos_d, distances, time_dec] = decoding_dynamics_time_course_vis(data, params)
% Time-resolved, cross-validated Mahalanobis decoding used for Figure 2d/e.
%
% data: trial x channel x time
% Required params:
%   theta, time_dat, toi, n_folds, reps, steps, span, w_length,
%   s_factor, hz

validateattributes(data, {'numeric'}, {'real', '3d', 'nonempty'});
validateattributes(params.theta, {'numeric'}, ...
    {'vector', 'real', 'finite', 'numel', size(data, 1)});
validateattributes(params.time_dat, {'numeric'}, ...
    {'vector', 'real', 'finite', 'numel', size(data, 3)});

theta = params.theta(:);
time_dat = params.time_dat(:)';

window_samples = round(params.w_length * params.hz / 1000);
span_samples = round(params.span * params.hz / 1000);
step_seconds = params.steps / 1000;
eligible = find(time_dat > params.toi(1) & time_dat <= params.toi(2));


% Construct the requested decoding times in seconds and map them to the
% nearest samples. This also supports step sizes that do not correspond to
% an integer number of samples.
requested_times = time_dat(eligible(1)):step_seconds:time_dat(eligible(end));
time_indices = zeros(size(requested_times));
for k = 1:numel(requested_times)
    [~, time_indices(k)] = min(abs(time_dat - requested_times(k)));
end
time_indices = unique(time_indices, 'stable');

if time_indices(1) - window_samples + 1 < 1
    error(['The first decoding time does not have params.w_length ms of ', ...
        'preceding data. Move params.toi(1) later.']);
end

time_dec = time_dat(time_indices);
n_segments = window_samples / span_samples;
dat_dec = nan(size(data, 1), size(data, 2) * n_segments, numel(time_indices));

for t = 1:numel(time_indices)
    ind = time_indices(t);
    window_data = data(:, :, ind-window_samples+1:ind);
    window_data = window_data - mean(window_data, 3);
    window_data = movmean(window_data, span_samples, 3, ...
        'Endpoints', 'discard');
    window_data = window_data(:, :, 1:span_samples:end);
    dat_dec(:, :, t) = reshape(window_data, ...
        size(window_data, 1), size(window_data, 2) * size(window_data, 3));
end

n_conditions = numel(unique(theta));
rep_cos = nan(params.reps, size(dat_dec, 3));
rep_distances = nan(params.reps, n_conditions, size(dat_dec, 3));

for rep = 1:params.reps
    [trial_cos, trial_distances] = ...
        mahal_func_theta_kfold_b(dat_dec, theta, params.n_folds);
    rep_cos(rep, :) = mean(trial_cos, 1, 'omitnan');
    rep_distances(rep, :, :) = squeeze(mean(trial_distances, 2, 'omitnan'));
end

cos_d = mean(rep_cos, 1, 'omitnan');
smooth_window = max(1, round(5 * params.s_factor / params.steps));
if smooth_window > 1
    cos_d = smoothdata(cos_d, 2, 'gaussian', smooth_window);
end
distances = squeeze(mean(rep_distances, 1, 'omitnan'));
end

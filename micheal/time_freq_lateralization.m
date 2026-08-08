% Time-frequency lateralization analysis
% Adapted from micheal_time_freq.m for the MemImp3_mem_whole dataset.
%
% Per-subject output (results/freq/sub##.mat):
%   lat_trials  : [n_trials x n_freq x n_time]  contra - ipsi (dB, log10)
%                 contralateral minus ipsilateral to the early-tested item
%   sess_label  : [n_trials x 1]  1 = sess1, 2 = sess2
%   incl1/incl2 : good-trial indices per session
%   cue_loc     : cue-location flag (Results col 3; 1 = early item on right)
%   frequencies : [1 x n_freq] Hz
%   plot_time   : [1 x n_time] ms
%
% Group figure: subject-average contra-ipsi lateralisation with
%               cluster-permutation significance contours.

close all
clear all

main_dir = '/home/dilay/project2/tw';
dat_dir  = '/home/dilay/project2/tw/Data_whole';
save_dir = '/home/dilay/project2/tw/results/freq';

if ~exist(save_dir, 'dir'); mkdir(save_dir); end

addpath(genpath(main_dir))
addpath(fullfile(main_dir, 'fieldtrip'))
ft_defaults

left_ch  = {'P7','P5','P3','P1','PO7','PO3','O1'};
right_ch = {'P8','P6','P4','P2','PO8','PO4','O2'};

frequencies = 6:0.5:16;   % Hz, 0.5-Hz steps

% Pre-allocate group array using time axis from subject 1
tmp       = load(fullfile(dat_dir, 'MemImp3_mem_whole_sess1_1.mat'));
tmp_toi   = find(tmp.ft_mem.time >= -0.1 & tmp.ft_mem.time <= 4.8);
toi_vec   = tmp.ft_mem.time(tmp_toi(1:5:end));   % shared toi (same for all subs)
plot_time = toi_vec * 1000;                        % ms
n_toi     = numel(toi_vec);
n_freq    = numel(frequencies);
clear tmp tmp_toi

freq_lat_sub = nan(19, n_freq, n_toi);   % subject-average for group plot

% ── Start parallel pool ───────────────────────────────────────────────────────
if isempty(gcp('nocreate'))
    parpool('local', 4);
end
warning('off', 'all')

% ── Per-subject loop (parallel) ───────────────────────────────────────────────
parfor sub = 1:19
    sub_file = fullfile(save_dir, sprintf('sub%02d.mat', sub));

    % Skip if already computed
    if exist(sub_file, 'file')
        fprintf('Subject %d: cached, skipping\n', sub)
        tmp_sub = load(sub_file, 'lat_trials');
        freq_lat_sub(sub, :, :) = mean(tmp_sub.lat_trials, 1);
        continue
    end

    fprintf('Subject %d\n', sub)

    lat_avg = process_subject(sub, dat_dir, sub_file, ...
                              frequencies, toi_vec, left_ch, right_ch, ...
                              plot_time);
    freq_lat_sub(sub, :, :) = lat_avg;
end

% ── Cluster-permutation test on group lateralisation ─────────────────────────
[datobs, datrnd] = cluster_test_helper(permute(freq_lat_sub, [2 3 1]), 50000);
[h, p_lat, ~]    = cluster_test(datobs, datrnd, 0, 0.05, 0.05);

dt = mean(diff(plot_time));    % ~10 ms
df = mean(diff(frequencies));  % 0.5 Hz

% ── Group-average figure ──────────────────────────────────────────────────────
B = bwboundaries(h);
fhandle = figure;
imagesc(plot_time, frequencies, squeeze(mean(freq_lat_sub, 1)), [-0.05 0.05]);
axis xy; hold all
for k = 1:length(B)
    bnd = B{k};
    t_c = plot_time(1)   + (bnd(:,2) - 1) * dt;
    f_c = frequencies(1) + (bnd(:,1) - 1) * df;
    plot(t_c, f_c, 'k', 'LineWidth', 1.5)
end
colormap('jet')
cb = colorbar; cb.Label.String = 'Lateralization (dB) contra - ipsi';
xlim([-100 4800]); ylim([6 16])
pbaspect([2, 0.25, 0.25])
set(gca, 'TickDir', 'out')
ax = gca;
ax.YTick = frequencies(1:4:end);
ax.XTick = 0:400:4800;
set(fhandle, 'Position', [100, 100, 1200, 200])
xlabel('Time (ms) relative to onset of memory items')
ylabel('Frequency (Hz)')
title('Alpha lateralisation: contra - ipsi (N=19)')

saveas(fhandle, fullfile(save_dir, 'freq_lateralization_contra_ipsi.png'))

% ── Save group summary ────────────────────────────────────────────────────────
save(fullfile(save_dir, 'group_freq_lateralization.mat'), ...
     'freq_lat_sub', 'h', 'p_lat', 'frequencies', 'plot_time', '-v7.3')

fprintf('Done. Results saved to:\n  %s\n', save_dir)

% ── Per-subject processing (called from parfor; local vars auto-freed on return)
function lat_avg = process_subject(sub, dat_dir, sub_file, ...
                                   frequencies, toi_vec, left_ch, right_ch, ...
                                   plot_time)
    sess1 = load(fullfile(dat_dir, sprintf('MemImp3_mem_whole_sess1_%d.mat', sub)));
    ft1   = sess1.ft_mem;
    sess2 = load(fullfile(dat_dir, sprintf('MemImp3_mem_whole_sess2_%d.mat', sub)));
    ft2   = sess2.ft_mem;
    clear sess1 sess2

    cue_loc = ft1.Results(1, 3);

    bad1  = unique([ft1.bad_trials_mem(:)', ft1.bad_trials_imp1(:)', ...
                    ft1.bad_trials_imp2(:)', ft1.bad_trials_probe1(:)', ...
                    ft1.bad_trials_probe2(:)']);
    bad2  = unique([ft2.bad_trials_mem(:)', ft2.bad_trials_imp1(:)', ...
                    ft2.bad_trials_imp2(:)', ft2.bad_trials_probe1(:)', ...
                    ft2.bad_trials_probe2(:)']);
    incl1 = find(~ismember(1:size(ft1.trial, 1), bad1));
    incl2 = find(~ismember(1:size(ft2.trial, 1), bad2));

    cfg            = [];
    cfg.output     = 'pow';
    cfg.method     = 'mtmconvol';
    cfg.taper      = 'hanning';
    cfg.foi        = frequencies;
    cfg.t_ftimwin  = 5 ./ cfg.foi;
    cfg.toi        = toi_vec;
    cfg.keeptrials = 'yes';

    cfg.trials = incl1;
    freq1 = ft_freqanalysis(cfg, ft1);
    clear ft1

    cfg.trials = incl2;
    freq2 = ft_freqanalysis(cfg, ft2);
    clear ft2

    lch = ismember(upper(freq1.label), upper(left_ch));
    rch = ismember(upper(freq1.label), upper(right_ch));

    d1 = log10(freq1.powspctrm);  clear freq1
    l1 = squeeze(mean(d1(:, lch, :, :), 2)); % average over left channels 【n_trials x n_freq x n_time】
    r1 = squeeze(mean(d1(:, rch, :, :), 2));
    clear d1

    d2 = log10(freq2.powspctrm);  clear freq2
    l2 = squeeze(mean(d2(:, lch, :, :), 2));
    r2 = squeeze(mean(d2(:, rch, :, :), 2));
    clear d2

    if cue_loc == 1
        lat_trials = cat(1, r1 - l1, l2 - r2); %【n_trials x n_freq x n_time】 
    else
        lat_trials = cat(1, l1 - r1, r2 - l2);
    end

    save(sub_file, 'lat_trials', 'cue_loc', 'frequencies', 'plot_time', '-v7.3')
    fprintf('  Saved %s\n', sub_file)

    lat_avg = mean(lat_trials, 1);
end

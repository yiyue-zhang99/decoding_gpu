function  [distance_cos,distances] = mahal_func_theta_kfold_b_sep(data_test,data_train,theta,n_folds)

%% input

% data format is trial by channel by time
% data_test and data_train are different epochs from the same trials

% theta is vector with angles in radians for each trial (must comprise the
% whole circle, thus, for orientation data, which is only 180 degrees, make
% sure to multiply by 2)

% n_folds is the desired number of folds cross validation 
%% output
% output is trial by time, to summarize average over trials

% distance_cos (preferred) is a measure of decoding accuracy, cosine weighted distances
% of pattern-difference between trials of increasinglt dissimilar
% orientations

% distances is the ordered mean-centred distances

%%
theta=circ_dist(theta,0);

u_theta=unique(theta);

train_partitions = cvpartition(theta,'KFold',n_folds); % split data n times using Kfold
distances=nan(length(u_theta),size(data_test,1),size(data_test,3)); % prepare for output 

theta_dist=circ_dist2(u_theta',theta)';

for tst=1:n_folds % run for each fold
    trn_ind = training(train_partitions,tst); % get training trial rows
    tst_ind = test(train_partitions,tst); % get test trial rows
    trn_dat = data_train(trn_ind,:,:); % isolate training data
    tst_dat = data_test(tst_ind,:,:); % isolate test data
    trn_theta =theta(trn_ind);
    tst_theta =theta(tst_ind);
    m=double(nan(length(unique(u_theta)),size(data_test,2),size(data_test,3)));   
    n_conds = [u_theta,histc(trn_theta,u_theta)];
    for c=1:length(unique(u_theta))
        temp1=trn_dat(trn_theta==u_theta(c),:,:);        
        ind=randsample(1:size(temp1,1),min(n_conds(:,2)));
        m(c,:,:)=mean(temp1(ind,:,:),1);
    end
    for t=1:size(data_test,3) % decode at each time-point
        if ~isnan(trn_dat(:,:,t))
            % compute pair-wise mahalabonis distance between test-trials
            % and averaged training data, using the covariance matrix
            % computed from the training data
            temp=pdist2(squeeze(m(:,:,t)), squeeze(tst_dat(:,:,t)),'mahalanobis',covdiag(trn_dat(:,:,t)));            
            distances(:,tst_ind,t)=temp;
        end
    end
end
distance_cos=-mean(bsxfun(@times,cos(theta_dist)',distances),1); % take cosine-weigthed mean of distances
% reorder distances so that same condition distance is in the middle
for c=1:length(u_theta)
    temp=round(circ_dist(u_theta,u_theta(c)),4);
    temp(temp==round(pi,4))=round(-pi,4);
    [~,i]=sort(temp);
    distances(:,theta==u_theta(c),:)=distances(i,theta==u_theta(c),:);
end
distances=-bsxfun(@minus,distances,mean(distances,1)); % mean-centre distances for prettier visuals


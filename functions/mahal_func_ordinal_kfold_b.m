function  [distance_beta,distances] = mahal_func_ordinal_kfold_b(data,conditions,n_folds)
%% 
% capturing the parametric nature of condition difference magnitute
% through linearly regressing the mahalnobis distances as a function of 
% condition difference
%% input
% data format is trial by channel/dimensions by time
% conditions contains condition numbers, the numbers sensibly reflect the 
% relationship between conditions on a continuous scale
% n_folds is the desired number of folds for cross validation 
%% output
% output is trial by time, so average over trials

% distance_beta is a measure of decoding accuracy, reflecting the beta
% value of the linear regression of the mahal. distances as a function of
% condition distance (i.e. positive value indicates a positive relationship
% between mahal. distance and condition difference, providing evidence for
% a parametric neural code of the conditions).

% distances are the trial-wise distances between a trial and the
% condition-averaged data of the training set.
%%
train_partitions = cvpartition(conditions,'KFold',n_folds); % split data n times using Kfold
distances=nan(length(unique(conditions)),size(data,1),size(data,3)); % prepare for output
distance_beta=nan(size(data,1),size(data,3));
u_cond=unique(conditions); 
for tst=1:n_folds % run for each fold
    trn_ind = training(train_partitions,tst); % get training trial rows
    tst_ind = test(train_partitions,tst); % get test trial rows
    trn_dat = data(trn_ind,:,:); % isolate training data
    tst_dat = data(tst_ind,:,:); % isolate test data
    trn_cond =conditions(trn_ind);
    n_conds = [u_cond,histc(trn_cond,u_cond)]; % get number of trials for each condition
    m=(nan(length(u_cond),size(data,2),size(data,3)));
    for c=1:length(u_cond)
        temp1=trn_dat(trn_cond==u_cond(c),:,:);
        ind=randsample(1:size(temp1,1),min(n_conds(:,2))); % r
        m(c,:,:)=mean(temp1(ind,:,:),1);
    end
    for ti=1:size(data,3) % decode at each time-point
        if ~isnan(trn_dat(:,:,ti))
            % compute pair-wise mahalabonis distance between test-trials
            % and averaged training data, using the covariance matrix
            % computed from the training data
            temp=pdist2(squeeze(m(:,:,ti)), squeeze(tst_dat(:,:,ti)),'mahalanobis',covdiag(trn_dat(:,:,ti)));
            distances(:,tst_ind,ti)=temp;
        end
    end
end
% linear regression for each trial (not very efficient)
u_cond_diff=abs(bsxfun(@minus,u_cond,u_cond'));
for cond=u_cond'
    X=u_cond_diff(cond,:)';
    X(:,2)=1;
    Y=squeeze(distances(:,conditions==cond,:));
    for ti=1:size(data,3)
        beta=pinv(X)*Y(:,:,ti);
        distance_beta(conditions==cond,ti)=beta(1,:);
    end
end

%%
    function sigma=covdiag(x)
        
        % x (t*n): t iid observations on n random variables
        % sigma (n*n): invertible covariance matrix estimator
        %
        % Shrinks towards diagonal matrix
        % as described in Ledoit and Wolf, 2004
        
        % de-mean returns
        [t,n]=size(x);
        meanx=mean(x);
        x=x-meanx(ones(t,1),:);
        
        % compute sample covariance matrix
        sample=(1/t).*(x'*x);
        
        % compute prior
        prior=diag(diag(sample));
        
        % compute shrinkage parameters
        d=1/n*norm(sample-prior,'fro')^2;
        y=x.^2;
        r2=1/n/t^2*sum(sum(y'*y))-1/n/t*sum(sum(sample.^2));
        
        % compute the estimator
        shrinkage=max(0,min(1,r2/d));
        sigma=shrinkage*prior+(1-shrinkage)*sample;
    end
end

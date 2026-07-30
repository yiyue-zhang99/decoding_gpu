function  [distance_beta,distances] = mahal_func_ordinal_kfold_b_sep(data_test,data_train,conditions,n_folds)
%% 
% capturing the parametric nature of condition difference magnitute
% through linearly regressing the mahalnobis distances as a function of 
% condition difference
% training and testing on different epochs of the same trials
%% input
% data format is trial by channel/dimensions by time
% data_test and data_train are different epochs from the same trials
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
distances=nan(length(unique(conditions)),size(data_test,1),size(data_test,3)); % prepare for output
distance_beta=nan(size(data_test,1),size(data_test,3));

% absolut difference between trial-wise conditions and all other possible
% conditions, used for linear regression later on
cond_diff=abs(bsxfun(@minus,conditions,unique(conditions)'));
u_cond=unique(conditions);
for tst=1:n_folds % run for each fold
    trn_ind = training(train_partitions,tst); % get training trial rows
    tst_ind = test(train_partitions,tst); % get test trial rows
    trn_dat = data_train(trn_ind,:,:); % isolate training data
    tst_dat = data_test(tst_ind,:,:); % isolate test data
    trn_cond =conditions(trn_ind);
    n_conds = [u_cond,histc(trn_cond,u_cond)];
    m=(nan(length(u_cond),size(data_test,2),size(data_test,3)));
    for c=1:length(u_cond)
        temp1=trn_dat(trn_cond==u_cond(c),:,:);
        ind=randsample(1:size(temp1,1),min(n_conds(:,2)));
        m(c,:,:)=mean(temp1(ind,:,:),1);
    end
    for ti=1:size(data_test,3) % decode at each time-point
        if ~isnan(trn_dat(:,:,ti))
            % compute pair-wise mahalabonis distance between test-trials
            % and averaged training data, using the covariance matrix
            % computed from the training data
            temp=pdist2(squeeze(m(:,:,ti)), squeeze(tst_dat(:,:,ti)),'mahalanobis',covdiag(trn_dat(:,:,ti)));
            distances(:,tst_ind,ti)=temp;
        end
    end
    tst_rows=find(tst_ind);
    % linear regression for each trial (not efficient)
    for trl=1:sum(tst_ind)
        X=cond_diff(tst_rows(trl),:)';
        Y=squeeze(distances(:,tst_rows(trl),:));
        X(:,2)=1;
        beta=pinv(X)*Y;
        distance_beta(tst_rows(trl),:)=beta(1,:);
    end
end

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

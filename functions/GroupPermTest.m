function p = GroupPermTest(dat, nSims,tail)
%% test dat against zero
%% dat can be any number of dimensions. Group must be the first dimension

mdat = mean(dat,1);

p = zeros(size(mdat));
for sim=1:nSims
    permind = sign(rand(size(dat,1),1)-.5);
    permind = repmat(permind,[size(mdat)]);
    p = p + (mean(dat.*permind)>=mdat);
end
p = p./nSims;
if p==0
    p=1/nSims;
end
if tail==-1
    p=1-p;
elseif tail==2
    p(p>=.5) = 1-p(p>=.5);
    p = p*2;
end
end
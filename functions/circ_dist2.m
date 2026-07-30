function distance = circ_dist2(alpha, beta)
% Pairwise signed circular differences between two angle vectors.

alpha = alpha(:);
beta = beta(:);
distance = angle(exp(1i .* alpha) ./ exp(1i .* beta.'));
end

function distance = circ_dist(alpha, beta)
% Signed circular difference between angles in radians.

distance = angle(exp(1i .* alpha) ./ exp(1i .* beta));
end

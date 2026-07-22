function [h] = sr_create(kind, seed_hi, seed_lo)
mex_id_ = 'c o simdrng_state* = sr_create_w(c i int, c i double, c i double)';
[h] = simdrng_mex(mex_id_, kind, seed_hi, seed_lo);

% -----------------------------------------------------------------------
% Uniform doubles in [0,1). Returns an n-vector (reshaped by the classdef).

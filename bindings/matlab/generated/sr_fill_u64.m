function [y] = sr_fill_u64(self, n)
mex_id_ = 'c o mxArray = sr_fill_u64_w(c i simdrng_state*, c i int64_t)';
[y] = simdrng_mex(mex_id_, self, n);

% -----------------------------------------------------------------------
% Introspection + free.

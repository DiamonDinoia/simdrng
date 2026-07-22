function [y] = sr_fill_double(self, n)
mex_id_ = 'sr_fill_double_w(c i simdrng_state*, c o double[x], c i int64_t)';
[y] = simdrng_mex(mex_id_, self, n, n);

% -----------------------------------------------------------------------
% Raw uint64 draws (mxArray* pass-through -> genuine uint64 column vector).

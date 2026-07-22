function [k] = sr_get_kind(self)
mex_id_ = 'c o int = sr_get_kind_w(c i simdrng_state*)';
[k] = simdrng_mex(mex_id_, self);


classdef simdrng < handle
% SIMDRNG  MATLAB/Octave handle class over the simdrng SIMD generators.
%
% Thin classdef over the mwrap-generated sr_*.m stubs (see simdrng.mw). The
% opaque simdrng_t handle lives in the `mwptr` property, read/written via mwrap's
% R2008OO object-pointer convention. The API mirrors numpy's Generator: a seeded
% object with `random` (uniform doubles) and `integers`/`raw` (raw uint64).
%
% CONSTRUCTION
%   rng = simdrng()                 xoshiro256++, seed 0
%   rng = simdrng(kind)             named kind, seed 0
%   rng = simdrng(kind, seed)       seed is any nonneg integer (full 64-bit)
%
%   kind — one of: 'splitmix' 'xoshiro' 'chacha8' 'chacha12' 'chacha20'
%          'philox4x32' 'philox2x32' 'philox4x64' 'philox2x64'
%          (or the numeric enum value).
%
% USAGE
%   u = rng.random()                scalar uniform double in [0,1)
%   u = rng.random(n)               n-vector
%   u = rng.random(m, n)            m-by-n
%   u = rng.random([m n ...])       arbitrary shape
%   x = rng.raw(...)                same shapes, raw uint64
%   k = rng.kind()                  enum value the handle was created with
%   delete(rng)                     frees C-side memory

    properties (SetAccess = private)
        mwptr        % opaque simdrng_t handle (mwrap object pointer)
    end

    methods

        function obj = simdrng(kind, seed)
            if nargin < 1, kind = 'xoshiro'; end
            if nargin < 2, seed = 0; end
            code = simdrng.kind_code(kind);
            % Split the 64-bit seed into two 32-bit halves; each is exact as a
            % double, so mwrap's double-only marshalling stays lossless.
            s  = uint64(seed);
            hi = double(bitshift(s, -32));
            lo = double(bitand(s, uint64(4294967295)));
            obj.mwptr = sr_create(double(code), hi, lo);
        end

        function u = random(obj, varargin)
        % RANDOM  Uniform doubles in [0,1), shaped like rand/zeros.
            sz = simdrng.parse_size(varargin);
            y  = sr_fill_double(obj, double(prod(sz)));
            u  = reshape(y, sz);
        end

        function x = raw(obj, varargin)
        % RAW  Raw uint64 draws, shaped like rand/zeros.
            sz = simdrng.parse_size(varargin);
            y  = sr_fill_u64(obj, double(prod(sz)));
            x  = reshape(y, sz);
        end

        function k = kind(obj)
            k = sr_get_kind(obj);
        end

        function delete(obj)
            if ~isempty(obj.mwptr)
                sr_free(obj);
                obj.mwptr = [];
            end
        end

    end

    methods (Static, Access = private)

        function code = kind_code(kind)
        % Map a name (or numeric enum) to the simdrng_kind value in capi.h.
            if isnumeric(kind)
                code = double(kind);
                return;
            end
            names = {'splitmix', 'xoshiro', 'chacha8', 'chacha12', 'chacha20', ...
                     'philox4x32', 'philox2x32', 'philox4x64', 'philox2x64'};
            idx = find(strcmpi(kind, names), 1);
            if isempty(idx)
                error('simdrng:kind', 'unknown generator kind ''%s''.', kind);
            end
            code = idx - 1;   % 0-based enum
        end

        function sz = parse_size(args)
        % rand-style size parsing: () -> [1 1]; (n) -> [n 1] (numpy-like vector);
        % (m,n,...) -> [m n ...]; ([m n ...]) -> that row.
            if isempty(args)
                sz = [1 1];
            elseif numel(args) == 1
                v = double(args{1});
                if isscalar(v)
                    sz = [v 1];
                else
                    sz = v(:)';
                end
            else
                sz = cellfun(@double, args);
            end
        end

    end
end

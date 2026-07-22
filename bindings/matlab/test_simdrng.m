% test_simdrng.m — MATLAB/Octave smoke test for the simdrng binding.
% Calls error() on failure so `octave --eval test_simdrng` / `matlab -batch`
% exits non-zero and the CTest target fails.

function test_simdrng
    kinds = {'splitmix', 'xoshiro', 'chacha8', 'chacha12', 'chacha20', ...
             'philox4x32', 'philox2x32', 'philox4x64', 'philox2x64'};

    for i = 1:numel(kinds)
        k = kinds{i};

        % Uniform doubles land in [0,1) and honour the requested shape.
        rng = simdrng(k, 42);
        u = rng.random(3, 4);
        assert(isequal(size(u), [3 4]), 'random shape wrong for %s', k);
        assert(all(u(:) >= 0 & u(:) < 1), 'random out of [0,1) for %s', k);

        % Determinism: same kind + seed -> identical streams.
        a = simdrng(k, 7);
        b = simdrng(k, 7);
        assert(isequal(a.raw(100), b.raw(100)), 'not deterministic for %s', k);

        % Bulk uint64 fill equals sequential draws is covered by the C++ test
        % suite; here we just check raw() returns uint64 of the right length.
        x = a.raw(16);
        assert(strcmp(class(x), 'uint64'), 'raw not uint64 for %s', k);
        assert(numel(x) == 16, 'raw length wrong for %s', k);

        assert(rng.kind() == i - 1, 'kind mismatch for %s', k);
    end

    % Full 64-bit seed is preserved (would collide if truncated to a double).
    s1 = simdrng('xoshiro', uint64(2)^63 + 1);
    s2 = simdrng('xoshiro', uint64(2)^63 + 2);
    assert(~isequal(s1.raw(4), s2.raw(4)), '64-bit seed truncated');

    disp('test_simdrng: OK');
end

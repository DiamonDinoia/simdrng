using Test
using SimdRNG

const SEED = 0x9E3779B97F4A7C15
const N = 1024

@testset "SimdRNG" begin
    @test !isempty(version())

    @testset "$k" for k in SimdRNG.KINDS
        g = Generator(k; seed = SEED)
        @test kind(g) == k

        # determinism: same seed -> identical raw stream
        a = raw(g, N)
        b = raw(Generator(k; seed = SEED), N)
        @test a == b

        # divergence: different seed -> different stream
        @test raw(Generator(k; seed = SEED + 1), N) != a

        # range contract for the uniform reals
        d = random(Generator(k; seed = SEED), N)
        @test all(0.0 .<= d .< 1.0)
        @test size(random(Generator(k; seed = SEED), 4, 4)) == (4, 4)

        # self-consistency: bulk random == repeated scalar draws on one stream
        bulk = random(Generator(k; seed = SEED), N)
        gs = Generator(k; seed = SEED)
        @test bulk == [random(gs) for _ in 1:N]

        @test raw(g, 8) isa Vector{UInt64}
    end
end

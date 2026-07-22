"""
    SimdRNG

Julia wrapper for the simdrng C ABI (`libsimdrng_c`, see include/simdrng/capi.h).
Thin `ccall` layer over the opaque `simdrng_t` handle: construct a `Generator`,
then draw uniform reals with [`random`](@ref) or raw 64-bit words with
[`raw`](@ref). The bulk fills run the generator's own SIMD path.

```julia
using SimdRNG
g = Generator(:xoshiro; seed = 0x9E3779B97F4A7C15)
random(g, 1000)      # Vector{Float64} in [0,1)
random(g, 4, 4)      # 4×4 Matrix{Float64}
raw(g, 8)            # Vector{UInt64}
kind(g)              # :xoshiro
```

## Library resolution

`libsimdrng_c` is located, in order: the `LIBSIMDRNG_C` environment variable
(explicit path, always wins); the loader search path (`Libdl.find_library`); a
sibling CMake `build*/` tree found by walking up from the package dir. The CMake
`julia_simdrng` test sets `LIBSIMDRNG_C` to the freshly-built library.
"""
module SimdRNG

using Libdl

# ---- library resolution -------------------------------------------------

function _find_in_build_tree(start::AbstractString)
    libname = "libsimdrng_c." * Libdl.dlext
    dir = abspath(start)
    while true
        for bd in sort(filter(isdir, readdir(dir; join = true)))
            if startswith(basename(bd), "build")
                # the lib lands at the CMake binary-dir root
                for cand in (joinpath(bd, libname), _walk_for(bd, libname))
                    cand !== nothing && isfile(cand) && return cand
                end
            end
        end
        parent = dirname(dir)
        parent == dir && return nothing
        dir = parent
    end
end

# shallow search for libname under root (build/<preset>/libsimdrng_c.so)
function _walk_for(root::AbstractString, libname::AbstractString)
    for (d, _, files) in walkdir(root)
        libname in files && return joinpath(d, libname)
    end
    return nothing
end

function _resolve()
    env = get(ENV, "LIBSIMDRNG_C", "")
    isempty(env) || return env
    found = Libdl.find_library(["libsimdrng_c", "simdrng_c"])
    isempty(found) || return found
    walked = _find_in_build_tree(@__DIR__)
    walked === nothing || return walked
    return "libsimdrng_c." * Libdl.dlext
end

const LIBSIMDRNG = _resolve()

function __init__()
    if !isfile(LIBSIMDRNG) && Libdl.find_library([LIBSIMDRNG]) == ""
        @warn "SimdRNG: could not locate libsimdrng_c ($LIBSIMDRNG). Set the " *
              "LIBSIMDRNG_C env var or build the project (a build*/ dir near the repo root)."
    end
end

# ---- generator kinds (mirror simdrng_kind) ------------------------------

const KINDS = (
    :splitmix, :xoshiro, :chacha8, :chacha12, :chacha20,
    :philox4x32, :philox2x32, :philox4x64, :philox2x64,
)
# name -> 0-based C enum value
const _KIND_CODE = Dict(k => Cint(i - 1) for (i, k) in enumerate(KINDS))

_last_error() = (p = ccall((:simdrng_last_error, LIBSIMDRNG), Cstring, ());
                 p == C_NULL ? "" : unsafe_string(p))

"""Linked-library version string, e.g. "0.0.2-dev.150+g..."."""
version() = unsafe_string(ccall((:simdrng_version, LIBSIMDRNG), Cstring, ()))

# ---- handle -------------------------------------------------------------

"""
    Generator(kind::Symbol = :xoshiro; seed::Integer = 0)

An opaque simdrng generator of the given `kind` (one of `SimdRNG.KINDS`) seeded
with the 64-bit `seed`. Memory is released by a finalizer.
"""
mutable struct Generator
    ptr  :: Ptr{Cvoid}
    kind :: Symbol
end

function Generator(kind::Symbol = :xoshiro; seed::Integer = 0)
    code = get(_KIND_CODE, kind) do
        error("unknown generator kind :$kind; expected one of $(KINDS)")
    end
    ptr = ccall((:simdrng_create, LIBSIMDRNG), Ptr{Cvoid},
                (Cint, UInt64), code, UInt64(seed) % UInt64)
    ptr == C_NULL && error("simdrng_create failed: " * _last_error())
    g = Generator(ptr, kind)
    finalizer(g) do h
        h.ptr == C_NULL ||
            (h.ptr = ccall((:simdrng_free, LIBSIMDRNG), Ptr{Cvoid}, (Ptr{Cvoid},), h.ptr))
    end
    return g
end

"""The kind the generator was created with (round-trips through the C handle)."""
function kind(g::Generator)
    code = ccall((:simdrng_get_kind, LIBSIMDRNG), Cint, (Ptr{Cvoid},), g.ptr)
    return KINDS[code + 1]
end

# ---- draws --------------------------------------------------------------

"""
    random(g::Generator, dims::Integer...) -> Array{Float64}

Uniform `Float64` draws in `[0,1)`. With no `dims`, returns a single scalar;
otherwise an array of the requested shape.
"""
function random(g::Generator, dims::Integer...)
    if isempty(dims)
        return ccall((:simdrng_next_double, LIBSIMDRNG), Cdouble, (Ptr{Cvoid},), g.ptr)
    end
    out = Array{Float64}(undef, dims...)
    GC.@preserve out ccall((:simdrng_fill_double, LIBSIMDRNG), Cvoid,
                           (Ptr{Cvoid}, Ptr{Cdouble}, Csize_t), g.ptr, out, Csize_t(length(out)))
    return out
end

"""
    raw(g::Generator, n::Integer) -> Vector{UInt64}

`n` raw 64-bit words straight from the generator (no `[0,1)` transform).
"""
function raw(g::Generator, n::Integer)
    out = Vector{UInt64}(undef, n)
    GC.@preserve out ccall((:simdrng_fill_u64, LIBSIMDRNG), Cvoid,
                           (Ptr{Cvoid}, Ptr{UInt64}, Csize_t), g.ptr, out, Csize_t(n))
    return out
end

Base.show(io::IO, g::Generator) = print(io, "Generator(:$(g.kind))")

export Generator, random, raw, kind, version

end # module SimdRNG

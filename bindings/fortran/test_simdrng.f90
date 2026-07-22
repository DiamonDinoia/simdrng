! test_simdrng.f90 — self-checking conformance test for the Fortran binding.
!
! Mirrors tests/test_capi.c: for every generator kind it checks handle identity,
! determinism (same seed -> identical stream), stream divergence (different seed
! -> different stream), the [0,1) range contract for the real fills, and
! self-consistency (fill_double and repeated next_double agree bit-for-bit on one
! stream). Failures are counted; a nonzero count ends the program with
! `error stop <count>` so the process exit code is nonzero.
program test_simdrng
    use, intrinsic :: iso_c_binding
    use simdrng
    implicit none

    integer, parameter :: N = 1024
    integer(c_int64_t) :: a64(N), b64(N)
    integer(c_int32_t) :: a32(N)
    real(c_double)     :: ad(N), bd(N)
    real(c_float)      :: af(N)
    type(c_ptr)        :: g, g2
    integer            :: kind, i, fails
    integer(c_int64_t) :: seed

    fails = 0
    seed = int(z'9E3779B97F4A7C15', c_int64_t)

    print *, "simdrng version: ", simdrng_version_string()

    do kind = SIMDRNG_SPLITMIX, SIMDRNG_KIND_COUNT - 1
        ! -- construction + identity --
        g = simdrng_create(kind, seed)
        if (.not. c_associated(g)) then
            call fail(fails, kind, "create returned NULL: " // simdrng_error_message())
            cycle
        end if
        if (simdrng_get_kind(g) /= kind) then
            call fail(fails, kind, "get_kind mismatch")
        end if

        ! -- determinism: a second handle on the same seed reproduces the stream --
        g2 = simdrng_create(kind, seed)
        call simdrng_fill_u64(g, a64, int(N, c_size_t))
        call simdrng_fill_u64(g2, b64, int(N, c_size_t))
        if (any(a64 /= b64)) call fail(fails, kind, "u64 stream not deterministic")
        g2 = simdrng_free(g2)

        ! -- divergence: a different seed gives a different stream --
        g2 = simdrng_create(kind, seed + 1_c_int64_t)
        call simdrng_fill_u64(g2, b64, int(N, c_size_t))
        if (all(a64 == b64)) call fail(fails, kind, "distinct seeds gave identical stream")
        g2 = simdrng_free(g2)

        ! -- real fills honour [0,1) --
        call simdrng_fill_double(g, ad, int(N, c_size_t))
        if (any(ad < 0.0_c_double) .or. any(ad >= 1.0_c_double)) &
            call fail(fails, kind, "fill_double out of [0,1)")
        call simdrng_fill_float(g, af, int(N, c_size_t))
        if (any(af < 0.0_c_float) .or. any(af >= 1.0_c_float)) &
            call fail(fails, kind, "fill_float out of [0,1)")

        ! -- u32 fill runs without corrupting neighbouring memory (smoke) --
        call simdrng_fill_u32(g, a32, int(N, c_size_t))
        g = simdrng_free(g)

        ! -- self-consistency: fill_double == repeated next_double on one stream --
        g = simdrng_create(kind, seed)
        call simdrng_fill_double(g, ad, int(N, c_size_t))
        g = simdrng_free(g)
        g = simdrng_create(kind, seed)
        do i = 1, N
            bd(i) = simdrng_next_double(g)
        end do
        g = simdrng_free(g)
        if (any(ad /= bd)) call fail(fails, kind, "fill_double vs next_double disagree")
    end do

    if (fails /= 0) then
        print *, "test_simdrng: FAILED (", fails, " checks)"
        error stop fails
    end if
    print *, "test_simdrng: OK"

contains

    subroutine fail(count, kind, msg)
        integer, intent(inout)       :: count
        integer, intent(in)          :: kind
        character(len=*), intent(in) :: msg
        count = count + 1
        print *, "  [kind ", kind, "] ", msg
    end subroutine fail

end program test_simdrng

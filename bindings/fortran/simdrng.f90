! simdrng.f90 — Fortran 2018 binding for the simdrng C ABI (see ../../include/simdrng/capi.h).
!
! A thin, faithful `iso_c_binding` shim: every C entry point is exposed as a
! Fortran procedure of the same name via an `interface` block, so the mapping is
! one-to-one and explicit. The opaque `simdrng_t` handle is a `type(c_ptr)`.
!
! uint64 arguments/results are carried as `integer(c_int64_t)` — Fortran has no
! unsigned integer, but the bit pattern round-trips exactly, which is all the
! generator and the raw-draw API care about. Seeds and raw u64 draws are simply
! reinterpreted; treat them as bit patterns, not signed magnitudes.
!
! Precision lives in the fill routine name (FINUFFT/FFTW style): fill_double /
! fill_float for reals, fill_u64 / fill_u32 for raw bits.
module simdrng
    use, intrinsic :: iso_c_binding
    implicit none
    public

    ! ---- generator family (simdrng_kind) -----------------------------------
    integer(c_int), parameter :: SIMDRNG_SPLITMIX   = 0_c_int
    integer(c_int), parameter :: SIMDRNG_XOSHIRO    = 1_c_int
    integer(c_int), parameter :: SIMDRNG_CHACHA8    = 2_c_int
    integer(c_int), parameter :: SIMDRNG_CHACHA12   = 3_c_int
    integer(c_int), parameter :: SIMDRNG_CHACHA20   = 4_c_int
    integer(c_int), parameter :: SIMDRNG_PHILOX4X32 = 5_c_int
    integer(c_int), parameter :: SIMDRNG_PHILOX2X32 = 6_c_int
    integer(c_int), parameter :: SIMDRNG_PHILOX4X64 = 7_c_int
    integer(c_int), parameter :: SIMDRNG_PHILOX2X64 = 8_c_int
    integer(c_int), parameter :: SIMDRNG_KIND_COUNT = 9_c_int

    interface
        ! ---- lifetime ---------------------------------------------------
        function simdrng_create(kind, seed) bind(C, name="simdrng_create") result(handle)
            import :: c_int, c_int64_t, c_ptr
            integer(c_int),     value :: kind
            integer(c_int64_t), value :: seed
            type(c_ptr)               :: handle
        end function simdrng_create

        function simdrng_free(g) bind(C, name="simdrng_free") result(null_handle)
            import :: c_ptr
            type(c_ptr), value :: g
            type(c_ptr)        :: null_handle
        end function simdrng_free

        function simdrng_get_kind(g) bind(C, name="simdrng_get_kind") result(kind)
            import :: c_ptr, c_int
            type(c_ptr), value :: g
            integer(c_int)     :: kind
        end function simdrng_get_kind

        ! ---- single draws -----------------------------------------------
        function simdrng_next_u64(g) bind(C, name="simdrng_next_u64") result(x)
            import :: c_ptr, c_int64_t
            type(c_ptr), value :: g
            integer(c_int64_t) :: x
        end function simdrng_next_u64

        function simdrng_next_double(g) bind(C, name="simdrng_next_double") result(x)
            import :: c_ptr, c_double
            type(c_ptr), value :: g
            real(c_double)     :: x
        end function simdrng_next_double

        ! ---- bulk fills (out must hold at least n elements) -------------
        subroutine simdrng_fill_u64(g, out, n) bind(C, name="simdrng_fill_u64")
            import :: c_ptr, c_int64_t, c_size_t
            type(c_ptr),        value       :: g
            integer(c_int64_t), intent(out) :: out(*)
            integer(c_size_t),  value       :: n
        end subroutine simdrng_fill_u64

        subroutine simdrng_fill_u32(g, out, n) bind(C, name="simdrng_fill_u32")
            import :: c_ptr, c_int32_t, c_size_t
            type(c_ptr),        value       :: g
            integer(c_int32_t), intent(out) :: out(*)
            integer(c_size_t),  value       :: n
        end subroutine simdrng_fill_u32

        subroutine simdrng_fill_double(g, out, n) bind(C, name="simdrng_fill_double")
            import :: c_ptr, c_double, c_size_t
            type(c_ptr),       value       :: g
            real(c_double),    intent(out) :: out(*)
            integer(c_size_t), value       :: n
        end subroutine simdrng_fill_double

        subroutine simdrng_fill_float(g, out, n) bind(C, name="simdrng_fill_float")
            import :: c_ptr, c_float, c_size_t
            type(c_ptr),       value       :: g
            real(c_float),     intent(out) :: out(*)
            integer(c_size_t), value       :: n
        end subroutine simdrng_fill_float

        ! ---- introspection ----------------------------------------------
        function simdrng_version() bind(C, name="simdrng_version") result(msg)
            import :: c_ptr
            type(c_ptr) :: msg
        end function simdrng_version

        function simdrng_last_error() bind(C, name="simdrng_last_error") result(msg)
            import :: c_ptr
            type(c_ptr) :: msg
        end function simdrng_last_error

        ! libc strlen, used to size the returned C strings.
        function c_strlen(s) bind(C, name="strlen") result(n)
            import :: c_ptr, c_size_t
            type(c_ptr), value :: s
            integer(c_size_t)  :: n
        end function c_strlen
    end interface

contains

    ! Copy a NUL-terminated C string into a Fortran string (empty if NULL/"").
    function simdrng_c_string(p) result(str)
        type(c_ptr), intent(in)         :: p
        character(len=:), allocatable   :: str
        character(kind=c_char), pointer :: buf(:)
        integer(c_size_t)               :: n
        integer                         :: i

        if (.not. c_associated(p)) then
            str = ""
            return
        end if
        n = c_strlen(p)
        if (n == 0_c_size_t) then
            str = ""
            return
        end if
        call c_f_pointer(p, buf, [n])
        allocate (character(len=int(n)) :: str)
        do i = 1, int(n)
            str(i:i) = buf(i)
        end do
    end function simdrng_c_string

    function simdrng_version_string() result(str)
        character(len=:), allocatable :: str
        str = simdrng_c_string(simdrng_version())
    end function simdrng_version_string

    function simdrng_error_message() result(str)
        character(len=:), allocatable :: str
        str = simdrng_c_string(simdrng_last_error())
    end function simdrng_error_message

end module simdrng

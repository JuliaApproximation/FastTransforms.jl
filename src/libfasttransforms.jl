ft_build_from_source = get(ENV, "FT_BUILD_FROM_SOURCE", "false")
if ft_build_from_source == "true"
    using Libdl
    const libfasttransforms = find_library("libfasttransforms", [joinpath(dirname(@__DIR__), "deps")])
    if libfasttransforms ≡ nothing || length(libfasttransforms) == 0
        error("FastTransforms is not properly installed. Please run Pkg.build(\"FastTransforms\") ",
              "and restart Julia.")
    end
else
    using FastTransforms_jll
end

ft_set_num_threads(n::Integer) = ccall((:ft_set_num_threads, libfasttransforms), Cvoid, (Cint, ), n)
ft_fftw_plan_with_nthreads(n::Integer) = ccall((:ft_fftw_plan_with_nthreads, libfasttransforms), Cvoid, (Cint, ), n)

function __init__()
    n = ceil(Int, Sys.CPU_THREADS/2)
    ft_set_num_threads(n)
    ccall((:ft_fftw_init_threads, libfasttransforms), Cint, ())
    ft_fftw_plan_with_nthreads(n)
end


"""
    mpfr_t <: AbstractFloat

A Julia struct that exactly matches `mpfr_t`.
"""
struct mpfr_t <: AbstractFloat
    prec::Clong
    sign::Cint
    exp::Clong
    d::Ptr{Limb}
end

"""
`BigFloat` is a mutable struct and there is no guarantee that each entry in
an `Array{BigFloat}` has unique pointers. For example, looking at the `Limb`s,

    Id = Matrix{BigFloat}(I, 3, 3)
    map(x->x.d, Id)

shows that the ones and the zeros all share the same pointers. If a C function
assumes unicity of each datum, then the array must be renewed with a `deepcopy`.
"""
function renew!(x::Array{BigFloat})
    for i in eachindex(x)
        @inbounds x[i] = deepcopy(x[i])
    end
    return x
end

function horner!(c::Vector{Float64}, x::Vector{Float64}, f::Vector{Float64})
    @assert length(x) == length(f)
    ccall((:ft_horner, libfasttransforms), Cvoid, (Cint, Ptr{Float64}, Cint, Cint, Ptr{Float64}, Ptr{Float64}), length(c), c, 1, length(x), x, f)
    f
end

function horner!(c::Vector{Float32}, x::Vector{Float32}, f::Vector{Float32})
    @assert length(x) == length(f)
    ccall((:ft_hornerf, libfasttransforms), Cvoid, (Cint, Ptr{Float32}, Cint, Cint, Ptr{Float32}, Ptr{Float32}), length(c), c, 1, length(x), x, f)
    f
end

function check_clenshaw_recurrences(N, A, B, C)
    if length(A) < N || length(B) < N || length(C) < N+1
        throw(ArgumentError("A, B must contain at least $N entries and C must contain at least $(N+1) entrie"))
    end
end

function check_clenshaw_points(x, ϕ₀, f)
    length(x) == length(ϕ₀) == length(f) || throw(ArgumentError("Dimensions must match"))
end

function check_clenshaw_points(x, f)
    length(x) == length(f) || throw(ArgumentError("Dimensions must match"))
end

function _clenshaw!(::AbstractStridedLayout, ::AbstractColumnMajor, ::AbstractColumnMajor, c::AbstractVector{Float64}, x::AbstractVector{Float64}, f::AbstractVector{Float64})
    @boundscheck check_clenshaw_points(x, f)
    ccall((:ft_clenshaw, libfasttransforms), Cvoid, (Cint, Ptr{Float64}, Cint, Cint, Ptr{Float64}, Ptr{Float64}), length(c), c, stride(c,1), length(x), x, f)
    f
end

function _clenshaw!(::AbstractStridedLayout, ::AbstractColumnMajor, ::AbstractColumnMajor, c::AbstractVector{Float32}, x::AbstractVector{Float32}, f::AbstractVector{Float32})
    @boundscheck check_clenshaw_points(x, f)
    ccall((:ft_clenshawf, libfasttransforms), Cvoid, (Cint, Ptr{Float32}, Cint, Cint, Ptr{Float32}, Ptr{Float32}), length(c), c, stride(c,1), length(x), x, f)
    f
end

function clenshaw!(c::Vector{Float64}, A::Vector{Float64}, B::Vector{Float64}, C::Vector{Float64}, x::Vector{Float64}, ϕ₀::Vector{Float64}, f::Vector{Float64})
    N = length(c)
    @boundscheck check_clenshaw_recurrences(N, A, B, C)
    @boundscheck check_clenshaw_points(x, ϕ₀, f)
    ccall((:ft_orthogonal_polynomial_clenshaw, libfasttransforms), Cvoid, (Cint, Ptr{Float64}, Cint, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Cint, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}), N, c, 1, A, B, C, length(x), x, ϕ₀, f)
    f
end

function clenshaw!(c::Vector{Float32}, A::Vector{Float32}, B::Vector{Float32}, C::Vector{Float32}, x::Vector{Float32}, ϕ₀::Vector{Float32}, f::Vector{Float32})
    N = length(c)
    @boundscheck check_clenshaw_recurrences(N, A, B, C)
    @boundscheck check_clenshaw_points(x, ϕ₀, f)
    ccall((:ft_orthogonal_polynomial_clenshawf, libfasttransforms), Cvoid, (Cint, Ptr{Float32}, Cint, Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Cint, Ptr{Float32}, Ptr{Float32}, Ptr{Float32}), N, c, 1, A, B, C, length(x), x, ϕ₀, f)
    f
end

const LEG2CHEB              = 0
const CHEB2LEG              = 1
const ULTRA2ULTRA           = 2
const JAC2JAC               = 3
const LAG2LAG               = 4
const JAC2ULTRA             = 5
const ULTRA2JAC             = 6
const JAC2CHEB              = 7
const CHEB2JAC              = 8
const ULTRA2CHEB            = 9
const CHEB2ULTRA           = 10
const ASSOCIATEDJAC2JAC    = 11
const SPHERE               = 12
const SPHEREV              = 13
const DISK                 = 14
const RECTDISK             = 15
const TRIANGLE             = 16
const TETRAHEDRON          = 17
const SPINSPHERE           = 18
const SPHERESYNTHESIS      = 19
const SPHEREANALYSIS       = 20
const SPHEREVSYNTHESIS     = 21
const SPHEREVANALYSIS      = 22
const DISKSYNTHESIS        = 23
const DISKANALYSIS         = 24
const RECTDISKSYNTHESIS    = 25
const RECTDISKANALYSIS     = 26
const TRIANGLESYNTHESIS    = 27
const TRIANGLEANALYSIS     = 28
const TETRAHEDRONSYNTHESIS = 29
const TETRAHEDRONANALYSIS  = 30
const SPINSPHERESYNTHESIS  = 31
const SPINSPHEREANALYSIS   = 32
const SPHERICALISOMETRY    = 33


let k2s = Dict(LEG2CHEB             => "Legendre--Chebyshev",
               CHEB2LEG             => "Chebyshev--Legendre",
               ULTRA2ULTRA          => "ultraspherical--ultraspherical",
               JAC2JAC              => "Jacobi--Jacobi",
               LAG2LAG              => "Laguerre--Laguerre",
               JAC2ULTRA            => "Jacobi--ultraspherical",
               ULTRA2JAC            => "ultraspherical--Jacobi",
               JAC2CHEB             => "Jacobi--Chebyshev",
               CHEB2JAC             => "Chebyshev--Jacobi",
               ULTRA2CHEB           => "ultraspherical--Chebyshev",
               CHEB2ULTRA           => "Chebyshev--ultraspherical",
               ASSOCIATEDJAC2JAC    => "Associated Jacobi--Jacobi",
               SPHERE               => "Spherical harmonic--Fourier",
               SPHEREV              => "Spherical vector field--Fourier",
               DISK                 => "Zernike--Chebyshev×Fourier",
               RECTDISK             => "Dunkl-Xu--Chebyshev²",
               TRIANGLE             => "Proriol--Chebyshev²",
               TETRAHEDRON          => "Proriol--Chebyshev³",
               SPINSPHERE           => "Spin-weighted spherical harmonic--Fourier",
               SPHERESYNTHESIS      => "FFTW Fourier synthesis on the sphere",
               SPHEREANALYSIS       => "FFTW Fourier analysis on the sphere",
               SPHEREVSYNTHESIS     => "FFTW Fourier synthesis on the sphere (vector field)",
               SPHEREVANALYSIS      => "FFTW Fourier analysis on the sphere (vector field)",
               DISKSYNTHESIS        => "FFTW Chebyshev×Fourier synthesis on the disk",
               DISKANALYSIS         => "FFTW Chebyshev×Fourier analysis on the disk",
               RECTDISKSYNTHESIS    => "FFTW Chebyshev synthesis on the rectangularized disk",
               RECTDISKANALYSIS     => "FFTW Chebyshev analysis on the rectangularized disk",
               TRIANGLESYNTHESIS    => "FFTW Chebyshev synthesis on the triangle",
               TRIANGLEANALYSIS     => "FFTW Chebyshev analysis on the triangle",
               TETRAHEDRONSYNTHESIS => "FFTW Chebyshev synthesis on the tetrahedron",
               TETRAHEDRONANALYSIS  => "FFTW Chebyshev analysis on the tetrahedron",
               SPINSPHERESYNTHESIS  => "FFTW Fourier synthesis on the sphere (spin-weighted)",
               SPINSPHEREANALYSIS   => "FFTW Fourier analysis on the sphere (spin-weighted)",
               SPHERICALISOMETRY    => "Spherical isometry")
    global kind2string
    kind2string(k::Integer) = k2s[Int(k)]
end

struct ft_plan_struct end

mutable struct FTPlan{T, N, K}
    plan::Ptr{ft_plan_struct}
    n::Int
    l::Int
    m::Int
    function FTPlan{T, N, K}(plan::Ptr{ft_plan_struct}, n::Int) where {T, N, K}
        p = new(plan, n)
        finalizer(destroy_plan, p)
        p
    end
    function FTPlan{T, N, K}(plan::Ptr{ft_plan_struct}, n::Int, m::Int) where {T, N, K}
        p = new(plan, n, -1, m)
        finalizer(destroy_plan, p)
        p
    end
    function FTPlan{T, N, K}(plan::Ptr{ft_plan_struct}, n::Int, l::Int, m::Int) where {T, N, K}
        p = new(plan, n, l, m)
        finalizer(destroy_plan, p)
        p
    end
end

eltype(p::FTPlan{T}) where {T} = T
ndims(p::FTPlan{T, N}) where {T, N} = N
show(io::IO, p::FTPlan{T, 1, K}) where {T, K} = print(io, "FastTransforms ", kind2string(K), " plan for $(p.n)-element array of ", T)
show(io::IO, p::FTPlan{T, 2, SPHERE}) where T = print(io, "FastTransforms ", kind2string(SPHERE), " plan for $(p.n)×$(2p.n-1)-element array of ", T)
show(io::IO, p::FTPlan{T, 2, SPHEREV}) where T = print(io, "FastTransforms ", kind2string(SPHEREV), " plan for $(p.n)×$(2p.n-1)-element array of ", T)
show(io::IO, p::FTPlan{T, 2, DISK}) where T = print(io, "FastTransforms ", kind2string(DISK), " plan for $(p.n)×$(4p.n-3)-element array of ", T)
show(io::IO, p::FTPlan{T, 2, RECTDISK}) where T = print(io, "FastTransforms ", kind2string(RECTDISK), " plan for $(p.n)×$(p.n)-element array of ", T)
show(io::IO, p::FTPlan{T, 2, TRIANGLE}) where T = print(io, "FastTransforms ", kind2string(TRIANGLE), " plan for $(p.n)×$(p.n)-element array of ", T)
show(io::IO, p::FTPlan{T, 3, TETRAHEDRON}) where T = print(io, "FastTransforms ", kind2string(TETRAHEDRON), " plan for $(p.n)×$(p.n)×$(p.n)-element array of ", T)
show(io::IO, p::FTPlan{T, 2, SPINSPHERE}) where T = print(io, "FastTransforms ", kind2string(SPINSPHERE), " plan for $(p.n)×$(2p.n-1)-element array of ", T)
show(io::IO, p::FTPlan{T, 2, K}) where {T, K} = print(io, "FastTransforms plan for ", kind2string(K), " for $(p.n)×$(p.m)-element array of ", T)
show(io::IO, p::FTPlan{T, 3, K}) where {T, K} = print(io, "FastTransforms plan for ", kind2string(K), " for $(p.n)×$(p.l)×$(p.m)-element array of ", T)
show(io::IO, p::FTPlan{T, 2, SPHERICALISOMETRY}) where T = print(io, "FastTransforms ", kind2string(SPHERICALISOMETRY), " plan for $(p.n)×$(2p.n-1)-element array of ", T)

function checksize(p::FTPlan{T, 1}, x::Array{T}) where T
    if p.n != size(x, 1)
        throw(DimensionMismatch("FTPlan has dimensions $(p.n) × $(p.n), x has leading dimension $(size(x, 1))"))
    end
end

for (N, K) in ((2, RECTDISK), (2, TRIANGLE), (3, TETRAHEDRON))
    @eval function checksize(p::FTPlan{T, $N, $K}, x::Array{T, $N}) where T
        if p.n != size(x, 1)
            throw(DimensionMismatch("FTPlan has dimensions $(p.n) × $(p.n), x has leading dimension $(size(x, 1))"))
        end
    end
end

for K in (SPHERE, SPHEREV, DISK, SPINSPHERE)
    @eval function checksize(p::FTPlan{T, 2, $K}, x::Matrix{T}) where T
        if p.n != size(x, 1)
            throw(DimensionMismatch("FTPlan has dimensions $(p.n) × $(p.n), x has leading dimension $(size(x, 1))"))
        end
        if iseven(size(x, 2))
            throw(DimensionMismatch("This FTPlan only operates on arrays with an odd number of columns."))
        end
    end
end

function checksize(p::FTPlan{T, 2}, x::Array{T, 2}) where T
    if p.n != size(x, 1) || p.m != size(x, 2)
        throw(DimensionMismatch("FTPlan has dimensions $(p.n) × $(p.m), x has dimensions $(size(x, 1)) × $(size(x, 2))"))
    end
end

function checksize(p::FTPlan{T, 3}, x::Array{T, 3}) where T
    if p.n != size(x, 1) || p.l != size(x, 2) || p.m != size(x, 3)
        throw(DimensionMismatch("FTPlan has dimensions $(p.n) × $(p.l) × $(p.m), x has dimensions $(size(x, 1)) × $(size(x, 2)) × $(size(x, 3))"))
    end
end

function checksize(p::FTPlan{T, 2, SPHERICALISOMETRY}, x::Matrix{T}) where T
    if p.n != size(x, 1) || 2p.n-1 != size(x, 2)
        throw(DimensionMismatch("This FTPlan must operate on arrays of size $(p.n) × $(2p.n-1)."))
    end
end

unsafe_convert(::Type{Ptr{ft_plan_struct}}, p::FTPlan) = p.plan
unsafe_convert(::Type{Ptr{mpfr_t}}, p::FTPlan) = unsafe_convert(Ptr{mpfr_t}, p.plan)

destroy_plan(p::FTPlan{Float32, 1}) = ccall((:ft_destroy_tb_eigen_FMMf, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)
destroy_plan(p::FTPlan{Float64, 1}) = ccall((:ft_destroy_tb_eigen_FMM, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)
destroy_plan(p::FTPlan{BigFloat, 1}) = ccall((:ft_mpfr_destroy_plan, libfasttransforms), Cvoid, (Ptr{mpfr_t}, Cint), p, p.n)
destroy_plan(p::FTPlan{Float32, 1, ASSOCIATEDJAC2JAC}) = ccall((:ft_destroy_btb_eigen_FMMf, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)
destroy_plan(p::FTPlan{Float64, 1, ASSOCIATEDJAC2JAC}) = ccall((:ft_destroy_btb_eigen_FMM, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)
destroy_plan(p::FTPlan{Float64}) = ccall((:ft_destroy_harmonic_plan, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)
destroy_plan(p::FTPlan{Complex{Float64}, 2, SPINSPHERE}) = ccall((:ft_destroy_spin_harmonic_plan, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)
destroy_plan(p::FTPlan{Float64, 2, SPHERESYNTHESIS}) = ccall((:ft_destroy_sphere_fftw_plan, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)
destroy_plan(p::FTPlan{Float64, 2, SPHEREANALYSIS}) = ccall((:ft_destroy_sphere_fftw_plan, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)
destroy_plan(p::FTPlan{Float64, 2, SPHEREVSYNTHESIS}) = ccall((:ft_destroy_sphere_fftw_plan, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)
destroy_plan(p::FTPlan{Float64, 2, SPHEREVANALYSIS}) = ccall((:ft_destroy_sphere_fftw_plan, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)
destroy_plan(p::FTPlan{Float64, 2, DISKSYNTHESIS}) = ccall((:ft_destroy_disk_fftw_plan, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)
destroy_plan(p::FTPlan{Float64, 2, DISKANALYSIS}) = ccall((:ft_destroy_disk_fftw_plan, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)
destroy_plan(p::FTPlan{Float64, 2, RECTDISKSYNTHESIS}) = ccall((:ft_destroy_rectdisk_fftw_plan, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)
destroy_plan(p::FTPlan{Float64, 2, RECTDISKANALYSIS}) = ccall((:ft_destroy_rectdisk_fftw_plan, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)
destroy_plan(p::FTPlan{Float64, 2, TRIANGLESYNTHESIS}) = ccall((:ft_destroy_triangle_fftw_plan, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)
destroy_plan(p::FTPlan{Float64, 2, TRIANGLEANALYSIS}) = ccall((:ft_destroy_triangle_fftw_plan, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)
destroy_plan(p::FTPlan{Float64, 3, TETRAHEDRONSYNTHESIS}) = ccall((:ft_destroy_tetrahedron_fftw_plan, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)
destroy_plan(p::FTPlan{Float64, 3, TETRAHEDRONANALYSIS}) = ccall((:ft_destroy_tetrahedron_fftw_plan, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)
destroy_plan(p::FTPlan{Complex{Float64}, 2, SPINSPHERESYNTHESIS}) = ccall((:ft_destroy_spinsphere_fftw_plan, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)
destroy_plan(p::FTPlan{Complex{Float64}, 2, SPINSPHEREANALYSIS}) = ccall((:ft_destroy_spinsphere_fftw_plan, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)
destroy_plan(p::FTPlan{Float64, 2, SPHERICALISOMETRY}) = ccall((:ft_destroy_sph_isometry_plan, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ), p)

struct AdjointFTPlan{T, S, R}
    parent::S
    adjoint::R
    function AdjointFTPlan{T, S, R}(parent::S) where {T, S, R}
        new(parent)
    end
    function AdjointFTPlan{T, S, R}(parent::S, adjoint::R) where {T, S, R}
        new(parent, adjoint)
    end
end

AdjointFTPlan(p::FTPlan) = AdjointFTPlan{eltype(p), typeof(p), typeof(p)}(p)
AdjointFTPlan(p::FTPlan, q::FTPlan) = AdjointFTPlan{eltype(q), typeof(p), typeof(q)}(p, q)

adjoint(p::FTPlan) = AdjointFTPlan(p)
adjoint(p::AdjointFTPlan) = p.parent

eltype(p::AdjointFTPlan{T}) where T = T
ndims(p::AdjointFTPlan) = ndims(p.parent)
function show(io::IO, p::AdjointFTPlan)
    print(io, "Adjoint ")
    show(io, p.parent)
end

function checksize(p::AdjointFTPlan, x)
    try
        checksize(p.adjoint, x)
    catch
        checksize(p.parent, x)
    end
end

function unsafe_convert(::Type{Ptr{ft_plan_struct}}, p::AdjointFTPlan)
    try
        unsafe_convert(Ptr{ft_plan_struct}, p.adjoint)
    catch
        unsafe_convert(Ptr{ft_plan_struct}, p.parent)
    end
end

function unsafe_convert(::Type{Ptr{mpfr_t}}, p::AdjointFTPlan)
    try
        unsafe_convert(Ptr{mpfr_t}, p.adjoint)
    catch
        unsafe_convert(Ptr{mpfr_t}, p.parent)
    end
end

struct TransposeFTPlan{T, S, R}
    parent::S
    transpose::R
    function TransposeFTPlan{T, S, R}(parent::S) where {T, S, R}
        new(parent)
    end
    function TransposeFTPlan{T, S, R}(parent::S, transpose::R) where {T, S, R}
        new(parent, transpose)
    end
end

TransposeFTPlan(p::FTPlan) = TransposeFTPlan{eltype(p), typeof(p), typeof(p)}(p)
TransposeFTPlan(p::FTPlan, q::FTPlan) = TransposeFTPlan{eltype(q), typeof(p), typeof(q)}(p, q)

transpose(p::FTPlan) = TransposeFTPlan(p)
transpose(p::TransposeFTPlan) = p.parent

eltype(p::TransposeFTPlan{T}) where T = T
ndims(p::TransposeFTPlan) = ndims(p.parent)
function show(io::IO, p::TransposeFTPlan)
    print(io, "Transpose ")
    show(io, p.parent)
end

function checksize(p::TransposeFTPlan, x)
    try
        checksize(p.transpose, x)
    catch
        checksize(p.parent, x)
    end
end

function unsafe_convert(::Type{Ptr{ft_plan_struct}}, p::TransposeFTPlan)
    try
        unsafe_convert(Ptr{ft_plan_struct}, p.transpose)
    catch
        unsafe_convert(Ptr{ft_plan_struct}, p.parent)
    end
end

function unsafe_convert(::Type{Ptr{mpfr_t}}, p::TransposeFTPlan)
    try
        unsafe_convert(Ptr{mpfr_t}, p.transpose)
    catch
        unsafe_convert(Ptr{mpfr_t}, p.parent)
    end
end

for f in (:leg2cheb, :cheb2leg, :ultra2ultra, :jac2jac,
          :lag2lag, :jac2ultra, :ultra2jac, :jac2cheb,
          :cheb2jac, :ultra2cheb, :cheb2ultra, :associatedjac2jac,
          :sph2fourier, :sphv2fourier, :disk2cxf,
          :rectdisk2cheb, :tri2cheb, :tet2cheb)
    plan_f = Symbol("plan_", f)
    @eval begin
        $plan_f(x::AbstractArray{T}, y...; z...) where T = $plan_f(T, size(x, 1), y...; z...)
        $plan_f(::Type{Complex{T}}, y...; z...) where T <: Real = $plan_f(T, y...; z...)
        $f(x::AbstractArray, y...; z...) = $plan_f(x, y...; z...)*x
    end
end

for (f, plan_f) in ((:fourier2sph, :plan_sph2fourier), (:fourier2sphv, :plan_sphv2fourier),
                    (:cxf2disk, :plan_disk2cxf), (:cheb2rectdisk, :plan_rectdisk2cheb),
                    (:cheb2tri, :plan_tri2cheb), (:cheb2tet, :plan_tet2cheb))
    @eval begin
        $f(x::AbstractArray, y...; z...) = $plan_f(x, y...; z...)\x
    end
end

plan_spinsph2fourier(x::AbstractArray{T}, y...; z...) where T = plan_spinsph2fourier(T, size(x, 1), y...; z...)
spinsph2fourier(x::AbstractArray, y...; z...) = plan_spinsph2fourier(x, y...; z...)*x
fourier2spinsph(x::AbstractArray, y...; z...) = plan_spinsph2fourier(x, y...; z...)\x

function plan_leg2cheb(::Type{Float32}, n::Integer; normleg::Bool=false, normcheb::Bool=false)
    plan = ccall((:ft_plan_legendre_to_chebyshevf, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint), normleg, normcheb, n)
    return FTPlan{Float32, 1, LEG2CHEB}(plan, n)
end

function plan_cheb2leg(::Type{Float32}, n::Integer; normcheb::Bool=false, normleg::Bool=false)
    plan = ccall((:ft_plan_chebyshev_to_legendref, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint), normcheb, normleg, n)
    return FTPlan{Float32, 1, CHEB2LEG}(plan, n)
end

function plan_ultra2ultra(::Type{Float32}, n::Integer, λ, μ; norm1::Bool=false, norm2::Bool=false)
    plan = ccall((:ft_plan_ultraspherical_to_ultrasphericalf, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Float32, Float32), norm1, norm2, n, λ, μ)
    return FTPlan{Float32, 1, ULTRA2ULTRA}(plan, n)
end

function plan_jac2jac(::Type{Float32}, n::Integer, α, β, γ, δ; norm1::Bool=false, norm2::Bool=false)
    plan = ccall((:ft_plan_jacobi_to_jacobif, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Float32, Float32, Float32, Float32), norm1, norm2, n, α, β, γ, δ)
    return FTPlan{Float32, 1, JAC2JAC}(plan, n)
end

function plan_lag2lag(::Type{Float32}, n::Integer, α, β; norm1::Bool=false, norm2::Bool=false)
    plan = ccall((:ft_plan_laguerre_to_laguerref, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Float32, Float32), norm1, norm2, n, α, β)
    return FTPlan{Float32, 1, LAG2LAG}(plan, n)
end

function plan_jac2ultra(::Type{Float32}, n::Integer, α, β, λ; normjac::Bool=false, normultra::Bool=false)
    plan = ccall((:ft_plan_jacobi_to_ultrasphericalf, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Float32, Float32, Float32), normjac, normultra, n, α, β, λ)
    return FTPlan{Float32, 1, JAC2ULTRA}(plan, n)
end

function plan_ultra2jac(::Type{Float32}, n::Integer, λ, α, β; normultra::Bool=false, normjac::Bool=false)
    plan = ccall((:ft_plan_ultraspherical_to_jacobif, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Float32, Float32, Float32), normultra, normjac, n, λ, α, β)
    return FTPlan{Float32, 1, ULTRA2JAC}(plan, n)
end

function plan_jac2cheb(::Type{Float32}, n::Integer, α, β; normjac::Bool=false, normcheb::Bool=false)
    plan = ccall((:ft_plan_jacobi_to_chebyshevf, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Float32, Float32), normjac, normcheb, n, α, β)
    return FTPlan{Float32, 1, JAC2CHEB}(plan, n)
end

function plan_cheb2jac(::Type{Float32}, n::Integer, α, β; normcheb::Bool=false, normjac::Bool=false)
    plan = ccall((:ft_plan_chebyshev_to_jacobif, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Float32, Float32), normcheb, normjac, n, α, β)
    return FTPlan{Float32, 1, CHEB2JAC}(plan, n)
end

function plan_ultra2cheb(::Type{Float32}, n::Integer, λ; normultra::Bool=false, normcheb::Bool=false)
    plan = ccall((:ft_plan_ultraspherical_to_chebyshevf, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Float32), normultra, normcheb, n, λ)
    return FTPlan{Float32, 1, ULTRA2CHEB}(plan, n)
end

function plan_cheb2ultra(::Type{Float32}, n::Integer, λ; normcheb::Bool=false, normultra::Bool=false)
    plan = ccall((:ft_plan_chebyshev_to_ultrasphericalf, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Float32), normcheb, normultra, n, λ)
    return FTPlan{Float32, 1, CHEB2ULTRA}(plan, n)
end

function plan_associatedjac2jac(::Type{Float32}, n::Integer, c::Integer, α, β, γ, δ; norm1::Bool=false, norm2::Bool=false)
    plan = ccall((:ft_plan_associated_jacobi_to_jacobif, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Cint, Float32, Float32, Float32, Float32), norm1, norm2, n, c, α, β, γ, δ)
    return FTPlan{Float32, 1, ASSOCIATEDJAC2JAC}(plan, n)
end


function plan_leg2cheb(::Type{Float64}, n::Integer; normleg::Bool=false, normcheb::Bool=false)
    plan = ccall((:ft_plan_legendre_to_chebyshev, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint), normleg, normcheb, n)
    return FTPlan{Float64, 1, LEG2CHEB}(plan, n)
end

function plan_cheb2leg(::Type{Float64}, n::Integer; normcheb::Bool=false, normleg::Bool=false)
    plan = ccall((:ft_plan_chebyshev_to_legendre, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint), normcheb, normleg, n)
    return FTPlan{Float64, 1, CHEB2LEG}(plan, n)
end

function plan_ultra2ultra(::Type{Float64}, n::Integer, λ, μ; norm1::Bool=false, norm2::Bool=false)
    plan = ccall((:ft_plan_ultraspherical_to_ultraspherical, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Float64, Float64), norm1, norm2, n, λ, μ)
    return FTPlan{Float64, 1, ULTRA2ULTRA}(plan, n)
end

function plan_jac2jac(::Type{Float64}, n::Integer, α, β, γ, δ; norm1::Bool=false, norm2::Bool=false)
    plan = ccall((:ft_plan_jacobi_to_jacobi, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Float64, Float64, Float64, Float64), norm1, norm2, n, α, β, γ, δ)
    return FTPlan{Float64, 1, JAC2JAC}(plan, n)
end

function plan_lag2lag(::Type{Float64}, n::Integer, α, β; norm1::Bool=false, norm2::Bool=false)
    plan = ccall((:ft_plan_laguerre_to_laguerre, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Float64, Float64), norm1, norm2, n, α, β)
    return FTPlan{Float64, 1, LAG2LAG}(plan, n)
end

function plan_jac2ultra(::Type{Float64}, n::Integer, α, β, λ; normjac::Bool=false, normultra::Bool=false)
    plan = ccall((:ft_plan_jacobi_to_ultraspherical, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Float64, Float64, Float64), normjac, normultra, n, α, β, λ)
    return FTPlan{Float64, 1, JAC2ULTRA}(plan, n)
end

function plan_ultra2jac(::Type{Float64}, n::Integer, λ, α, β; normultra::Bool=false, normjac::Bool=false)
    plan = ccall((:ft_plan_ultraspherical_to_jacobi, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Float64, Float64, Float64), normultra, normjac, n, λ, α, β)
    return FTPlan{Float64, 1, ULTRA2JAC}(plan, n)
end

function plan_jac2cheb(::Type{Float64}, n::Integer, α, β; normjac::Bool=false, normcheb::Bool=false)
    plan = ccall((:ft_plan_jacobi_to_chebyshev, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Float64, Float64), normjac, normcheb, n, α, β)
    return FTPlan{Float64, 1, JAC2CHEB}(plan, n)
end

function plan_cheb2jac(::Type{Float64}, n::Integer, α, β; normcheb::Bool=false, normjac::Bool=false)
    plan = ccall((:ft_plan_chebyshev_to_jacobi, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Float64, Float64), normcheb, normjac, n, α, β)
    return FTPlan{Float64, 1, CHEB2JAC}(plan, n)
end

function plan_ultra2cheb(::Type{Float64}, n::Integer, λ; normultra::Bool=false, normcheb::Bool=false)
    plan = ccall((:ft_plan_ultraspherical_to_chebyshev, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Float64), normultra, normcheb, n, λ)
    return FTPlan{Float64, 1, ULTRA2CHEB}(plan, n)
end

function plan_cheb2ultra(::Type{Float64}, n::Integer, λ; normcheb::Bool=false, normultra::Bool=false)
    plan = ccall((:ft_plan_chebyshev_to_ultraspherical, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Float64), normcheb, normultra, n, λ)
    return FTPlan{Float64, 1, CHEB2ULTRA}(plan, n)
end

function plan_associatedjac2jac(::Type{Float64}, n::Integer, c::Integer, α, β, γ, δ; norm1::Bool=false, norm2::Bool=false)
    plan = ccall((:ft_plan_associated_jacobi_to_jacobi, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Cint, Float64, Float64, Float64, Float64), norm1, norm2, n, c, α, β, γ, δ)
    return FTPlan{Float64, 1, ASSOCIATEDJAC2JAC}(plan, n)
end


function plan_leg2cheb(::Type{BigFloat}, n::Integer; normleg::Bool=false, normcheb::Bool=false)
    plan = ccall((:ft_mpfr_plan_legendre_to_chebyshev, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Clong, Int32), normleg, normcheb, n, precision(BigFloat), Base.MPFR.ROUNDING_MODE[])
    return FTPlan{BigFloat, 1, LEG2CHEB}(plan, n)
end

function plan_cheb2leg(::Type{BigFloat}, n::Integer; normcheb::Bool=false, normleg::Bool=false)
    plan = ccall((:ft_mpfr_plan_chebyshev_to_legendre, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Clong, Int32), normcheb, normleg, n, precision(BigFloat), Base.MPFR.ROUNDING_MODE[])
    return FTPlan{BigFloat, 1, CHEB2LEG}(plan, n)
end

function plan_ultra2ultra(::Type{BigFloat}, n::Integer, λ, μ; norm1::Bool=false, norm2::Bool=false)
    plan = ccall((:ft_mpfr_plan_ultraspherical_to_ultraspherical, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Ref{BigFloat}, Ref{BigFloat}, Clong, Int32), norm1, norm2, n, λ, μ, precision(BigFloat), Base.MPFR.ROUNDING_MODE[])
    return FTPlan{BigFloat, 1, ULTRA2ULTRA}(plan, n)
end

function plan_jac2jac(::Type{BigFloat}, n::Integer, α, β, γ, δ; norm1::Bool=false, norm2::Bool=false)
    plan = ccall((:ft_mpfr_plan_jacobi_to_jacobi, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, Clong, Int32), norm1, norm2, n, α, β, γ, δ, precision(BigFloat), Base.MPFR.ROUNDING_MODE[])
    return FTPlan{BigFloat, 1, JAC2JAC}(plan, n)
end

function plan_lag2lag(::Type{BigFloat}, n::Integer, α, β; norm1::Bool=false, norm2::Bool=false)
    plan = ccall((:ft_mpfr_plan_laguerre_to_laguerre, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Ref{BigFloat}, Ref{BigFloat}, Clong, Int32), norm1, norm2, n, α, β, precision(BigFloat), Base.MPFR.ROUNDING_MODE[])
    return FTPlan{BigFloat, 1, LAG2LAG}(plan, n)
end

function plan_jac2ultra(::Type{BigFloat}, n::Integer, α, β, λ; normjac::Bool=false, normultra::Bool=false)
    plan = ccall((:ft_mpfr_plan_jacobi_to_ultraspherical, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, Clong, Int32), normjac, normultra, n, α, β, λ, precision(BigFloat), Base.MPFR.ROUNDING_MODE[])
    return FTPlan{BigFloat, 1, JAC2ULTRA}(plan, n)
end

function plan_ultra2jac(::Type{BigFloat}, n::Integer, λ, α, β; normultra::Bool=false, normjac::Bool=false)
    plan = ccall((:ft_mpfr_plan_ultraspherical_to_jacobi, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, Clong, Int32), normultra, normjac, n, λ, α, β, precision(BigFloat), Base.MPFR.ROUNDING_MODE[])
    return FTPlan{BigFloat, 1, ULTRA2JAC}(plan, n)
end

function plan_jac2cheb(::Type{BigFloat}, n::Integer, α, β; normjac::Bool=false, normcheb::Bool=false)
    plan = ccall((:ft_mpfr_plan_jacobi_to_chebyshev, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Ref{BigFloat}, Ref{BigFloat}, Clong, Int32), normjac, normcheb, n, α, β, precision(BigFloat), Base.MPFR.ROUNDING_MODE[])
    return FTPlan{BigFloat, 1, JAC2CHEB}(plan, n)
end

function plan_cheb2jac(::Type{BigFloat}, n::Integer, α, β; normcheb::Bool=false, normjac::Bool=false)
    plan = ccall((:ft_mpfr_plan_chebyshev_to_jacobi, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Ref{BigFloat}, Ref{BigFloat}, Clong, Int32), normcheb, normjac, n, α, β, precision(BigFloat), Base.MPFR.ROUNDING_MODE[])
    return FTPlan{BigFloat, 1, CHEB2JAC}(plan, n)
end

function plan_ultra2cheb(::Type{BigFloat}, n::Integer, λ; normultra::Bool=false, normcheb::Bool=false)
    plan = ccall((:ft_mpfr_plan_ultraspherical_to_chebyshev, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Ref{BigFloat}, Clong, Int32), normultra, normcheb, n, λ, precision(BigFloat), Base.MPFR.ROUNDING_MODE[])
    return FTPlan{BigFloat, 1, ULTRA2CHEB}(plan, n)
end

function plan_cheb2ultra(::Type{BigFloat}, n::Integer, λ; normcheb::Bool=false, normultra::Bool=false)
    plan = ccall((:ft_mpfr_plan_chebyshev_to_ultraspherical, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint, Ref{BigFloat}, Clong, Int32), normcheb, normultra, n, λ, precision(BigFloat), Base.MPFR.ROUNDING_MODE[])
    return FTPlan{BigFloat, 1, CHEB2ULTRA}(plan, n)
end


function plan_sph2fourier(::Type{Float64}, n::Integer)
    plan = ccall((:ft_plan_sph2fourier, libfasttransforms), Ptr{ft_plan_struct}, (Cint, ), n)
    return FTPlan{Float64, 2, SPHERE}(plan, n)
end

function plan_sphv2fourier(::Type{Float64}, n::Integer)
    plan = ccall((:ft_plan_sph2fourier, libfasttransforms), Ptr{ft_plan_struct}, (Cint, ), n)
    return FTPlan{Float64, 2, SPHEREV}(plan, n)
end

function plan_disk2cxf(::Type{Float64}, n::Integer, α, β)
    plan = ccall((:ft_plan_disk2cxf, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Float64, Float64), n, α, β)
    return FTPlan{Float64, 2, DISK}(plan, n)
end

function plan_rectdisk2cheb(::Type{Float64}, n::Integer, β)
    plan = ccall((:ft_plan_rectdisk2cheb, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Float64), n, β)
    return FTPlan{Float64, 2, RECTDISK}(plan, n)
end

function plan_tri2cheb(::Type{Float64}, n::Integer, α, β, γ)
    plan = ccall((:ft_plan_tri2cheb, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Float64, Float64, Float64), n, α, β, γ)
    return FTPlan{Float64, 2, TRIANGLE}(plan, n)
end

function plan_tet2cheb(::Type{Float64}, n::Integer, α, β, γ, δ)
    plan = ccall((:ft_plan_tet2cheb, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Float64, Float64, Float64, Float64), n, α, β, γ, δ)
    return FTPlan{Float64, 3, TETRAHEDRON}(plan, n)
end

function plan_spinsph2fourier(::Type{Complex{Float64}}, n::Integer, s::Integer)
    plan = ccall((:ft_plan_spinsph2fourier, libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint), n, s)
    return FTPlan{Complex{Float64}, 2, SPINSPHERE}(plan, n)
end

for (fJ, fadJ, fC, fE, K) in ((:plan_sph_synthesis, :plan_sph_analysis, :ft_plan_sph_synthesis, :ft_execute_sph_synthesis, SPHERESYNTHESIS),
                              (:plan_sph_analysis, :plan_sph_synthesis, :ft_plan_sph_analysis, :ft_execute_sph_analysis, SPHEREANALYSIS),
                              (:plan_sphv_synthesis, :plan_sphv_analysis, :ft_plan_sphv_synthesis, :ft_execute_sphv_synthesis, SPHEREVSYNTHESIS),
                              (:plan_sphv_analysis, :plan_sphv_synthesis, :ft_plan_sphv_analysis, :ft_execute_sphv_analysis, SPHEREVANALYSIS),
                              (:plan_disk_synthesis, :plan_disk_analysis, :ft_plan_disk_synthesis, :ft_execute_disk_synthesis, DISKSYNTHESIS),
                              (:plan_disk_analysis, :plan_disk_synthesis, :ft_plan_disk_analysis, :ft_execute_disk_analysis, DISKANALYSIS),
                              (:plan_rectdisk_synthesis, :plan_rectdisk_analysis, :ft_plan_rectdisk_synthesis, :ft_execute_rectdisk_synthesis, RECTDISKSYNTHESIS),
                              (:plan_rectdisk_analysis, :plan_rectdisk_synthesis, :ft_plan_rectdisk_analysis, :ft_execute_rectdisk_analysis, RECTDISKANALYSIS),
                              (:plan_tri_synthesis, :plan_tri_analysis, :ft_plan_tri_synthesis, :ft_execute_tri_synthesis, TRIANGLESYNTHESIS),
                              (:plan_tri_analysis, :plan_tri_synthesis, :ft_plan_tri_analysis, :ft_execute_tri_analysis, TRIANGLEANALYSIS))
    @eval begin
        $fJ(x::Matrix{T}) where T = $fJ(T, size(x, 1), size(x, 2))
        $fJ(::Type{Complex{T}}, x...) where T <: Real = $fJ(T, x...)
        function $fJ(::Type{Float64}, n::Integer, m::Integer)
            plan = ccall(($(string(fC)), libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint), n, m)
            return FTPlan{Float64, 2, $K}(plan, n, m)
        end
        adjoint(p::FTPlan{T, 2, $K}) where T = AdjointFTPlan(p, $fadJ(T, p.n, p.m))
        transpose(p::FTPlan{T, 2, $K}) where T = TransposeFTPlan(p, $fadJ(T, p.n, p.m))
        function lmul!(p::FTPlan{Float64, 2, $K}, x::Matrix{Float64})
            checksize(p, x)
            ccall(($(string(fE)), libfasttransforms), Cvoid, (Cint, Ptr{ft_plan_struct}, Ptr{Float64}, Cint, Cint), 'N', p, x, size(x, 1), size(x, 2))
            return x
        end
        function lmul!(p::AdjointFTPlan{Float64, FTPlan{Float64, 2, $K}}, x::Matrix{Float64})
            checksize(p, x)
            ccall(($(string(fE)), libfasttransforms), Cvoid, (Cint, Ptr{ft_plan_struct}, Ptr{Float64}, Cint, Cint), 'T', p, x, size(x, 1), size(x, 2))
            return x
        end
        function lmul!(p::TransposeFTPlan{Float64, FTPlan{Float64, 2, $K}}, x::Matrix{Float64})
            checksize(p, x)
            ccall(($(string(fE)), libfasttransforms), Cvoid, (Cint, Ptr{ft_plan_struct}, Ptr{Float64}, Cint, Cint), 'T', p, x, size(x, 1), size(x, 2))
            return x
        end
    end
end

for (fJ, fadJ, fC, fE, K) in ((:plan_tet_synthesis, :plan_tet_analysis, :ft_plan_tet_synthesis, :ft_execute_tet_synthesis, TETRAHEDRONSYNTHESIS),
                              (:plan_tet_analysis, :plan_tet_synthesis, :ft_plan_tet_analysis, :ft_execute_tet_analysis, TETRAHEDRONANALYSIS))
    @eval begin
        $fJ(x::Array{T, 3}) where T = $fJ(T, size(x, 1), size(x, 2), size(x, 3))
        $fJ(::Type{Complex{T}}, x...) where T <: Real = $fJ(T, x...)
        function $fJ(::Type{Float64}, n::Integer, l::Integer, m::Integer)
            plan = ccall(($(string(fC)), libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint), n, l, m)
            return FTPlan{Float64, 3, $K}(plan, n, l, m)
        end
        adjoint(p::FTPlan{T, 3, $K}) where T = AdjointFTPlan(p, $fadJ(T, p.n, p.l, p.m))
        transpose(p::FTPlan{T, 3, $K}) where T = TransposeFTPlan(p, $fadJ(T, p.n, p.l, p.m))
        function lmul!(p::FTPlan{Float64, 3, $K}, x::Array{Float64, 3})
            checksize(p, x)
            ccall(($(string(fE)), libfasttransforms), Cvoid, (Cint, Ptr{ft_plan_struct}, Ptr{Float64}, Cint, Cint, Cint), 'N', p, x, size(x, 1), size(x, 2), size(x, 3))
            return x
        end
        function lmul!(p::AdjointFTPlan{Float64, FTPlan{Float64, 3, $K}}, x::Array{Float64, 3})
            checksize(p, x)
            ccall(($(string(fE)), libfasttransforms), Cvoid, (Cint, Ptr{ft_plan_struct}, Ptr{Float64}, Cint, Cint, Cint), 'T', p, x, size(x, 1), size(x, 2), size(x, 3))
            return x
        end
        function lmul!(p::TransposeFTPlan{Float64, FTPlan{Float64, 3, $K}}, x::Array{Float64, 3})
            checksize(p, x)
            ccall(($(string(fE)), libfasttransforms), Cvoid, (Cint, Ptr{ft_plan_struct}, Ptr{Float64}, Cint, Cint, Cint), 'T', p, x, size(x, 1), size(x, 2), size(x, 3))
            return x
        end
    end
end

for (fJ, fadJ, fC, fE, K) in ((:plan_spinsph_synthesis, :plan_spinsph_analysis, :ft_plan_spinsph_synthesis, :ft_execute_spinsph_synthesis, SPINSPHERESYNTHESIS),
                              (:plan_spinsph_analysis, :plan_spinsph_synthesis, :ft_plan_spinsph_analysis, :ft_execute_spinsph_analysis, SPINSPHEREANALYSIS))
    @eval begin
        $fJ(x::Matrix{T}, s::Integer) where T = $fJ(T, size(x, 1), size(x, 2), s)
        function $fJ(::Type{Complex{Float64}}, n::Integer, m::Integer, s::Integer)
            plan = ccall(($(string(fC)), libfasttransforms), Ptr{ft_plan_struct}, (Cint, Cint, Cint), n, m, s)
            return FTPlan{Complex{Float64}, 2, $K}(plan, n, m)
        end
        get_spin(p::FTPlan{T, 2, $K}) where T = ccall((:ft_get_spin_spinsphere_fftw_plan, libfasttransforms), Cint, (Ptr{ft_plan_struct},), p)
        adjoint(p::FTPlan{T, 2, $K}) where T = AdjointFTPlan(p, $fadJ(T, p.n, p.m, get_spin(p)))
        transpose(p::FTPlan{T, 2, $K}) where T = TransposeFTPlan(p, $fadJ(T, p.n, p.m, get_spin(p)))
        function lmul!(p::FTPlan{Complex{Float64}, 2, $K}, x::Matrix{Complex{Float64}})
            checksize(p, x)
            ccall(($(string(fE)), libfasttransforms), Cvoid, (Cint, Ptr{ft_plan_struct}, Ptr{Float64}, Cint, Cint), 'N', p, x, size(x, 1), size(x, 2))
            return x
        end
        function lmul!(p::AdjointFTPlan{Complex{Float64}, FTPlan{Complex{Float64}, 2, $K}}, x::Matrix{Complex{Float64}})
            checksize(p, x)
            ccall(($(string(fE)), libfasttransforms), Cvoid, (Cint, Ptr{ft_plan_struct}, Ptr{Float64}, Cint, Cint), 'C', p, x, size(x, 1), size(x, 2))
            return x
        end
        function lmul!(p::TransposeFTPlan{Complex{Float64}, FTPlan{Complex{Float64}, 2, $K}}, x::Matrix{Complex{Float64}})
            checksize(p, x)
            conj!(x)
            ccall(($(string(fE)), libfasttransforms), Cvoid, (Cint, Ptr{ft_plan_struct}, Ptr{Float64}, Cint, Cint), 'C', p, x, size(x, 1), size(x, 2))
            conj!(x)
            return x
        end
    end
end

function plan_sph_isometry(::Type{Float64}, n::Integer)
    plan = ccall((:ft_plan_sph_isometry, libfasttransforms), Ptr{ft_plan_struct}, (Cint, ), n)
    return FTPlan{Float64, 2, SPHERICALISOMETRY}(plan, n)
end

*(p::FTPlan{T}, x::AbstractArray{T}) where T = lmul!(p, Array(x))
*(p::AdjointFTPlan{T}, x::AbstractArray{T}) where T = lmul!(p, Array(x))
*(p::TransposeFTPlan{T}, x::AbstractArray{T}) where T = lmul!(p, Array(x))
\(p::FTPlan{T}, x::AbstractArray{T}) where T = ldiv!(p, Array(x))
\(p::AdjointFTPlan{T}, x::AbstractArray{T}) where T = ldiv!(p, Array(x))
\(p::TransposeFTPlan{T}, x::AbstractArray{T}) where T = ldiv!(p, Array(x))

*(p::FTPlan{T, 1}, x::UniformScaling{S}) where {T, S} = lmul!(p, Matrix{promote_type(T, S)}(x, p.n, p.n))
*(p::AdjointFTPlan{T, FTPlan{T, 1, K}}, x::UniformScaling{S}) where {T, S, K} = lmul!(p, Matrix{promote_type(T, S)}(x, p.parent.n, p.parent.n))
*(p::TransposeFTPlan{T, FTPlan{T, 1, K}}, x::UniformScaling{S}) where {T, S, K} = lmul!(p, Matrix{promote_type(T, S)}(x, p.parent.n, p.parent.n))
\(p::FTPlan{T, 1}, x::UniformScaling{S}) where {T, S} = ldiv!(p, Matrix{promote_type(T, S)}(x, p.n, p.n))
\(p::AdjointFTPlan{T, FTPlan{T, 1, K}}, x::UniformScaling{S}) where {T, S, K} = ldiv!(p, Matrix{promote_type(T, S)}(x, p.parent.n, p.parent.n))
\(p::TransposeFTPlan{T, FTPlan{T, 1, K}}, x::UniformScaling{S}) where {T, S, K} = ldiv!(p, Matrix{promote_type(T, S)}(x, p.parent.n, p.parent.n))

for (fJ, fC, elty) in ((:lmul!, :ft_bfmvf, :Float32),
                       (:ldiv!, :ft_bfsvf, :Float32),
                       (:lmul!, :ft_bfmv , :Float64),
                       (:ldiv!, :ft_bfsv , :Float64))
    @eval begin
        function $fJ(p::FTPlan{$elty, 1}, x::Vector{$elty})
            checksize(p, x)
            ccall(($(string(fC)), libfasttransforms), Cvoid, (Cint, Ptr{ft_plan_struct}, Ptr{$elty}), 'N', p, x)
            return x
        end
        function $fJ(p::AdjointFTPlan{$elty, FTPlan{$elty, 1, K}}, x::Vector{$elty}) where K
            checksize(p, x)
            ccall(($(string(fC)), libfasttransforms), Cvoid, (Cint, Ptr{ft_plan_struct}, Ptr{$elty}), 'T', p, x)
            return x
        end
        function $fJ(p::TransposeFTPlan{$elty, FTPlan{$elty, 1, K}}, x::Vector{$elty}) where K
            checksize(p, x)
            ccall(($(string(fC)), libfasttransforms), Cvoid, (Cint, Ptr{ft_plan_struct}, Ptr{$elty}), 'T', p, x)
            return x
        end
    end
end

for (fJ, fC, elty) in ((:lmul!, :ft_bbbfmvf, :Float32),
                       (:lmul!, :ft_bbbfmv , :Float64))
    @eval begin
        function $fJ(p::FTPlan{$elty, 1, ASSOCIATEDJAC2JAC}, x::Vector{$elty})
            checksize(p, x)
            ccall(($(string(fC)), libfasttransforms), Cvoid, (Cint, Cint, Cint, Ptr{ft_plan_struct}, Ptr{$elty}), 'N', '2', '1', p, x)
            return x
        end
        function $fJ(p::AdjointFTPlan{$elty, FTPlan{$elty, 1, ASSOCIATEDJAC2JAC}}, x::Vector{$elty})
            checksize(p, x)
            ccall(($(string(fC)), libfasttransforms), Cvoid, (Cint, Cint, Cint, Ptr{ft_plan_struct}, Ptr{$elty}), 'T', '1', '2', p, x)
            return x
        end
        function $fJ(p::TransposeFTPlan{$elty, FTPlan{$elty, 1, ASSOCIATEDJAC2JAC}}, x::Vector{$elty})
            checksize(p, x)
            ccall(($(string(fC)), libfasttransforms), Cvoid, (Cint, Cint, Cint, Ptr{ft_plan_struct}, Ptr{$elty}), 'T', '1', '2', p, x)
            return x
        end
    end
end

for (fJ, fC) in ((:lmul!, :ft_mpfr_trmv_ptr),
                 (:ldiv!, :ft_mpfr_trsv_ptr))
    @eval begin
        function $fJ(p::FTPlan{BigFloat, 1}, x::Vector{BigFloat})
            checksize(p, x)
            ccall(($(string(fC)), libfasttransforms), Cvoid, (Cint, Cint, Ptr{mpfr_t}, Cint, Ptr{BigFloat}, Int32), 'N', p.n, p, p.n, renew!(x), Base.MPFR.ROUNDING_MODE[])
            return x
        end
        function $fJ(p::AdjointFTPlan{BigFloat, FTPlan{BigFloat, 1, K}}, x::Vector{BigFloat}) where K
            checksize(p, x)
            ccall(($(string(fC)), libfasttransforms), Cvoid, (Cint, Cint, Ptr{mpfr_t}, Cint, Ptr{BigFloat}, Int32), 'T', p.parent.n, p, p.parent.n, renew!(x), Base.MPFR.ROUNDING_MODE[])
            return x
        end
        function $fJ(p::TransposeFTPlan{BigFloat, FTPlan{BigFloat, 1, K}}, x::Vector{BigFloat}) where K
            checksize(p, x)
            ccall(($(string(fC)), libfasttransforms), Cvoid, (Cint, Cint, Ptr{mpfr_t}, Cint, Ptr{BigFloat}, Int32), 'T', p.parent.n, p, p.parent.n, renew!(x), Base.MPFR.ROUNDING_MODE[])
            return x
        end
    end
end

for (fJ, fC, elty) in ((:lmul!, :ft_bfmmf, :Float32),
                       (:ldiv!, :ft_bfsmf, :Float32),
                       (:lmul!, :ft_bfmm , :Float64),
                       (:ldiv!, :ft_bfsm , :Float64))
    @eval begin
        function $fJ(p::FTPlan{$elty, 1}, x::Matrix{$elty})
            checksize(p, x)
            ccall(($(string(fC)), libfasttransforms), Cvoid, (Cint, Ptr{ft_plan_struct}, Ptr{$elty}, Cint, Cint), 'N', p, x, size(x, 1), size(x, 2))
            return x
        end
        function $fJ(p::AdjointFTPlan{$elty, FTPlan{$elty, 1, K}}, x::Matrix{$elty}) where K
            checksize(p, x)
            ccall(($(string(fC)), libfasttransforms), Cvoid, (Cint, Ptr{ft_plan_struct}, Ptr{$elty}, Cint, Cint), 'T', p, x, size(x, 1), size(x, 2))
            return x
        end
        function $fJ(p::TransposeFTPlan{$elty, FTPlan{$elty, 1, K}}, x::Matrix{$elty}) where K
            checksize(p, x)
            ccall(($(string(fC)), libfasttransforms), Cvoid, (Cint, Ptr{ft_plan_struct}, Ptr{$elty}, Cint, Cint), 'T', p, x, size(x, 1), size(x, 2))
            return x
        end
    end
end

for (fJ, fC, elty) in ((:lmul!, :ft_bbbfmmf, :Float32),
                       (:lmul!, :ft_bbbfmm , :Float64))
    @eval begin
        function $fJ(p::FTPlan{$elty, 1, ASSOCIATEDJAC2JAC}, x::Matrix{$elty})
            checksize(p, x)
            ccall(($(string(fC)), libfasttransforms), Cvoid, (Cint, Cint, Cint, Ptr{ft_plan_struct}, Ptr{$elty}, Cint, Cint), 'N', '2', '1', p, x, size(x, 1), size(x, 2))
            return x
        end
        function $fJ(p::AdjointFTPlan{$elty, FTPlan{$elty, 1, ASSOCIATEDJAC2JAC}}, x::Matrix{$elty})
            checksize(p, x)
            ccall(($(string(fC)), libfasttransforms), Cvoid, (Cint, Cint, Cint, Ptr{ft_plan_struct}, Ptr{$elty}, Cint, Cint), 'T', '1', '2', p, x, size(x, 1), size(x, 2))
            return x
        end
        function $fJ(p::TransposeFTPlan{$elty, FTPlan{$elty, 1, ASSOCIATEDJAC2JAC}}, x::Matrix{$elty})
            checksize(p, x)
            ccall(($(string(fC)), libfasttransforms), Cvoid, (Cint, Cint, Cint, Ptr{ft_plan_struct}, Ptr{$elty}, Cint, Cint), 'T', '1', '2', p, x, size(x, 1), size(x, 2))
            return x
        end
    end
end

for (fJ, fC) in ((:lmul!, :ft_mpfr_trmm_ptr),
                 (:ldiv!, :ft_mpfr_trsm_ptr))
    @eval begin
        function $fJ(p::FTPlan{BigFloat, 1}, x::Matrix{BigFloat})
            checksize(p, x)
            ccall(($(string(fC)), libfasttransforms), Cvoid, (Cint, Cint, Ptr{mpfr_t}, Cint, Ptr{BigFloat}, Cint, Cint, Int32), 'N', p.n, p, p.n, renew!(x), size(x, 1), size(x, 2), Base.MPFR.ROUNDING_MODE[])
            return x
        end
        function $fJ(p::AdjointFTPlan{BigFloat, FTPlan{BigFloat, 1, K}}, x::Matrix{BigFloat}) where K
            checksize(p, x)
            ccall(($(string(fC)), libfasttransforms), Cvoid, (Cint, Cint, Ptr{mpfr_t}, Cint, Ptr{BigFloat}, Cint, Cint, Int32), 'T', p.parent.n, p, p.parent.n, renew!(x), size(x, 1), size(x, 2), Base.MPFR.ROUNDING_MODE[])
            return x
        end
        function $fJ(p::TransposeFTPlan{BigFloat, FTPlan{BigFloat, 1, K}}, x::Matrix{BigFloat}) where K
            checksize(p, x)
            ccall(($(string(fC)), libfasttransforms), Cvoid, (Cint, Cint, Ptr{mpfr_t}, Cint, Ptr{BigFloat}, Cint, Cint, Int32), 'T', p.parent.n, p, p.parent.n, renew!(x), size(x, 1), size(x, 2), Base.MPFR.ROUNDING_MODE[])
            return x
        end
    end
end

for (fJ, fC, T, N, K) in ((:lmul!, :ft_execute_sph2fourier, Float64, 2, SPHERE),
                          (:ldiv!, :ft_execute_fourier2sph, Float64, 2, SPHERE),
                          (:lmul!, :ft_execute_sphv2fourier, Float64, 2, SPHEREV),
                          (:ldiv!, :ft_execute_fourier2sphv, Float64, 2, SPHEREV),
                          (:lmul!, :ft_execute_spinsph2fourier, Complex{Float64}, 2, SPINSPHERE),
                          (:ldiv!, :ft_execute_fourier2spinsph, Complex{Float64}, 2, SPINSPHERE),
                          (:lmul!, :ft_execute_disk2cxf, Float64, 2, DISK),
                          (:ldiv!, :ft_execute_cxf2disk, Float64, 2, DISK),
                          (:lmul!, :ft_execute_rectdisk2cheb, Float64, 2, RECTDISK),
                          (:ldiv!, :ft_execute_cheb2rectdisk, Float64, 2, RECTDISK),
                          (:lmul!, :ft_execute_tri2cheb, Float64, 2, TRIANGLE),
                          (:ldiv!, :ft_execute_cheb2tri, Float64, 2, TRIANGLE))
    @eval begin
        function $fJ(p::FTPlan{$T, $N, $K}, x::Array{$T, $N})
            checksize(p, x)
            ccall(($(string(fC)), libfasttransforms), Cvoid, (Cint, Ptr{ft_plan_struct}, Ptr{$T}, Cint, Cint), 'N', p, x, size(x, 1), size(x, 2))
            return x
        end
        function $fJ(p::AdjointFTPlan{$T, FTPlan{$T, $N, $K}}, x::Array{$T, $N})
            checksize(p, x)
            ccall(($(string(fC)), libfasttransforms), Cvoid, (Cint, Ptr{ft_plan_struct}, Ptr{$T}, Cint, Cint), 'T', p, x, size(x, 1), size(x, 2))
            return x
        end
        function $fJ(p::TransposeFTPlan{$T, FTPlan{$T, $N, $K}}, x::Array{$T, $N})
            checksize(p, x)
            ccall(($(string(fC)), libfasttransforms), Cvoid, (Cint, Ptr{ft_plan_struct}, Ptr{$T}, Cint, Cint), 'T', p, x, size(x, 1), size(x, 2))
            return x
        end
    end
end

for (fJ, fC) in ((:lmul!, :ft_execute_tet2cheb),
                 (:ldiv!, :ft_execute_cheb2tet))
    @eval begin
        function $fJ(p::FTPlan{Float64, 3, TETRAHEDRON}, x::Array{Float64, 3})
            checksize(p, x)
            ccall(($(string(fC)), libfasttransforms), Cvoid, (Cint, Ptr{ft_plan_struct}, Ptr{Float64}, Cint, Cint, Cint), 'N', p, x, size(x, 1), size(x, 2), size(x, 3))
            return x
        end
        function $fJ(p::AdjointFTPlan{Float64, FTPlan{Float64, 3, TETRAHEDRON}}, x::Array{Float64, 3})
            checksize(p, x)
            ccall(($(string(fC)), libfasttransforms), Cvoid, (Cint, Ptr{ft_plan_struct}, Ptr{Float64}, Cint, Cint, Cint), 'T', p, x, size(x, 1), size(x, 2), size(x, 3))
            return x
        end
        function $fJ(p::TransposeFTPlan{Float64, FTPlan{Float64, 3, TETRAHEDRON}}, x::Array{Float64, 3})
            checksize(p, x)
            ccall(($(string(fC)), libfasttransforms), Cvoid, (Cint, Ptr{ft_plan_struct}, Ptr{Float64}, Cint, Cint, Cint), 'T', p, x, size(x, 1), size(x, 2), size(x, 3))
            return x
        end
    end
end

function execute_sph_polar_rotation!(x::Matrix{Float64}, α)
    ccall((:ft_execute_sph_polar_rotation, libfasttransforms), Cvoid, (Ptr{Float64}, Cint, Cint, Float64, Float64), x, size(x, 1), size(x, 2), sin(α), cos(α))
    return x
end

function execute_sph_polar_reflection!(x::Matrix{Float64})
    ccall((:ft_execute_sph_polar_reflection, libfasttransforms), Cvoid, (Ptr{Float64}, Cint, Cint), x, size(x, 1), size(x, 2))
    return x
end

struct ft_orthogonal_transformation
    Q::NTuple{9, Float64}
end

function convert(::Type{ft_orthogonal_transformation}, Q::AbstractMatrix)
    @assert size(Q, 1) ≥ 3 && size(Q, 2) ≥ 3
    return ft_orthogonal_transformation((Q[1, 1], Q[2, 1], Q[3, 1], Q[1, 2], Q[2, 2], Q[3, 2], Q[1, 3], Q[2, 3], Q[3, 3]))
end
convert(::Type{ft_orthogonal_transformation}, Q::NTuple{9, Float64}) = ft_orthogonal_transformation(Q)

function execute_sph_orthogonal_transformation!(p::FTPlan{Float64, 2, SPHERICALISOMETRY}, Q, x::Matrix{Float64})
    checksize(p, x)
    ccall((:ft_execute_sph_orthogonal_transformation, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ft_orthogonal_transformation, Ptr{Float64}, Cint, Cint), p, Q, x, size(x, 1), size(x, 2))
    return x
end

function execute_sph_yz_axis_exchange!(p::FTPlan{Float64, 2, SPHERICALISOMETRY}, x::Matrix{Float64})
    checksize(p, x)
    ccall((:ft_execute_sph_yz_axis_exchange, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, Ptr{Float64}, Cint, Cint), p, x, size(x, 1), size(x, 2))
    return x
end

function execute_sph_rotation!(p::FTPlan{Float64, 2, SPHERICALISOMETRY}, α, β, γ, x::Matrix{Float64})
    checksize(p, x)
    ccall((:ft_execute_sph_rotation, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, Float64, Float64, Float64, Ptr{Float64}, Cint, Cint), p, α, β, γ, x, size(x, 1), size(x, 2))
    return x
end

struct ft_reflection
    w::NTuple{3, Float64}
end

function convert(::Type{ft_reflection}, w::AbstractVector)
    @assert length(w) ≥ 3
    return ft_reflection((w[1], w[2], w[3]))
end
convert(::Type{ft_reflection}, w::NTuple{3, Float64}) = ft_reflection(w)

function execute_sph_reflection!(p::FTPlan{Float64, 2, SPHERICALISOMETRY}, w, x::Matrix{Float64})
    checksize(p, x)
    ccall((:ft_execute_sph_reflection, libfasttransforms), Cvoid, (Ptr{ft_plan_struct}, ft_reflection, Ptr{Float64}, Cint, Cint), p, w, x, size(x, 1), size(x, 2))
    return x
end
execute_sph_reflection!(p::FTPlan{Float64, 2, SPHERICALISOMETRY}, w1, w2, w3, x::Matrix{Float64}) = execute_sph_reflection!(p, ft_reflection(w1, w2, w3), x)

*(p::FTPlan{T}, x::Array{Complex{T}}) where T = lmul!(p, deepcopy(x))
*(p::AdjointFTPlan{T}, x::Array{Complex{T}}) where T = lmul!(p, deepcopy(x))
*(p::TransposeFTPlan{T}, x::Array{Complex{T}}) where T = lmul!(p, deepcopy(x))
\(p::FTPlan{T}, x::Array{Complex{T}}) where T = ldiv!(p, deepcopy(x))
\(p::AdjointFTPlan{T}, x::Array{Complex{T}}) where T = ldiv!(p, deepcopy(x))
\(p::TransposeFTPlan{T}, x::Array{Complex{T}}) where T = ldiv!(p, deepcopy(x))

for fJ in (:lmul!, :ldiv!)
    @eval begin
        function $fJ(p::FTPlan{T}, x::AbstractArray{Complex{T}}) where T
            x .= complex.($fJ(p, real(x)), $fJ(p, imag(x)))
            return x
        end
        function $fJ(p::AdjointFTPlan{T}, x::AbstractArray{Complex{T}}) where T
            x .= complex.($fJ(p, real(x)), $fJ(p, imag(x)))
            return x
        end
        function $fJ(p::TransposeFTPlan{T}, x::AbstractArray{Complex{T}}) where T
            x .= complex.($fJ(p, real(x)), $fJ(p, imag(x)))
            return x
        end
    end
end

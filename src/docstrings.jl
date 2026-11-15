"""
	leg2cheb(v::AbstractVector; normleg::Bool=false, normcheb::Bool=false)

Convert the vector of expansions coefficients `v` from a Legendre to a Chebyshev basis.
The keyword arguments denote whether the bases are normalized.
"""
leg2cheb

"""
	cheb2leg(v::AbstractVector; normcheb::Bool=false, normleg::Bool=false)

Convert the vector of expansions coefficients `v` from a Chebyshev to a Legendre basis.
The keyword arguments denote whether the bases are normalized.
"""
cheb2leg

"""
	ultra2ultra(v::AbstractVector, λ, μ; norm1::Bool=false, norm2::Bool=false)

Convert the vector of expansions coefficients `v` from an Ultraspherical basis of
order `λ` to an Ultraspherical basis of order `μ`.
The keyword arguments denote whether the bases are normalized.
"""
ultra2ultra

"""
	jac2jac(v::AbstractVector, α, β, γ, δ; norm1::Bool=false, norm2::Bool=false)

Convert the vector of expansions coefficients `v` from a Jacobi basis of
order `(α,β)` to a Jacobi basis of order `(γ,δ)`.
The keyword arguments denote whether the bases are normalized.
"""
jac2jac

"""
	lag2lag(v::AbstractVector, α, β; norm1::Bool=false, norm2::Bool=false)

Convert the vector of expansions coefficients `v` from a Laguerre basis of
order `α` to a La basis of order `β`.
The keyword arguments denote whether the bases are normalized."""
lag2lag

"""
	jac2ultra(v::AbstractVector, α, β, λ; normjac::Bool=false, normultra::Bool=false)

Convert the vector of expansions coefficients `v` from a Jacobi basis of
order `(α,β)` to an Ultraspherical basis of order `λ`.
The keyword arguments denote whether the bases are normalized."""
jac2ultra

"""
	ultra2jac(v::AbstractVector, λ, α, β; normultra::Bool=false, normjac::Bool=false)

Convert the vector of expansions coefficients `v` from an Ultraspherical basis of
order `λ` to a Jacobi basis of order `(α,β)`.
The keyword arguments denote whether the bases are normalized.
"""
ultra2jac

"""
	jac2cheb(v::AbstractVector, α, β; normjac::Bool=false, normcheb::Bool=false)

Convert the vector of expansions coefficients `v` from a Jacobi basis of
order `(α,β)` to a Chebyshev basis.
The keyword arguments denote whether the bases are normalized.
"""
jac2cheb

"""
	cheb2jac(v::AbstractVector, α, β; normcheb::Bool=false, normjac::Bool=false)

Convert the vector of expansions coefficients `v` from a Chebyshev basis to a
Jacobi basis of order `(α,β)`.
The keyword arguments denote whether the bases are normalized.
"""
cheb2jac

"""
	ultra2cheb(v::AbstractVector, λ; normultra::Bool=false, normcheb::Bool=false)

Convert the vector of expansions coefficients `v` from an Ultraspherical basis of
order `λ` to a Chebyshev basis.
The keyword arguments denote whether the bases are normalized.
"""
ultra2cheb

"""
	cheb2ultra(v::AbstractVector, λ; normcheb::Bool=false, normultra::Bool=false)

Convert the vector of expansions coefficients `v` from a Chebyshev basis
to an Ultraspherical basis of order `λ`.
The keyword arguments denote whether the bases are normalized.
"""
cheb2ultra

"""
	associatedjac2jac(v::AbstractVector, c::Integer, α, β, γ, δ; norm1::Bool=false, norm2::Bool=false)

Convert the vector of expansions coefficients `v` from an associated Jacobi basis
of orders `(α,β)` to a Jacobi basis of order `(γ,δ)`.
The keyword arguments denote whether the bases are normalized.
"""
associatedjac2jac

"""
	modifiedjac2jac(v::AbstractVector{T}, α, β, u::Vector{T}; verbose::Bool=false) where {T}
	modifiedjac2jac(v::AbstractVector{T}, α, β, u::Vector{T}, v::Vector{T}; verbose::Bool=false) where {T}
"""
modifiedjac2jac

"""
	modifiedlag2lag(v::AbstractVector{T}, α, u::Vector{T}; verbose::Bool=false)
	modifiedlag2lag(v::AbstractVector{T}, α, u::Vector{T}, v::Vector{T}; verbose::Bool=false) where {T}
"""
modifiedlag2lag

"""
	modifiedherm2herm(v::AbstractVector{T}, u::Vector{T}; verbose::Bool=false)
	modifiedherm2herm(v::AbstractVector{T}, u::Vector{T}, v::Vector{T}; verbose::Bool=false) where {T}
"""
modifiedherm2herm

# Harmonic polynomial transforms. The manual is in docs/src/harmonic.md.

"""
	sph2fourier(F::AbstractMatrix)

Convert the `n × (2n-1)` array `F` of real orthonormal spherical harmonic
coefficients to an array of the same size of bivariate Fourier coefficients. An
array with `n` rows stores an expansion of degree at most `n-1`.

The first column of `F` stores the order `m = 0`, and columns `2j` and `2j+1`
store the orders `m = -j` and `m = j`. In the column of order `m`, the
coefficients of degrees `|m|, |m|+1, …, n-1` are stored in rows `1:n-|m|`.

The entries of `F` that do not store a coefficient must be zero. They are not
ignored: the transform reads them as part of the input and overwrites them with
part of the output. Use [`sphzeros`](@ref) to create an array with the right
pattern of zeros.

See also [`fourier2sph`](@ref) for the inverse and [`plan_sph2fourier`](@ref) to
reuse the precomputation.
"""
sph2fourier

"""
	fourier2sph(G::AbstractMatrix)

Convert the `n × (2n-1)` array `G` of bivariate Fourier coefficients to an array
of the same size of real orthonormal spherical harmonic coefficients. This is
the inverse of [`sph2fourier`](@ref), where the arrays and the parameters are
described.
"""
fourier2sph

"""
	plan_sph2fourier(F::AbstractMatrix)
	plan_sph2fourier(Float64, n::Integer)

Precompute the conversion between real orthonormal spherical harmonic
coefficients and bivariate Fourier coefficients, for `n × (2n-1)` arrays. The
size `n` is the number of rows of `F`.

For a plan `P`, `P*F` is the same as [`sph2fourier`](@ref) and `P\\G` is the
same as [`fourier2sph`](@ref). `lmul!(P, F)` and `ldiv!(P, G)` do the same in
place.
"""
plan_sph2fourier

"""
	disk2cxf(F::AbstractMatrix, α = 0, β = 0)

Convert the `n × (4n-3)` array `F` of real orthonormal generalized Zernike
coefficients to an array of the same size of Chebyshev×Fourier coefficients. An
array with `n` rows stores an expansion of degree at most `2n-2`. The
polynomials are orthonormal with respect to the weight `r^(2α+1) (1-r^2)^β dr
dθ` on the unit disk.

The first column of `F` stores the order `m = 0`, and columns `2j` and `2j+1`
store the orders `m = -j` and `m = j`. In the column of order `m`, the
coefficients of degrees `|m|, |m|+2, …` up to `2n-2` are stored in rows
`1:n-⌈|m|/2⌉`.

The entries of `F` that do not store a coefficient must be zero. They are not
ignored: the transform reads them as part of the input and overwrites them with
part of the output. Use [`diskzeros`](@ref) to create an array with the right
pattern of zeros.

See also [`cxf2disk`](@ref) for the inverse and [`plan_disk2cxf`](@ref) to reuse
the precomputation.
"""
disk2cxf

"""
	cxf2disk(G::AbstractMatrix, α = 0, β = 0)

Convert the `n × (4n-3)` array `G` of Chebyshev×Fourier coefficients to an array
of the same size of real orthonormal generalized Zernike coefficients. This is
the inverse of [`disk2cxf`](@ref), where the arrays and the parameters are
described.
"""
cxf2disk

"""
	plan_disk2cxf(F::AbstractMatrix, α = 0, β = 0)
	plan_disk2cxf(Float64, n::Integer, α = 0, β = 0)

Precompute the conversion between real orthonormal generalized Zernike
coefficients and Chebyshev×Fourier coefficients, for `n × (4n-3)` arrays. The
size `n` is the number of rows of `F`.

For a plan `P`, `P*F` is the same as [`disk2cxf`](@ref) and `P\\G` is the same
as [`cxf2disk`](@ref). `lmul!(P, F)` and `ldiv!(P, G)` do the same in place.
"""
plan_disk2cxf

"""
	ann2cxf(F::AbstractMatrix, α, β, γ, ρ)

Convert the `n × (4n-3)` array `F` of real orthonormal annulus polynomial
coefficients to an array of the same size of Chebyshev×Fourier coefficients. An
array with `n` rows stores an expansion of degree at most `2n-2`. The
polynomials are orthonormal with respect to the weight `r^(2γ+1) (r^2-ρ^2)^α
(1-r^2)^β dr dθ` on the annulus `ρ < r < 1`.

The first column of `F` stores the order `m = 0`, and columns `2j` and `2j+1`
store the orders `m = -j` and `m = j`. In the column of order `m`, the
coefficients of degrees `|m|, |m|+2, …` up to `2n-2` are stored in rows
`1:n-⌈|m|/2⌉`, as in [`disk2cxf`](@ref).

The entries of `F` that do not store a coefficient must be zero. They are not
ignored: the transform reads them as part of the input and overwrites them with
part of the output. Use [`diskzeros`](@ref) to create an array with the right
pattern of zeros.

See also [`cxf2ann`](@ref) for the inverse and [`plan_ann2cxf`](@ref) to reuse
the precomputation.
"""
ann2cxf

"""
	cxf2ann(G::AbstractMatrix, α, β, γ, ρ)

Convert the `n × (4n-3)` array `G` of Chebyshev×Fourier coefficients to an array
of the same size of real orthonormal annulus polynomial coefficients. This is
the inverse of [`ann2cxf`](@ref), where the arrays and the parameters are
described.
"""
cxf2ann

"""
	plan_ann2cxf(F::AbstractMatrix, α, β, γ, ρ)
	plan_ann2cxf(Float64, n::Integer, α, β, γ, ρ)

Precompute the conversion between real orthonormal annulus polynomial
coefficients and Chebyshev×Fourier coefficients, for `n × (4n-3)` arrays. The
size `n` is the number of rows of `F`.

For a plan `P`, `P*F` is the same as [`ann2cxf`](@ref) and `P\\G` is the same as
[`cxf2ann`](@ref). `lmul!(P, F)` and `ldiv!(P, G)` do the same in place.
"""
plan_ann2cxf

"""
	rectdisk2cheb(F::AbstractMatrix, β = 0)

Convert the `n × n` array `F` of real orthonormal Dunkl–Xu coefficients to an
array of the same size of partially weighted bivariate Chebyshev coefficients.
An array with `n` rows stores an expansion of degree at most `n-1`. The
polynomials are orthonormal with respect to the weight `(1-x^2-y^2)^β` on the
unit disk.

Column `m+1` of `F` stores the order `m`: the coefficients of degrees `m, m+1,
…, n-1` are stored in rows `1:n-m`.

The entries of `F` that do not store a coefficient must be zero. They are not
ignored: the transform reads them as part of the input and overwrites them with
part of the output. Use [`trizeros`](@ref) to create an array with the right
pattern of zeros.

See also [`cheb2rectdisk`](@ref) for the inverse and
[`plan_rectdisk2cheb`](@ref) to reuse the precomputation.
"""
rectdisk2cheb

"""
	cheb2rectdisk(G::AbstractMatrix, β = 0)

Convert the `n × n` array `G` of partially weighted bivariate Chebyshev
coefficients to an array of the same size of real orthonormal Dunkl–Xu
coefficients. This is the inverse of [`rectdisk2cheb`](@ref), where the arrays
and the parameters are described.
"""
cheb2rectdisk

"""
	plan_rectdisk2cheb(F::AbstractMatrix, β = 0)
	plan_rectdisk2cheb(Float64, n::Integer, β = 0)

Precompute the conversion between real orthonormal Dunkl–Xu coefficients and
partially weighted bivariate Chebyshev coefficients, for `n × n` arrays. The
size `n` is the number of rows of `F`.

For a plan `P`, `P*F` is the same as [`rectdisk2cheb`](@ref) and `P\\G` is the
same as [`cheb2rectdisk`](@ref). `lmul!(P, F)` and `ldiv!(P, G)` do the same in
place.
"""
plan_rectdisk2cheb

"""
	tri2cheb(F::AbstractMatrix, α = 0, β = 0, γ = 0)

Convert the `n × n` array `F` of Proriol polynomial coefficients to an array of
the same size of bivariate Chebyshev coefficients. An array with `n` rows stores
an expansion of degree at most `n-1`. The polynomials are orthogonal with
respect to the weight `x^α y^β (1-x-y)^γ` on the triangle with vertices `(0,0)`,
`(1,0)`, and `(0,1)`.

Column `m+1` of `F` stores the order `m`: the coefficients of degrees `m, m+1,
…, n-1` are stored in rows `1:n-m`.

The entries of `F` that do not store a coefficient must be zero. They are not
ignored: the transform reads them as part of the input and overwrites them with
part of the output. Use [`trizeros`](@ref) to create an array with the right
pattern of zeros.

See also [`cheb2tri`](@ref) for the inverse and [`plan_tri2cheb`](@ref) to reuse
the precomputation.
"""
tri2cheb

"""
	cheb2tri(G::AbstractMatrix, α = 0, β = 0, γ = 0)

Convert the `n × n` array `G` of bivariate Chebyshev coefficients to an array of
the same size of Proriol polynomial coefficients. This is the inverse of
[`tri2cheb`](@ref), where the arrays and the parameters are described.
"""
cheb2tri

"""
	plan_tri2cheb(F::AbstractMatrix, α = 0, β = 0, γ = 0)
	plan_tri2cheb(Float64, n::Integer, α = 0, β = 0, γ = 0)

Precompute the conversion between Proriol polynomial coefficients and bivariate
Chebyshev coefficients, for `n × n` arrays. The size `n` is the number of rows
of `F`.

For a plan `P`, `P*F` is the same as [`tri2cheb`](@ref) and `P\\G` is the same
as [`cheb2tri`](@ref). `lmul!(P, F)` and `ldiv!(P, G)` do the same in place.
"""
plan_tri2cheb

"""
	tet2cheb(F::AbstractArray{<:Any,3}, α = 0, β = 0, γ = 0, δ = 0)

Convert the `n × n × n` array `F` of Proriol polynomial coefficients to an array
of the same size of trivariate Chebyshev coefficients. An array with `n` rows
stores an expansion of degree at most `n-1`. The polynomials are orthogonal with
respect to the weight `x^α y^β z^γ (1-x-y-z)^δ` on the tetrahedron with vertices
`(0,0,0)`, `(1,0,0)`, `(0,1,0)`, and `(0,0,1)`.

The entries `F[i, ℓ+1, m+1]` with `i = 1:n-ℓ-m` store the coefficients of
degrees `ℓ+m, ℓ+m+1, …, n-1`.

The entries of `F` that do not store a coefficient must be zero. They are not
ignored: the transform reads them as part of the input and overwrites them with
part of the output. Use [`tetzeros`](@ref) to create an array with the right
pattern of zeros.

See also [`cheb2tet`](@ref) for the inverse and [`plan_tet2cheb`](@ref) to reuse
the precomputation.
"""
tet2cheb

"""
	cheb2tet(G::AbstractArray{<:Any,3}, α = 0, β = 0, γ = 0, δ = 0)

Convert the `n × n × n` array `G` of trivariate Chebyshev coefficients to an
array of the same size of Proriol polynomial coefficients. This is the inverse
of [`tet2cheb`](@ref), where the arrays and the parameters are described.
"""
cheb2tet

"""
	plan_tet2cheb(F::AbstractArray{<:Any,3}, α = 0, β = 0, γ = 0, δ = 0)
	plan_tet2cheb(Float64, n::Integer, α = 0, β = 0, γ = 0, δ = 0)

Precompute the conversion between Proriol polynomial coefficients and trivariate
Chebyshev coefficients, for `n × n × n` arrays. The size `n` is the number of
rows of `F`.

For a plan `P`, `P*F` is the same as [`tet2cheb`](@ref) and `P\\G` is the same
as [`cheb2tet`](@ref). `lmul!(P, F)` and `ldiv!(P, G)` do the same in place.
"""
plan_tet2cheb

"""
	plan_spinsph2fourier(F::AbstractMatrix{<:Complex}, s::Integer)
	plan_spinsph2fourier(ComplexF64, n::Integer, s::Integer)

Precompute the conversion between spin-weighted spherical harmonic coefficients
with spin `s` and bivariate Fourier coefficients, for complex `n × (2n-1)`
arrays. The size `n` is the number of rows of `F`, and an array with `n` rows
stores an expansion of degree at most `n-1`.

For a plan `P`, `P*F` is the forward transform and `P\\G` is its inverse.
`lmul!(P, F)` and `ldiv!(P, G)` do the same in place.

The first column of `F` stores the order `m = 0`, and columns `2j` and `2j+1`
store the orders `m = -j` and `m = j`. In the column of order `m`, the
coefficients of degrees `ℓ₀, ℓ₀+1, …, n-1` are stored in rows `1:n-ℓ₀`, where
`ℓ₀ = max(|m|, |s|)`.

The entries of `F` that do not store a coefficient must be zero. They are not
ignored: the transform reads them as part of the input and overwrites them with
part of the output. Use [`spinsphzeros`](@ref) to create an array with the right
pattern of zeros.
"""
plan_spinsph2fourier

# Arrays of coefficients with the right pattern of zeros. Each family shares a docstring.

for f in (:sphzeros, :sphones, :sphrand, :sphrandn)
    @eval @doc """
        sphzeros(T, n, 2n-1)
        sphones(T, n, 2n-1)
        sphrand(T, n, 2n-1)
        sphrandn(T, n, 2n-1)

    Return an `n × (2n-1)` array of element type `T` of spherical harmonic
    coefficients for [`sph2fourier`](@ref). Every entry that stores a
    coefficient is zero, one, uniformly random, or normally random,
    respectively, and every other entry is zero.
    """ $f
end

for f in (:diskzeros, :diskones, :diskrand, :diskrandn)
    @eval @doc """
        diskzeros(T, n, 4n-3)
        diskones(T, n, 4n-3)
        diskrand(T, n, 4n-3)
        diskrandn(T, n, 4n-3)

    Return an `n × (4n-3)` array of element type `T` of Zernike coefficients for
    [`disk2cxf`](@ref), or of annulus polynomial coefficients for
    [`ann2cxf`](@ref). Every entry that stores a coefficient is zero, one,
    uniformly random, or normally random, respectively, and every other entry is
    zero.
    """ $f
end

for f in (:trizeros, :triones, :trirand, :trirandn)
    @eval @doc """
        trizeros(T, n, n)
        triones(T, n, n)
        trirand(T, n, n)
        trirandn(T, n, n)

    Return an `n × n` array of element type `T` of Proriol coefficients for
    [`tri2cheb`](@ref). The functions `rectdiskzeros`, `rectdiskones`,
    `rectdiskrand`, and `rectdiskrandn` are the same, for the Dunkl–Xu
    coefficients of [`rectdisk2cheb`](@ref). Every entry that stores a
    coefficient is zero, one, uniformly random, or normally random,
    respectively, and every other entry is zero.
    """ $f
end

for f in (:tetzeros, :tetones, :tetrand, :tetrandn)
    @eval @doc """
        tetzeros(T, n, n, n)
        tetones(T, n, n, n)
        tetrand(T, n, n, n)
        tetrandn(T, n, n, n)

    Return an `n × n × n` array of element type `T` of Proriol coefficients for
    [`tet2cheb`](@ref). Every entry that stores a coefficient is zero, one,
    uniformly random, or normally random, respectively, and every other entry is
    zero.
    """ $f
end

for f in (:spinsphzeros, :spinsphones, :spinsphrand, :spinsphrandn)
    @eval @doc """
        spinsphzeros(T, n, 2n-1)
        spinsphones(T, n, 2n-1, s)
        spinsphrand(T, n, 2n-1, s)
        spinsphrandn(T, n, 2n-1, s)

    Return an `n × (2n-1)` array of element type `T` of spin-weighted spherical
    harmonic coefficients with spin `s` for [`plan_spinsph2fourier`](@ref).
    Every entry that stores a coefficient is zero, one, uniformly random, or
    normally random, respectively, and every other entry is zero.
    """ $f
end

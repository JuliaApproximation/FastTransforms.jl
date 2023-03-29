"""
	leg2cheb(v::AbstractVector; normleg::Bool=false, normcheb::Bool=false)

Convert the vector of expansions coefficients `v` from a Legendre to a Chebyshev basis.
The keyword arguments denote whether the bases are normalized.
"""
leg2cheb

"""
	cheb2leg(v::StridedVector; normcheb::Bool=false, normleg::Bool=false)

Convert the vector of expansions coefficients `v` from a Chebyshev to a Legendre basis.
The keyword arguments denote whether the bases are normalized.
"""
cheb2leg

"""
	ultra2ultra(v::StridedVector, λ, μ; norm1::Bool=false, norm2::Bool=false)

Convert the vector of expansions coefficients `v` from an Ultraspherical basis of
order `λ` to an Ultraspherical basis of order `μ`.
The keyword arguments denote whether the bases are normalized.
"""
ultra2ultra

"""
	jac2jac(v::StridedVector, α, β, γ, δ; norm1::Bool=false, norm2::Bool=false)

Convert the vector of expansions coefficients `v` from a Jacobi basis of
order `(α,β)` to a Jacobi basis of order `(γ,δ)`.
The keyword arguments denote whether the bases are normalized.
"""
jac2jac

"""
	lag2lag(v::StridedVector, α, β; norm1::Bool=false, norm2::Bool=false)

Convert the vector of expansions coefficients `v` from a Laguerre basis of
order `α` to a La basis of order `β`.
The keyword arguments denote whether the bases are normalized."""
lag2lag

"""
	jac2ultra(v::StridedVector, α, β, λ; normjac::Bool=false, normultra::Bool=false)

Convert the vector of expansions coefficients `v` from a Jacobi basis of
order `(α,β)` to an Ultraspherical basis of order `λ`.
The keyword arguments denote whether the bases are normalized."""
jac2ultra

"""
	ultra2jac(v::StridedVector, λ, α, β; normultra::Bool=false, normjac::Bool=false)

Convert the vector of expansions coefficients `v` from an Ultraspherical basis of
order `λ` to a Jacobi basis of order `(α,β)`.
The keyword arguments denote whether the bases are normalized.
"""
ultra2jac

"""
	jac2cheb(v::StridedVector, α, β; normjac::Bool=false, normcheb::Bool=false)

Convert the vector of expansions coefficients `v` from a Jacobi basis of
order `(α,β)` to a Chebyshev basis.
The keyword arguments denote whether the bases are normalized.
"""
jac2cheb

"""
	cheb2jac(v::StridedVector, α, β; normcheb::Bool=false, normjac::Bool=false)

Convert the vector of expansions coefficients `v` from a Chebyshev basis to a
Jacobi basis of order `(α,β)`.
The keyword arguments denote whether the bases are normalized.
"""
cheb2jac

"""
	ultra2cheb(v::StridedVector, λ; normultra::Bool=false, normcheb::Bool=false)

Convert the vector of expansions coefficients `v` from an Ultraspherical basis of
order `λ` to a Chebyshev basis.
The keyword arguments denote whether the bases are normalized.
"""
ultra2cheb

"""
	cheb2ultra(v::StridedVector, λ; normcheb::Bool=false, normultra::Bool=false)

Convert the vector of expansions coefficients `v` from a Chebyshev basis
to an Ultraspherical basis of order `λ`.
The keyword arguments denote whether the bases are normalized.
"""
cheb2ultra

"""
	associatedjac2jac(v::StridedVector, c::Integer, α, β, γ, δ; norm1::Bool=false, norm2::Bool=false)

Convert the vector of expansions coefficients `v` from an associated Jacobi basis
of orders `(α,β)` to a Jacobi basis of order `(γ,δ)`.
The keyword arguments denote whether the bases are normalized.
"""
associatedjac2jac

"""
	modifiedjac2jac(v::StridedVector{T}, α, β, u::Vector{T}; verbose::Bool=false) where {T}
	modifiedjac2jac(v::StridedVector{T}, α, β, u::Vector{T}, v::Vector{T}; verbose::Bool=false) where {T}
"""
modifiedjac2jac

"""
	modifiedlag2lag(v::StridedVector{T}, α, u::Vector{T}; verbose::Bool=false)
	modifiedlag2lag(v::StridedVector{T}, α, u::Vector{T}, v::Vector{T}; verbose::Bool=false) where {T}
"""
modifiedlag2lag

"""
	modifiedherm2herm(v::StridedVector{T}, u::Vector{T}; verbose::Bool=false)
	modifiedherm2herm(v::StridedVector{T}, u::Vector{T}, v::Vector{T}; verbose::Bool=false) where {T}
"""
modifiedherm2herm

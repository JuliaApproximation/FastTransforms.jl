# FastTransforms

## Exported

---

<a id="method__cjt.1" class="lexicon_definition"></a>
#### cjt(c,  α,  β) [¶](#method__cjt.1)
Computes the Chebyshev expansion coefficients
given the Jacobi expansion coefficients ``c`` with parameters ``α`` and ``β``.

See also [`icjt`](#method__icjt.1) and [`jjt`](#method__jjt.1).


*source:*
[FastTransforms/src/cjt.jl:121](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/9a0fff4d389746d63718883170537e95ad849e76/src/cjt.jl#L121)

---

<a id="method__gaunt.1" class="lexicon_definition"></a>
#### gaunt(m::Int64,  n::Int64,  μ::Int64,  ν::Int64) [¶](#method__gaunt.1)
Calculates the Gaunt coefficients in 64-bit floating-point arithmetic.


*source:*
[FastTransforms/src/gaunt.jl:24](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/9a0fff4d389746d63718883170537e95ad849e76/src/gaunt.jl#L24)

---

<a id="method__gaunt.2" class="lexicon_definition"></a>
#### gaunt{T}(::Type{T},  m::Int64,  n::Int64,  μ::Int64,  ν::Int64) [¶](#method__gaunt.2)
Calculates the Gaunt coefficients, defined by:

    a(m,n,μ,ν,q) = (2(n+ν-2q)+1)/2 (n+ν-2q-m-μ)!/(n+ν-2q+m+μ)! ∫₋₁⁺¹ P_m^n(x) P_ν^μ(x) P_{n+ν-2q}^{m+μ}(x) dx.

or defined by:

    P_n^m(x) P_ν^μ(x) = ∑_{q=0}^{q_{max}} a(m,n,μ,ν,q) P_{n+ν-2q}^{m+μ}(x)

This is a Julia implementation of the stable recurrence described in:

    Y.-l. Xu, "Fast evaluation of Gaunt coefficients: recursive approach", J. Comp. Appl. Math., 85:53–65, 1997.


*source:*
[FastTransforms/src/gaunt.jl:14](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/9a0fff4d389746d63718883170537e95ad849e76/src/gaunt.jl#L14)

---

<a id="method__icjt.1" class="lexicon_definition"></a>
#### icjt(c,  α,  β) [¶](#method__icjt.1)
Computes the Jacobi expansion coefficients with parameters ``α`` and ``β``
given the Chebyshev expansion coefficients ``c``.

See also [`cjt`](#method__cjt.1) and [`jjt`](#method__jjt.1).


*source:*
[FastTransforms/src/cjt.jl:129](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/9a0fff4d389746d63718883170537e95ad849e76/src/cjt.jl#L129)

---

<a id="method__jjt.1" class="lexicon_definition"></a>
#### jjt(c,  α,  β,  γ,  δ) [¶](#method__jjt.1)
Computes the Jacobi expansion coefficients with parameters ``γ`` and ``δ``
given the Jacobi expansion coefficients ``c`` with parameters ``α`` and ``β``.

See also [`cjt`](#method__cjt.1) and [`icjt`](#method__icjt.1).


*source:*
[FastTransforms/src/cjt.jl:137](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/9a0fff4d389746d63718883170537e95ad849e76/src/cjt.jl#L137)

---

<a id="method__plan_cjt.1" class="lexicon_definition"></a>
#### plan_cjt(c::AbstractArray{T, 1},  α,  β) [¶](#method__plan_cjt.1)
Pre-plan optimized DCT-I and DST-I plans and pre-allocate the necessary
arrays, normalization constants, and recurrence coefficients for a forward Chebyshev—Jacobi transform.

``c`` is the vector of coefficients; and,

``α`` and ``β`` are the Jacobi parameters.

Optionally:

``M`` determines the number of terms in Hahn's asymptotic expansion.


*source:*
[FastTransforms/src/cjt.jl:158](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/9a0fff4d389746d63718883170537e95ad849e76/src/cjt.jl#L158)

---

<a id="method__plan_icjt.1" class="lexicon_definition"></a>
#### plan_icjt(c::AbstractArray{T, 1},  α,  β) [¶](#method__plan_icjt.1)
Pre-plan optimized DCT-I and DST-I plans and pre-allocate the necessary
arrays, normalization constants, and recurrence coefficients for an inverse Chebyshev—Jacobi transform.

``c`` is the vector of coefficients; and,

``α`` and ``β`` are the Jacobi parameters.

Optionally:

``M`` determines the number of terms in Hahn's asymptotic expansion.


*source:*
[FastTransforms/src/cjt.jl:172](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/9a0fff4d389746d63718883170537e95ad849e76/src/cjt.jl#L172)

## Internal

---

<a id="method__chebyshevjacobimoments1.1" class="lexicon_definition"></a>
#### chebyshevjacobimoments1{T<:AbstractFloat}(N::Int64,  α::T<:AbstractFloat,  β::T<:AbstractFloat) [¶](#method__chebyshevjacobimoments1.1)
Modified Chebyshev moments of the first kind with respect to the Jacobi weight:

    ∫₋₁⁺¹ T_n(x) (1-x)^α(1+x)^β dx.



*source:*
[FastTransforms/src/specialfunctions.jl:399](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/9a0fff4d389746d63718883170537e95ad849e76/src/specialfunctions.jl#L399)

---

<a id="method__chebyshevjacobimoments2.1" class="lexicon_definition"></a>
#### chebyshevjacobimoments2{T<:AbstractFloat}(N::Int64,  α::T<:AbstractFloat,  β::T<:AbstractFloat) [¶](#method__chebyshevjacobimoments2.1)
Modified Chebyshev moments of the second kind with respect to the Jacobi weight:

    ∫₋₁⁺¹ U_n(x) (1-x)^α(1+x)^β dx.



*source:*
[FastTransforms/src/specialfunctions.jl:417](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/9a0fff4d389746d63718883170537e95ad849e76/src/specialfunctions.jl#L417)

---

<a id="method__clenshawcurtisweights.1" class="lexicon_definition"></a>
#### clenshawcurtisweights{T<:AbstractFloat}(N::Int64,  α::T<:AbstractFloat,  β::T<:AbstractFloat) [¶](#method__clenshawcurtisweights.1)
Compute weights of the Clenshaw—Curtis quadrature rule with a Jacobi weight.


*source:*
[FastTransforms/src/clenshawcurtis.jl:12](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/9a0fff4d389746d63718883170537e95ad849e76/src/clenshawcurtis.jl#L12)

---

<a id="method__clenshawcurtis.1" class="lexicon_definition"></a>
#### clenshawcurtis{T<:AbstractFloat}(N::Int64,  α::T<:AbstractFloat,  β::T<:AbstractFloat) [¶](#method__clenshawcurtis.1)
Compute nodes and weights of the Clenshaw—Curtis quadrature rule with a Jacobi weight.


*source:*
[FastTransforms/src/clenshawcurtis.jl:6](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/9a0fff4d389746d63718883170537e95ad849e76/src/clenshawcurtis.jl#L6)

---

<a id="method__decrement945.1" class="lexicon_definition"></a>
#### decrementα!(c::AbstractArray{T, 1},  α,  β) [¶](#method__decrement945.1)
Compute Jacobi expansion coefficients in Pₙ^(α-1,β) given Jacobi expansion coefficients in Pₙ^(α,β) in-place.


*source:*
[FastTransforms/src/specialfunctions.jl:467](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/9a0fff4d389746d63718883170537e95ad849e76/src/specialfunctions.jl#L467)

---

<a id="method__decrement945946.1" class="lexicon_definition"></a>
#### decrementαβ!(c::AbstractArray{T, 1},  α,  β) [¶](#method__decrement945946.1)
Compute Jacobi expansion coefficients in Pₙ^(α-1,α-1) given Jacobi expansion coefficients in Pₙ^(α,α) in-place.


*source:*
[FastTransforms/src/specialfunctions.jl:489](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/9a0fff4d389746d63718883170537e95ad849e76/src/specialfunctions.jl#L489)

---

<a id="method__decrement946.1" class="lexicon_definition"></a>
#### decrementβ!(c::AbstractArray{T, 1},  α,  β) [¶](#method__decrement946.1)
Compute Jacobi expansion coefficients in Pₙ^(α,β-1) given Jacobi expansion coefficients in Pₙ^(α,β) in-place.


*source:*
[FastTransforms/src/specialfunctions.jl:478](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/9a0fff4d389746d63718883170537e95ad849e76/src/specialfunctions.jl#L478)

---

<a id="method__fejer1.1" class="lexicon_definition"></a>
#### fejer1{T<:AbstractFloat}(N::Int64,  α::T<:AbstractFloat,  β::T<:AbstractFloat) [¶](#method__fejer1.1)
Compute nodes and weights of Fejer's first quadrature rule with a Jacobi weight.


*source:*
[FastTransforms/src/fejer.jl:7](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/9a0fff4d389746d63718883170537e95ad849e76/src/fejer.jl#L7)

---

<a id="method__fejer2.1" class="lexicon_definition"></a>
#### fejer2{T<:AbstractFloat}(N::Int64,  α::T<:AbstractFloat,  β::T<:AbstractFloat) [¶](#method__fejer2.1)
Compute nodes and weights of Fejer's second quadrature rule with a Jacobi weight.


*source:*
[FastTransforms/src/fejer.jl:12](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/9a0fff4d389746d63718883170537e95ad849e76/src/fejer.jl#L12)

---

<a id="method__fejerweights1.1" class="lexicon_definition"></a>
#### fejerweights1{T<:AbstractFloat}(N::Int64,  α::T<:AbstractFloat,  β::T<:AbstractFloat) [¶](#method__fejerweights1.1)
Compute weights of Fejer's first quadrature rule with a Jacobi weight.


*source:*
[FastTransforms/src/fejer.jl:21](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/9a0fff4d389746d63718883170537e95ad849e76/src/fejer.jl#L21)

---

<a id="method__fejerweights2.1" class="lexicon_definition"></a>
#### fejerweights2{T<:AbstractFloat}(N::Int64,  α::T<:AbstractFloat,  β::T<:AbstractFloat) [¶](#method__fejerweights2.1)
Compute weights of Fejer's second quadrature rule with a Jacobi weight.


*source:*
[FastTransforms/src/fejer.jl:26](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/9a0fff4d389746d63718883170537e95ad849e76/src/fejer.jl#L26)

---

<a id="method__half.1" class="lexicon_definition"></a>
#### half(x::Number) [¶](#method__half.1)
Compute a typed 0.5.


*source:*
[FastTransforms/src/specialfunctions.jl:12](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/9a0fff4d389746d63718883170537e95ad849e76/src/specialfunctions.jl#L12)

---

<a id="method__increment945.1" class="lexicon_definition"></a>
#### incrementα!(c::AbstractArray{T, 1},  α,  β) [¶](#method__increment945.1)
Compute Jacobi expansion coefficients in Pₙ^(α+1,β) given Jacobi expansion coefficients in Pₙ^(α,β) in-place.


*source:*
[FastTransforms/src/specialfunctions.jl:432](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/9a0fff4d389746d63718883170537e95ad849e76/src/specialfunctions.jl#L432)

---

<a id="method__increment945946.1" class="lexicon_definition"></a>
#### incrementαβ!(c::AbstractArray{T, 1},  α,  β) [¶](#method__increment945946.1)
Compute Jacobi expansion coefficients in Pₙ^(α+1,α+1) given Jacobi expansion coefficients in Pₙ^(α,α) in-place.


*source:*
[FastTransforms/src/specialfunctions.jl:454](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/9a0fff4d389746d63718883170537e95ad849e76/src/specialfunctions.jl#L454)

---

<a id="method__increment946.1" class="lexicon_definition"></a>
#### incrementβ!(c::AbstractArray{T, 1},  α,  β) [¶](#method__increment946.1)
Compute Jacobi expansion coefficients in Pₙ^(α,β+1) given Jacobi expansion coefficients in Pₙ^(α,β) in-place.


*source:*
[FastTransforms/src/specialfunctions.jl:443](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/9a0fff4d389746d63718883170537e95ad849e76/src/specialfunctions.jl#L443)

---

<a id="method__pochhammer.1" class="lexicon_definition"></a>
#### pochhammer(x::Number,  n::Integer) [¶](#method__pochhammer.1)
Pochhammer symbol (x)_n = Γ(x+n)/Γ(x) for the rising factorial.


*source:*
[FastTransforms/src/specialfunctions.jl:32](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/9a0fff4d389746d63718883170537e95ad849e76/src/specialfunctions.jl#L32)

---

<a id="method__stirlingseries.1" class="lexicon_definition"></a>
#### stirlingseries(z) [¶](#method__stirlingseries.1)
Stirling series for Γ(z).


*source:*
[FastTransforms/src/specialfunctions.jl:63](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/9a0fff4d389746d63718883170537e95ad849e76/src/specialfunctions.jl#L63)

---

<a id="method__two.1" class="lexicon_definition"></a>
#### two(x::Number) [¶](#method__two.1)
Compute a typed 2.


*source:*
[FastTransforms/src/specialfunctions.jl:20](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/9a0fff4d389746d63718883170537e95ad849e76/src/specialfunctions.jl#L20)

---

<a id="method__923.1" class="lexicon_definition"></a>
#### Λ(x::Float64) [¶](#method__923.1)
For 64-bit floating-point arithmetic, the Lambda function uses the asymptotic series for τ in Appendix B of

    I. Bogaert and B. Michiels and J. Fostier, 𝒪(1) computation of Legendre polynomials and Gauss–Legendre nodes and weights for parallel computing, SIAM J. Sci. Comput., 34:C83–C101, 2012.


*source:*
[FastTransforms/src/specialfunctions.jl:147](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/9a0fff4d389746d63718883170537e95ad849e76/src/specialfunctions.jl#L147)

---

<a id="method__923.2" class="lexicon_definition"></a>
#### Λ(z::Number) [¶](#method__923.2)
The Lambda function Λ(z) = Γ(z+½)/Γ(z+1) for the ratio of gamma functions.


*source:*
[FastTransforms/src/specialfunctions.jl:141](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/9a0fff4d389746d63718883170537e95ad849e76/src/specialfunctions.jl#L141)

---

<a id="method__923.3" class="lexicon_definition"></a>
#### Λ(z::Number,  λ₁::Number,  λ₂::Number) [¶](#method__923.3)
The Lambda function Λ(z,λ₁,λ₂) = Γ(z+λ₁)/Γ(z+λ₂) for the ratio of gamma functions.


*source:*
[FastTransforms/src/specialfunctions.jl:160](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/9a0fff4d389746d63718883170537e95ad849e76/src/specialfunctions.jl#L160)

---

<a id="method__948.1" class="lexicon_definition"></a>
#### δ(k::Integer,  j::Integer) [¶](#method__948.1)
The Kronecker δ function.


*source:*
[FastTransforms/src/specialfunctions.jl:26](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/9a0fff4d389746d63718883170537e95ad849e76/src/specialfunctions.jl#L26)

---

<a id="type__toeplitzhankelplan.1" class="lexicon_definition"></a>
#### FastTransforms.ToeplitzHankelPlan{S} [¶](#type__toeplitzhankelplan.1)
Store a diagonally-scaled Toeplitz∘Hankel matrix:

    DL(T∘H)DR

where the Hankel matrix `H` is non-negative definite. This allows a Cholesky decomposition in 𝒪(K²N) operations and 𝒪(KN) storage, K = log N log ɛ⁻¹.


*source:*
[FastTransforms/src/toeplitzhankel.jl:8](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/9a0fff4d389746d63718883170537e95ad849e76/src/toeplitzhankel.jl#L8)


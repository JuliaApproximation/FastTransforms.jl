# FastTransforms

## Exported

---

<a id="method__cjt.1" class="lexicon_definition"></a>
#### cjt(c,  α,  β) [¶](#method__cjt.1)
Computes the Chebyshev expansion coefficients
given the Jacobi expansion coefficients ``c`` with parameters ``α`` and ``β``.

See also [`icjt`](#method__icjt.1) and [`jjt`](#method__jjt.1).


*source:*
[FastTransforms/src/cjt.jl:127](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/b1dce339ed19119766d44b5ec44d954932da58d3/src/cjt.jl#L127)

---

<a id="method__gaunt.1" class="lexicon_definition"></a>
#### gaunt(m::Int64,  n::Int64,  μ::Int64,  ν::Int64) [¶](#method__gaunt.1)
Calculates the Gaunt coefficients in 64-bit floating-point arithmetic.


*source:*
[FastTransforms/src/gaunt.jl:24](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/b1dce339ed19119766d44b5ec44d954932da58d3/src/gaunt.jl#L24)

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
[FastTransforms/src/gaunt.jl:14](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/b1dce339ed19119766d44b5ec44d954932da58d3/src/gaunt.jl#L14)

---

<a id="method__icjt.1" class="lexicon_definition"></a>
#### icjt(c,  α,  β) [¶](#method__icjt.1)
Computes the Jacobi expansion coefficients with parameters ``α`` and ``β``
given the Chebyshev expansion coefficients ``c``.

See also [`cjt`](#method__cjt.1) and [`jjt`](#method__jjt.1).


*source:*
[FastTransforms/src/cjt.jl:135](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/b1dce339ed19119766d44b5ec44d954932da58d3/src/cjt.jl#L135)

---

<a id="method__jjt.1" class="lexicon_definition"></a>
#### jjt(c,  α,  β,  γ,  δ) [¶](#method__jjt.1)
Computes the Jacobi expansion coefficients with parameters ``γ`` and ``δ``
given the Jacobi expansion coefficients ``c`` with parameters ``α`` and ``β``.

See also [`cjt`](#method__cjt.1) and [`icjt`](#method__icjt.1).


*source:*
[FastTransforms/src/cjt.jl:143](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/b1dce339ed19119766d44b5ec44d954932da58d3/src/cjt.jl#L143)

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
[FastTransforms/src/cjt.jl:157](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/b1dce339ed19119766d44b5ec44d954932da58d3/src/cjt.jl#L157)

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
[FastTransforms/src/cjt.jl:176](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/b1dce339ed19119766d44b5ec44d954932da58d3/src/cjt.jl#L176)

## Internal

---

<a id="method__chebyshevjacobimoments1.1" class="lexicon_definition"></a>
#### chebyshevjacobimoments1{T<:AbstractFloat}(N::Int64,  α::T<:AbstractFloat,  β::T<:AbstractFloat) [¶](#method__chebyshevjacobimoments1.1)
Modified Chebyshev moments of the first kind with respect to the Jacobi weight:

    ∫₋₁⁺¹ T_n(x) (1-x)^α(1+x)^β dx.



*source:*
[FastTransforms/src/specialfunctions.jl:362](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/b1dce339ed19119766d44b5ec44d954932da58d3/src/specialfunctions.jl#L362)

---

<a id="method__chebyshevjacobimoments2.1" class="lexicon_definition"></a>
#### chebyshevjacobimoments2{T<:AbstractFloat}(N::Int64,  α::T<:AbstractFloat,  β::T<:AbstractFloat) [¶](#method__chebyshevjacobimoments2.1)
Modified Chebyshev moments of the second kind with respect to the Jacobi weight:

    ∫₋₁⁺¹ U_n(x) (1-x)^α(1+x)^β dx.



*source:*
[FastTransforms/src/specialfunctions.jl:380](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/b1dce339ed19119766d44b5ec44d954932da58d3/src/specialfunctions.jl#L380)

---

<a id="method__clenshawcurtisweights.1" class="lexicon_definition"></a>
#### clenshawcurtisweights{T<:AbstractFloat}(N::Int64,  α::T<:AbstractFloat,  β::T<:AbstractFloat) [¶](#method__clenshawcurtisweights.1)
Compute weights of the Clenshaw—Curtis quadrature rule with a Jacobi weight.


*source:*
[FastTransforms/src/clenshawcurtis.jl:12](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/b1dce339ed19119766d44b5ec44d954932da58d3/src/clenshawcurtis.jl#L12)

---

<a id="method__clenshawcurtis.1" class="lexicon_definition"></a>
#### clenshawcurtis{T<:AbstractFloat}(N::Int64,  α::T<:AbstractFloat,  β::T<:AbstractFloat) [¶](#method__clenshawcurtis.1)
Compute nodes and weights of the Clenshaw—Curtis quadrature rule with a Jacobi weight.


*source:*
[FastTransforms/src/clenshawcurtis.jl:6](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/b1dce339ed19119766d44b5ec44d954932da58d3/src/clenshawcurtis.jl#L6)

---

<a id="method__fejer1.1" class="lexicon_definition"></a>
#### fejer1{T<:AbstractFloat}(N::Int64,  α::T<:AbstractFloat,  β::T<:AbstractFloat) [¶](#method__fejer1.1)
Compute nodes and weights of Fejer's first quadrature rule with a Jacobi weight.


*source:*
[FastTransforms/src/fejer.jl:7](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/b1dce339ed19119766d44b5ec44d954932da58d3/src/fejer.jl#L7)

---

<a id="method__fejer2.1" class="lexicon_definition"></a>
#### fejer2{T<:AbstractFloat}(N::Int64,  α::T<:AbstractFloat,  β::T<:AbstractFloat) [¶](#method__fejer2.1)
Compute nodes and weights of Fejer's second quadrature rule with a Jacobi weight.


*source:*
[FastTransforms/src/fejer.jl:12](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/b1dce339ed19119766d44b5ec44d954932da58d3/src/fejer.jl#L12)

---

<a id="method__fejerweights1.1" class="lexicon_definition"></a>
#### fejerweights1{T<:AbstractFloat}(N::Int64,  α::T<:AbstractFloat,  β::T<:AbstractFloat) [¶](#method__fejerweights1.1)
Compute weights of Fejer's first quadrature rule with a Jacobi weight.


*source:*
[FastTransforms/src/fejer.jl:21](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/b1dce339ed19119766d44b5ec44d954932da58d3/src/fejer.jl#L21)

---

<a id="method__fejerweights2.1" class="lexicon_definition"></a>
#### fejerweights2{T<:AbstractFloat}(N::Int64,  α::T<:AbstractFloat,  β::T<:AbstractFloat) [¶](#method__fejerweights2.1)
Compute weights of Fejer's second quadrature rule with a Jacobi weight.


*source:*
[FastTransforms/src/fejer.jl:26](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/b1dce339ed19119766d44b5ec44d954932da58d3/src/fejer.jl#L26)

---

<a id="method__pochhammer.1" class="lexicon_definition"></a>
#### pochhammer(x::Number,  n::Integer) [¶](#method__pochhammer.1)
Pochhammer symbol (x)_n = Γ(x+n)/Γ(x) for the rising factorial.


*source:*
[FastTransforms/src/specialfunctions.jl:13](https://github.com/MikaelSlevinsky/FastTransforms.jl/tree/b1dce339ed19119766d44b5ec44d954932da58d3/src/specialfunctions.jl#L13)


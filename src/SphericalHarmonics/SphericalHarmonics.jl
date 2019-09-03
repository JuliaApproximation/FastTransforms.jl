abstract type SphericalHarmonicPlan{T} end

function *(P::SphericalHarmonicPlan, X::AbstractMatrix)
    mul!(zero(X), P, X)
end


function \(P::SphericalHarmonicPlan, X::AbstractMatrix)
    mul!(zero(X), transpose(P), X)
end

include("sphfunctions.jl")
include("slowplan.jl")
include("Butterfly.jl")
include("fastplan.jl")
include("thinplan.jl")
include("synthesisanalysis.jl")
include("vectorfield.jl")

function plan_sph2fourier(A::AbstractMatrix; opts...)
    M, N = size(A)
    if M ≤ 1023
        SlowSphericalHarmonicPlan(A)
    else
        ThinSphericalHarmonicPlan(A; opts...)
    end
end

sph2fourier(A::AbstractMatrix; opts...) = plan_sph2fourier(A; opts...)*A
fourier2sph(A::AbstractMatrix; opts...) = plan_sph2fourier(A; opts...)\A

"""
Computes the bivariate Fourier series given by the spherical harmonic expansion:

```math
{\\rm SHT} : \\sum_{\\ell=0}^n\\sum_{m=-\\ell}^{\\ell} f_{\\ell}^m Y_{\\ell}^m(\\theta,\\varphi) \\to \\sum_{\\ell=0}^n\\sum_{m=-n}^{n} g_{\\ell}^m \\left\\{\\begin{array}{ccc}\\cos\\ell\\theta & {\\rm for} & m{\\rm~even}\\\\ \\sin(\\ell+1)\\theta & {\\rm for} & m{\\rm~odd}\\end{array}\\right\\}\\times \\sqrt{\\frac{2-\\delta_{m,0}}{2\\pi}} \\left\\{\\begin{array}{ccc} \\cos m\\varphi & {\\rm for} & m \\ge 0,\\\\ \\sin(-m\\varphi) & {\\rm for} & m < 0.\\end{array}\\right.
```

The spherical harmonic expansion coefficients are organized as follows:

```math
F = \\begin{pmatrix}
f_0^0 & f_1^{-1} & f_1^1 & f_2^{-2} & f_2^2 & \\cdots & f_n^{-n} & f_n^n\\\\
f_1^0 & f_2^{-1} & f_2^1 & f_3^{-2} & f_3^2 & \\cdots & 0 & 0\\\\
\\vdots & \\vdots & \\vdots &  \\vdots &  \\vdots & \\ddots & \\vdots & \\vdots\\\\
f_{n-2}^0 & f_{n-1}^{-1} & f_{n-1}^1 & f_n^{-2} & f_n^2 &  & \\vdots & \\vdots\\\\
f_{n-1}^0 & f_n^{-1} & f_n^1 & 0 & 0 & \\cdots & 0 & 0\\\\
f_n^0 & 0 & 0 & 0 & 0 & \\cdots & 0 & 0\\\\
\\end{pmatrix},
```

and the Fourier coefficients are organized similarly:

```math
G = \\begin{pmatrix}
g_0^0 & g_0^{-1} & g_0^1 & \\cdots & g_0^{-n} & g_0^n\\\\
g_1^0 & g_1^{-1} & g_1^1 & \\cdots & g_1^{-n} & g_1^n\\\\
\\vdots & \\vdots & \\vdots & \\ddots & \\vdots & \\vdots\\\\
g_{n-1}^0 & g_{n-1}^{-1} & g_{n-1}^1& \\cdots & g_{n-1}^{-n} & g_{n-1}^n\\\\
g_n^0 & 0 & 0 & \\cdots & g_n^{-n} & g_n^n\\\\
\\end{pmatrix}.
```
"""
sph2fourier(::AbstractMatrix; opts...)

"""
Computes the spherical harmonic expansion given by the bivariate Fourier series:

```math
{\\rm iSHT} : \\sum_{\\ell=0}^n\\sum_{m=-n}^{n} g_{\\ell}^m \\left\\{\\begin{array}{ccc}\\cos\\ell\\theta & {\\rm for} & m{\\rm~even}\\\\ \\sin(\\ell+1)\\theta & {\\rm for} & m{\\rm~odd}\\end{array}\\right\\}\\times \\sqrt{\\frac{2-\\delta_{m,0}}{2\\pi}} \\left\\{\\begin{array}{ccc} \\cos m\\varphi & {\\rm for} & m \\ge 0,\\\\ \\sin(-m\\varphi) & {\\rm for} & m < 0,\\end{array}\\right. \\to \\sum_{\\ell=0}^n\\sum_{m=-\\ell}^{\\ell} f_{\\ell}^m Y_{\\ell}^m(\\theta,\\varphi).
```

The spherical harmonic expansion coefficients are organized as follows:

```math
F = \\begin{pmatrix}
f_0^0 & f_1^{-1} & f_1^1 & f_2^{-2} & f_2^2 & \\cdots & f_n^{-n} & f_n^n\\\\
f_1^0 & f_2^{-1} & f_2^1 & f_3^{-2} & f_3^2 & \\cdots & 0 & 0\\\\
\\vdots & \\vdots & \\vdots &  \\vdots &  \\vdots & \\ddots & \\vdots & \\vdots\\\\
f_{n-2}^0 & f_{n-1}^{-1} & f_{n-1}^1 & f_n^{-2} & f_n^2 &  & \\vdots & \\vdots\\\\
f_{n-1}^0 & f_n^{-1} & f_n^1 & 0 & 0 & \\cdots & 0 & 0\\\\
f_n^0 & 0 & 0 & 0 & 0 & \\cdots & 0 & 0\\\\
\\end{pmatrix},
```

and the Fourier coefficients are organized similarly:

```math
G = \\begin{pmatrix}
g_0^0 & g_0^{-1} & g_0^1 & \\cdots & g_0^{-n} & g_0^n\\\\
g_1^0 & g_1^{-1} & g_1^1 & \\cdots & g_1^{-n} & g_1^n\\\\
\\vdots & \\vdots & \\vdots & \\ddots & \\vdots & \\vdots\\\\
g_{n-1}^0 & g_{n-1}^{-1} & g_{n-1}^1& \\cdots & g_{n-1}^{-n} & g_{n-1}^n\\\\
g_n^0 & 0 & 0 & \\cdots & g_n^{-n} & g_n^n\\\\
\\end{pmatrix}.
```
"""
fourier2sph(::AbstractMatrix; opts...)

"""
Pre-computes the spherical harmonic transform.
"""
plan_sph2fourier(::AbstractMatrix; opts...)

var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#FastTransforms.jl-Documentation-1",
    "page": "Home",
    "title": "FastTransforms.jl Documentation",
    "category": "section",
    "text": ""
},

{
    "location": "index.html#FastTransforms.cjt",
    "page": "Home",
    "title": "FastTransforms.cjt",
    "category": "Function",
    "text": "Computes the Chebyshev expansion coefficients given the Jacobi expansion coefficients c with parameters  and .\n\nSee also icjt and jjt.\n\n\n\n"
},

{
    "location": "index.html#FastTransforms.icjt",
    "page": "Home",
    "title": "FastTransforms.icjt",
    "category": "Function",
    "text": "Computes the Jacobi expansion coefficients with parameters  and  given the Chebyshev expansion coefficients c.\n\nSee also cjt and jjt.\n\n\n\n"
},

{
    "location": "index.html#FastTransforms.jjt",
    "page": "Home",
    "title": "FastTransforms.jjt",
    "category": "Function",
    "text": "Computes the Jacobi expansion coefficients with parameters  and  given the Jacobi expansion coefficients c with parameters  and .\n\nSee also cjt and icjt.\n\n\n\n"
},

{
    "location": "index.html#FastTransforms.plan_cjt",
    "page": "Home",
    "title": "FastTransforms.plan_cjt",
    "category": "Function",
    "text": "Pre-plan optimized DCT-I and DST-I plans and pre-allocate the necessary arrays, normalization constants, and recurrence coefficients for a forward Chebyshev—Jacobi transform.\n\nc is the vector of coefficients; and,\n\n and  are the Jacobi parameters.\n\nOptionally:\n\nM determines the number of terms in Hahn's asymptotic expansion.\n\n\n\n"
},

{
    "location": "index.html#FastTransforms.plan_icjt",
    "page": "Home",
    "title": "FastTransforms.plan_icjt",
    "category": "Function",
    "text": "Pre-plan optimized DCT-I and DST-I plans and pre-allocate the necessary arrays, normalization constants, and recurrence coefficients for an inverse Chebyshev—Jacobi transform.\n\nc is the vector of coefficients; and,\n\n and  are the Jacobi parameters.\n\nOptionally:\n\nM determines the number of terms in Hahn's asymptotic expansion.\n\n\n\n"
},

{
    "location": "index.html#FastTransforms.paduatransform",
    "page": "Home",
    "title": "FastTransforms.paduatransform",
    "category": "Function",
    "text": "Padua Transform maps from interpolant values at the Padua points to the 2D Chebyshev coefficients.\n\n\n\n"
},

{
    "location": "index.html#FastTransforms.ipaduatransform",
    "page": "Home",
    "title": "FastTransforms.ipaduatransform",
    "category": "Function",
    "text": "Inverse Padua Transform maps the 2D Chebyshev coefficients to the values of the interpolation polynomial at the Padua points.\n\n\n\n"
},

{
    "location": "index.html#FastTransforms.plan_paduatransform!",
    "page": "Home",
    "title": "FastTransforms.plan_paduatransform!",
    "category": "Function",
    "text": "Pre-plan a Padua Transform.\n\n\n\n"
},

{
    "location": "index.html#FastTransforms.plan_ipaduatransform!",
    "page": "Home",
    "title": "FastTransforms.plan_ipaduatransform!",
    "category": "Function",
    "text": "Pre-plan an Inverse Padua Transform.\n\n\n\n"
},

{
    "location": "index.html#Fast-Transforms-1",
    "page": "Home",
    "title": "Fast Transforms",
    "category": "section",
    "text": "cjticjtjjtplan_cjtplan_icjtpaduatransformipaduatransformplan_paduatransform!plan_ipaduatransform!"
},

{
    "location": "index.html#FastTransforms.gaunt",
    "page": "Home",
    "title": "FastTransforms.gaunt",
    "category": "Function",
    "text": "Calculates the Gaunt coefficients, defined by:\n\na(m,n,μ,ν,q) = (2(n+ν-2q)+1)/2 (n+ν-2q-m-μ)!/(n+ν-2q+m+μ)! ∫₋₁⁺¹ P_m^n(x) P_ν^μ(x) P_{n+ν-2q}^{m+μ}(x) dx.\n\nor defined by:\n\nP_n^m(x) P_ν^μ(x) = ∑_{q=0}^{q_{max}} a(m,n,μ,ν,q) P_{n+ν-2q}^{m+μ}(x)\n\nThis is a Julia implementation of the stable recurrence described in:\n\nY.-l. Xu, \"Fast evaluation of Gaunt coefficients: recursive approach\", J. Comp. Appl. Math., 85:53–65, 1997.\n\n\n\nCalculates the Gaunt coefficients in 64-bit floating-point arithmetic.\n\n\n\n"
},

{
    "location": "index.html#FastTransforms.paduapoints",
    "page": "Home",
    "title": "FastTransforms.paduapoints",
    "category": "Function",
    "text": "Returns coordinates of the (n+1)(n+2)/2 Padua points.\n\n\n\n"
},

{
    "location": "index.html#Other-Exported-Methods-1",
    "page": "Home",
    "title": "Other Exported Methods",
    "category": "section",
    "text": "gauntpaduapoints"
},

{
    "location": "index.html#FastTransforms.clenshawcurtis",
    "page": "Home",
    "title": "FastTransforms.clenshawcurtis",
    "category": "Function",
    "text": "Compute nodes and weights of the Clenshaw—Curtis quadrature rule with a Jacobi weight.\n\n\n\n"
},

{
    "location": "index.html#FastTransforms.clenshawcurtisweights",
    "page": "Home",
    "title": "FastTransforms.clenshawcurtisweights",
    "category": "Function",
    "text": "Compute weights of the Clenshaw—Curtis quadrature rule with a Jacobi weight.\n\n\n\n"
},

{
    "location": "index.html#FastTransforms.fejer1",
    "page": "Home",
    "title": "FastTransforms.fejer1",
    "category": "Function",
    "text": "Compute nodes and weights of Fejer's first quadrature rule with a Jacobi weight.\n\n\n\n"
},

{
    "location": "index.html#FastTransforms.fejer2",
    "page": "Home",
    "title": "FastTransforms.fejer2",
    "category": "Function",
    "text": "Compute nodes and weights of Fejer's second quadrature rule with a Jacobi weight.\n\n\n\n"
},

{
    "location": "index.html#FastTransforms.fejerweights1",
    "page": "Home",
    "title": "FastTransforms.fejerweights1",
    "category": "Function",
    "text": "Compute weights of Fejer's first quadrature rule with a Jacobi weight.\n\n\n\n"
},

{
    "location": "index.html#FastTransforms.fejerweights2",
    "page": "Home",
    "title": "FastTransforms.fejerweights2",
    "category": "Function",
    "text": "Compute weights of Fejer's second quadrature rule with a Jacobi weight.\n\n\n\n"
},

{
    "location": "index.html#FastTransforms.half",
    "page": "Home",
    "title": "FastTransforms.half",
    "category": "Function",
    "text": "Compute a typed 0.5.\n\n\n\n"
},

{
    "location": "index.html#FastTransforms.two",
    "page": "Home",
    "title": "FastTransforms.two",
    "category": "Function",
    "text": "Compute a typed 2.\n\n\n\n"
},

{
    "location": "index.html#FastTransforms.δ",
    "page": "Home",
    "title": "FastTransforms.δ",
    "category": "Function",
    "text": "The Kronecker δ function.\n\n\n\n"
},

{
    "location": "index.html#FastTransforms.Λ",
    "page": "Home",
    "title": "FastTransforms.Λ",
    "category": "Function",
    "text": "The Lambda function Λ(z) = Γ(z+½)/Γ(z+1) for the ratio of gamma functions.\n\n\n\nFor 64-bit floating-point arithmetic, the Lambda function uses the asymptotic series for τ in Appendix B of\n\nI. Bogaert and B. Michiels and J. Fostier, 𝒪(1) computation of Legendre polynomials and Gauss–Legendre nodes and weights for parallel computing, SIAM J. Sci. Comput., 34:C83–C101, 2012.\n\n\n\nThe Lambda function Λ(z,λ₁,λ₂) = Γ(z+λ₁)/Γ(z+λ₂) for the ratio of gamma functions.\n\n\n\n"
},

{
    "location": "index.html#FastTransforms.pochhammer",
    "page": "Home",
    "title": "FastTransforms.pochhammer",
    "category": "Function",
    "text": "Pochhammer symbol (x)_n = fracGamma(x+n)Gamma(x) for the rising factorial.\n\n\n\n"
},

{
    "location": "index.html#FastTransforms.stirlingseries",
    "page": "Home",
    "title": "FastTransforms.stirlingseries",
    "category": "Function",
    "text": "Stirling series for Γ(z).\n\n\n\n"
},

{
    "location": "index.html#Internal-Methods-1",
    "page": "Home",
    "title": "Internal Methods",
    "category": "section",
    "text": "FastTransforms.clenshawcurtisFastTransforms.clenshawcurtisweightsFastTransforms.fejer1FastTransforms.fejer2FastTransforms.fejerweights1FastTransforms.fejerweights2FastTransforms.halfFastTransforms.twoFastTransforms.δFastTransforms.ΛFastTransforms.pochhammerFastTransforms.stirlingseries"
},

]}

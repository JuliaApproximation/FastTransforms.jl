doc"""
Calculates the Gaunt coefficients, defined by:

    a(m,n,μ,ν,q) = (2(n+ν-2q)+1)/2 (n+ν-2q-m-μ)!/(n+ν-2q+m+μ)! ∫₋₁⁺¹ P_m^n(x) P_ν^μ(x) P_{n+ν-2q}^{m+μ}(x) dx.

or defined by:

    P_n^m(x) P_ν^μ(x) = ∑_{q=0}^{q_{max}} a(m,n,μ,ν,q) P_{n+ν-2q}^{m+μ}(x)

This is a Julia implementation of the stable recurrence described in:

    Y.-l. Xu, "Fast evaluation of Gaunt coefficients: recursive approach", J. Comp. Appl. Math., 85:53–65, 1997.
"""
function gaunt{T}(::Type{T},m::Int,n::Int,μ::Int,ν::Int;normalized::Bool=false)
    if normalized
        normalizedgaunt(T,m,n,μ,ν)
    else
        scale!(normalization(T,m,n,μ,ν),gaunt(T,m,n,μ,ν;normalized=true))
    end
end
doc"""
Calculates the Gaunt coefficients in 64-bit floating-point arithmetic.
"""
gaunt(m::Int,n::Int,μ::Int,ν::Int;kwds...) = gaunt(Float64,m,n,μ,ν;kwds...)

function normalization{T}(::Type{T},m::Int,n::Int,μ::Int,ν::Int)
    pochhammer(n+one(T),n)*pochhammer(ν+one(T),ν)/pochhammer(n+ν+one(T),n+ν)*gamma(n+ν-m-μ+one(T))/gamma(n-m+one(T))/gamma(ν-μ+one(T))
end

normalization(::Type{Float64},m::Int,n::Int,μ::Int,ν::Int) = normalization1(Float64,n,ν)*normalization2(Float64,n-m,ν-μ)

function normalization1(::Type{Float64},n::Int,ν::Int)
    if n ≥ 8
        if ν ≥ 8
            return exp((n+0.5)*log1p(n/(n+1))+(ν+0.5)*log1p(ν/(ν+1))+(n+ν+0.5)*log1p(-(n+ν)/(2n+2ν+1))+n*log1p(-2ν/(2n+2ν+1))+ν*log1p(-2n/(2n+2ν+1)))*stirlingseries(2n+1.0)*stirlingseries(2ν+1.0)*stirlingseries(n+ν+1.0)/stirlingseries(n+1.0)/stirlingseries(ν+1.0)/stirlingseries(2n+2ν+1.0)
        else
            return pochhammer(ν+1.0,ν)/(2n+2ν+1.0)^ν*exp(ν+(n+0.5)*log1p(n/(n+1))+(n+ν+0.5)*log1p(-(n+ν)/(2n+2ν+1))+n*log1p(-2ν/(2n+2ν+1)))*stirlingseries(2n+1.0)*stirlingseries(n+ν+1.0)/stirlingseries(n+1.0)/stirlingseries(2n+2ν+1.0)
        end
    elseif ν ≥ 8
        return pochhammer(n+1.0,n)/(2n+2ν+1.0)^n*exp(n+(ν+0.5)*log1p(ν/(ν+1))+(n+ν+0.5)*log1p(-(n+ν)/(2n+2ν+1))+ν*log1p(-2n/(2n+2ν+1)))*stirlingseries(2ν+1.0)*stirlingseries(n+ν+1.0)/stirlingseries(ν+1.0)/stirlingseries(2n+2ν+1.0)
    else
        return pochhammer(n+1.0,n)*pochhammer(ν+1.0,ν)/pochhammer(n+ν+1.0,n+ν)
    end
end

function normalization2(::Type{Float64},nm::Int,νμ::Int)
    if nm ≥ 8
        if νμ ≥ 8
            return edivsqrt2pi*exp((nm+0.5)*log1p(νμ/(nm+1))+(νμ+0.5)*log1p(nm/(νμ+1)))/sqrt(nm+νμ+1.0)*stirlingseries(nm+νμ+1.0)/stirlingseries(nm+1.0)/stirlingseries(νμ+1.0)
        else
            return (nm+νμ+1.0)^νμ*exp(-νμ+(nm+0.5)*log1p(νμ/(nm+1)))*stirlingseries(nm+νμ+1.0)/stirlingseries(nm+1.0)/gamma(νμ+1.0)
        end
    elseif νμ ≥ 8
        return (nm+νμ+1.0)^nm*exp(-nm+(νμ+0.5)*log1p(nm/(νμ+1)))*stirlingseries(nm+νμ+1.0)/stirlingseries(νμ+1.0)/gamma(nm+1.0)
    else
        return gamma(nm+νμ+1.0)/gamma(nm+1.0)/gamma(νμ+1.0)
    end
end

function normalizedgaunt{T}(::Type{T},m::Int,n::Int,μ::Int,ν::Int)
    qmax = min(n,ν,(n+ν-abs(m+μ))÷2)
    a = Vector{T}(qmax+1)
    a[1] = one(T)
    if μ == m && ν == n # zero class (i) of Aₚ
        if μ == m == 0
            for q = 1:qmax
                p = n+ν-2q
                a[q+1] = α(T,n,ν,p+2)/α(T,n,ν,p+1)*a[q]
            end
        else
            for q = 1:qmax
                p = n+ν-2q
                p₁,p₂ = p-m-μ,p+m+μ
                a[q+1] = (p+1)*(p₂+2)*α(T,n,ν,p+2)/(p+2)/(p₁+1)/α(T,n,ν,p+1)*a[q]
            end
        end
    else
        qmax > 0 && (a[2] = secondinitialcondition(T,m,n,μ,ν))
        q = 2
        if qmax > 1
            p = n+ν-2q
            p₁,p₂ = p-m-μ,p+m+μ
            if A(T,m,n,μ,ν,p+4) != 0
                a[q+1] = (c₁(T,m,n,μ,ν,p,p₁,p₂)*a[q] + c₂(T,m,n,μ,ν,p,p₂)*a[q-1])/c₀(T,m,n,μ,ν,p,p₁)
            else
                a[q+1] = thirdinitialcondition(T,m,n,μ,ν)
            end
            q+=1
        end
        while q ≤ qmax
            p = n+ν-2q
            p₁,p₂ = p-m-μ,p+m+μ
            if A(T,m,n,μ,ν,p+4) != 0
                a[q+1] = (c₁(T,m,n,μ,ν,p,p₁,p₂)*a[q] + c₂(T,m,n,μ,ν,p,p₂)*a[q-1])/c₀(T,m,n,μ,ν,p,p₁)
            elseif A(T,m,n,μ,ν,p+6) != 0
                a[q+1] = (d₁(T,m,n,μ,ν,p,p₁,p₂)*a[q] + d₂(T,m,n,μ,ν,p,p₁,p₂)*a[q-1] + d₃(T,m,n,μ,ν,p,p₂)*a[q-2])/d₀(T,m,n,μ,ν,p,p₁)
            else
                a[q+1] = (p+1)*(p₂+2)*α(T,n,ν,p+2)/(p+2)/(p₁+1)/α(T,n,ν,p+1)*a[q]
            end
            q+=1
        end
    end
    a
end

function secondinitialcondition{T}(::Type{T},m::Int,n::Int,μ::Int,ν::Int)
    n₄ = n+ν-m-μ
    mn = m-n
    μν = μ-ν
    temp = 2n+2ν-one(T)
    return (temp-2)/2*(1-temp/n₄/(n₄-1)*(mn*(mn+one(T))/(2n-1)+μν*(μν+one(T))/(2ν-1)))
end

function thirdinitialcondition{T}(::Type{T},m::Int,n::Int,μ::Int,ν::Int)
    n₄ = n+ν-m-μ
    mn = m-n
    μν = μ-ν
    temp = 2n+2ν-one(T)
    temp1 = mn*(mn+one(T))*(mn+2)*(mn+3)/(2n-1)/(2n-3) + 2mn*(mn+one(T))*μν*(μν+one(T))/(2n-1)/(2ν-1) + μν*(μν+one(T))*(μν+2)*(μν+3)/(2ν-1)/(2ν-3)
    temp2 = (temp-4)/(2(n₄-2)*(n₄-3))*temp1 - mn*(mn+one(T))/(2n-1)-μν*(μν+one(T))/(2ν-1)
    return temp*(temp-6)/4*( (temp-2)/n₄/(n₄-1)*temp2 + one(T)/2 )
end

α{T}(::Type{T},n::Int,ν::Int,p::Int) = (p^2-(n+ν+1)^2)*(p^2-(n-ν)^2)/(4p^2-one(T))
A{T}(::Type{T},m::Int,n::Int,μ::Int,ν::Int,p::Int) = p*(p-one(T))*(m-μ)-(m+μ)*(n-ν)*(n+ν+one(T))

c₀{T}(::Type{T},m::Int,n::Int,μ::Int,ν::Int,p::Int,p₁::Int) = (p+2)*(p+3)*(p₁+1)*(p₁+2)*A(T,m,n,μ,ν,p+4)*α(T,n,ν,p+1)
c₁{T}(::Type{T},m::Int,n::Int,μ::Int,ν::Int,p::Int,p₁::Int,p₂::Int) = A(T,m,n,μ,ν,p+2)*A(T,m,n,μ,ν,p+3)*A(T,m,n,μ,ν,p+4) + (p+1)*(p+3)*(p₁+2)*(p₂+2)*A(T,m,n,μ,ν,p+4)*α(T,n,ν,p+2) + (p+2)*(p+4)*(p₁+3)*(p₂+3)*A(T,m,n,μ,ν,p+2)*α(T,n,ν,p+3)
c₂{T}(::Type{T},m::Int,n::Int,μ::Int,ν::Int,p::Int,p₂::Int) = -(p+2)*(p+3)*(p₂+3)*(p₂+4)*A(T,m,n,μ,ν,p+2)*α(T,n,ν,p+4)

d₀{T}(::Type{T},m::Int,n::Int,μ::Int,ν::Int,p::Int,p₁::Int) = (p+2)*(p+3)*(p+5)*(p₁+2)*(p₁+4)*A(T,m,n,μ,ν,p+6)*α(T,n,ν,p+1)
d₁{T}(::Type{T},m::Int,n::Int,μ::Int,ν::Int,p::Int,p₁::Int,p₂::Int) = (p+5)*(p₁+4)*A(T,m,n,μ,ν,p+6)*( A(T,m,n,μ,ν,p+2)*A(T,m,n,μ,ν,p+3) + (p+1)*(p+3)*(p₁+2)*(p₂+2)*α(T,n,ν,p+2) )
d₂{T}(::Type{T},m::Int,n::Int,μ::Int,ν::Int,p::Int,p₁::Int,p₂::Int) = (p+2)*(p₂+3)*A(T,m,n,μ,ν,p+2)*( A(T,m,n,μ,ν,p+5)*A(T,m,n,μ,ν,p+6) + (p+4)*(p+6)*(p₁+5)*(p₂+5)*α(T,n,ν,p+5) )
d₃{T}(::Type{T},m::Int,n::Int,μ::Int,ν::Int,p::Int,p₂::Int) = -(p+2)*(p+4)*(p+5)*(p₂+3)*(p₂+5)*(p₂+6)*A(T,m,n,μ,ν,p+2)*α(T,n,ν,p+6)

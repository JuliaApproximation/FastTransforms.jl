export plan_chebyshevtransform, plan_ichebyshevtransform, plan_chebyshevtransform!, plan_ichebyshevtransform!,
            chebyshevtransform, ichebyshevtransform, chebyshevpoints,
            plan_chebyshevutransform, plan_ichebyshevutransform, plan_chebyshevutransform!, plan_ichebyshevutransform!,
            chebyshevutransform, ichebyshevutransform

## Transforms take values at Chebyshev points of the first and second kinds and produce Chebyshev coefficients


struct ChebyshevTransformPlan{T,kind,inplace,P} <: Plan{T}
    plan::P
end

ChebyshevTransformPlan{k,inp}(plan) where {k,inp} =
    ChebyshevTransformPlan{eltype(plan),k,inp,typeof(plan)}(plan)



function plan_chebyshevtransform!(x::AbstractVector{T}; kind::Integer=1) where T<:fftwNumber
    if kind == 1
        plan = isempty(x) ? fill(one(T),1,length(x)) : plan_r2r!(x, REDFT10)
        ChebyshevTransformPlan{1,true}(plan)
    elseif kind == 2
        plan = length(x) ≤ 1 ? fill(one(T),1,length(x)) : plan_r2r!(x, REDFT00)
        ChebyshevTransformPlan{2,true}(plan)
    end
end

function plan_chebyshevtransform(x::AbstractVector{T};kind::Integer=1) where T<:fftwNumber
    plan = plan_chebyshevtransform!(x;kind=kind)
    ChebyshevTransformPlan{kind,false}(plan)
end

function *(P::ChebyshevTransformPlan{T,1,true},x::AbstractVector{T}) where T
    n = length(x)
    n ≤ 1 && return x

    x = P.plan*x
    x[1] /= 2
    lmul!(inv(convert(T,n)), x)
end

function *(P::ChebyshevTransformPlan{T,2,true}, x::AbstractVector{T}) where T
    n = length(x)
    n ≤ 1 && return x

    x = P.plan*x
    x[1] /= 2; x[end] /= 2
    lmul!(inv(convert(T,n-1)),x)
end

chebyshevtransform!(x::AbstractVector{T};kind::Integer=1) where {T<:fftwNumber} =
    plan_chebyshevtransform!(x;kind=kind)*x

chebyshevtransform(x;kind::Integer=1) = chebyshevtransform!(copy(x);kind=kind)

*(P::ChebyshevTransformPlan{T,k,false}, x::AbstractVector{T}) where {T,k} = P.plan*copy(x)




## Inverse transforms take Chebyshev coefficients and produce values at Chebyshev points of the first and second kinds


struct IChebyshevTransformPlan{T,kind,inplace,P}
    plan::P
end

# second kind Chebyshev transforms share a plan with their inverse
# so we support this via inv
inv(P::ChebyshevTransformPlan{T,2,true}) where T = IChebyshevTransformPlan{T,2,true,typeof(P)}(P)
inv(P::IChebyshevTransformPlan{T,2,true}) where T = P.plan

\(P::ChebyshevTransformPlan, x::AbstractArray) = inv(P) * x
\(P::IChebyshevTransformPlan, x::AbstractArray) = inv(P) * x


function plan_ichebyshevtransform!(x::AbstractVector{T};kind::Integer=1) where T<:fftwNumber
    if kind == 1
        plan = isempty(x) ? fill(one(T),1,length(x)) : plan_r2r!(x, REDFT01)
        IChebyshevTransformPlan{T,1,true,typeof(plan)}(plan)
    elseif kind == 2
        inv(plan_chebyshevtransform!(x;kind=2))
    end
end

function plan_ichebyshevtransform(x::AbstractVector{T};kind::Integer=1) where T<:fftwNumber
    plan = plan_ichebyshevtransform!(similar(Vector{T},axes(x));kind=kind)
    IChebyshevTransformPlan{T,kind,false,typeof(plan)}(plan)
end

function *(P::IChebyshevTransformPlan{T,1,true},x::AbstractVector{T}) where T<:fftwNumber
    isempty(x) && return x
    x[1] *=2
    x = lmul!(convert(T,0.5), P.plan*x)
    x
end

function *(P::IChebyshevTransformPlan{T,2,true},x::AbstractVector{T}) where T<:fftwNumber
    n = length(x)
    n ≤ 1 && return x
    x[1] *= 2; x[end] *= 2
    x = P.plan*x
    x[1] *= 2; x[end] *= 2
    lmul!(convert(T,0.5(n-1)),x)
end

ichebyshevtransform!(x::AbstractVector{T};kind::Integer=1) where {T<:fftwNumber} =
    plan_ichebyshevtransform!(x;kind=kind)*x

ichebyshevtransform(x;kind::Integer=1) = ichebyshevtransform!(copy(x); kind=kind)

*(P::IChebyshevTransformPlan{T,k,false},x::AbstractVector{T}) where {T,k} = P.plan*copy(x)

## Code generation for integer inputs

for func in (:chebyshevtransform,:ichebyshevtransform)
    @eval $func(x::AbstractVector{T};kind::Integer=1) where {T<:Integer} = $func(convert(Float64,x);kind=kind)
end


# Matrix inputs
# 
#
# function chebyshevtransform!(X::AbstractMatrix{T}; kind::Integer=1) where T<:fftwNumber
#     if kind == 1
#         if size(X) == (1,1)
#             X
#         else
#             X=r2r!(X,REDFT10)
#             X[:,1]/=2;X[1,:]/=2;
#             lmul!(1/(size(X,1)*size(X,2)),X)
#         end
#     elseif kind == 2
#         if size(X) == (1,1)
#             X
#         else
#             X=r2r!(X,REDFT00)
#             lmul!(1/((size(X,1)-1)*(size(X,2)-1)),X)
#             X[:,1]/=2;X[:,end]/=2
#             X[1,:]/=2;X[end,:]/=2
#             X
#         end
#     end
# end
#
# function ichebyshevtransform!(X::AbstractMatrix{T}; kind::Integer=1) where T<:fftwNumber
#     if kind == 1
#         if size(X) == (1,1)
#             X
#         else
#             X[1,:]*=2;X[:,1]*=2
#             X = r2r(X,REDFT01)
#             lmul!(0.25, X)
#         end
#     elseif kind == 2
#         if size(X) == (1,1)
#             X
#         else
#             X[1,:]*=2;X[end,:]*=2;X[:,1]*=2;X[:,end]*=2
#             X=chebyshevtransform!(X;kind=kind)
#             X[1,:]*=2;X[end,:]*=2;X[:,1]*=2;X[:,end]*=2
#             lmul!((size(X,1)-1)*(size(X,2)-1)/4,X)
#         end
#     end
# end
#


## Chebyshev U

struct ChebyshevUTransformPlan{T,kind,inplace,P} <: Plan{T}
    plan::P
end

ChebyshevUTransformPlan{k,inp}(plan) where {k,inp} =
    ChebyshevUTransformPlan{eltype(plan),k,inp,typeof(plan)}(plan)



function plan_chebyshevutransform!(x::AbstractVector{T}; kind::Integer=1) where T<:fftwNumber
    if kind == 1
        plan = isempty(x) ? fill(one(T),1,length(x)) : plan_r2r!(x, RODFT10)
        ChebyshevUTransformPlan{1,true}(plan)
    elseif kind == 2
        plan = length(x) ≤ 1 ? fill(one(T),1,length(x)) : plan_r2r!(x, RODFT00)
        ChebyshevUTransformPlan{2,true}(plan)
    end
end

function plan_chebyshevutransform(x::AbstractVector{T};kind::Integer=1) where T<:fftwNumber
    plan = plan_chebyshevutransform!(x;kind=kind)
    ChebyshevUTransformPlan{kind,false}(plan)
end

function *(P::ChebyshevUTransformPlan{T,1,true},x::AbstractVector{T}) where T
    n = length(x)
    n ≤ 1 && return x

    for k=1:n # sqrt(1-x_j^2) weight
        x[k] *= sinpi(one(T)/(2n) + (k-one(T))/n)/n
    end
    P.plan * x
end

function *(P::ChebyshevUTransformPlan{T,2,true},x::AbstractVector{T}) where T
    n = length(x)
    n ≤ 1 && return x

    c = one(T)/ (n+1)
    for k=1:n # sqrt(1-x_j^2) weight
        x[k] *= sinpi(k*c)
    end
    lmul!(c, P.plan * x)
end

chebyshevutransform!(x::AbstractVector{T};kind::Integer=1) where {T<:fftwNumber} =
    plan_chebyshevutransform!(x;kind=kind)*x

chebyshevutransform(x;kind::Integer=1) = chebyshevutransform!(copy(x);kind=kind)

*(P::ChebyshevUTransformPlan{T,k,false},x::AbstractVector{T}) where {T,k} = P.plan*copy(x)

## Inverse transforms take ChebyshevU coefficients and produce values at ChebyshevU points of the first and second kinds


struct IChebyshevUTransformPlan{T,kind,inplace,P}
    plan::P
end


function plan_ichebyshevutransform!(x::AbstractVector{T};kind::Integer=1) where T<:fftwNumber
    if kind == 1
        plan = isempty(x) ? fill(one(T),1,length(x)) : plan_r2r!(x, RODFT01)
        IChebyshevUTransformPlan{T,1,true,typeof(plan)}(plan)
    elseif kind == 2
        plan = length(x) ≤ 1 ? fill(one(T),1,length(x)) : plan_r2r!(x, RODFT00)
        IChebyshevUTransformPlan{T,2,true,typeof(plan)}(plan)
    end
end

function plan_ichebyshevutransform(x::AbstractVector{T}; kind::Integer=1) where T<:fftwNumber
    plan = plan_ichebyshevutransform!(similar(Vector{T},axes(x)); kind=kind)
    IChebyshevUTransformPlan{T,kind,false,typeof(plan)}(plan)
end

function *(P::IChebyshevUTransformPlan{T,1,true}, x::AbstractVector{T}) where T<:fftwNumber
    n = length(x)
    n ≤ 1 && return x

    x = P.plan * x
    for k=1:n # sqrt(1-x_j^2) weight
        x[k] /= 2sinpi(one(T)/(2n) + (k-one(T))/n)
    end
    x
end



function *(P::IChebyshevUTransformPlan{T,2,true}, x::AbstractVector{T}) where T<:fftwNumber
    n = length(x)
    n ≤ 1 && return x

    c = one(T)/ (n+1)
    lmul!((n+1)/(2n+2*one(T)), x)
    x = P.plan * x
    for k=1:n # sqrt(1-x_j^2) weight
        x[k] /= sinpi(k*c)
    end
    x
end

ichebyshevutransform!(x::AbstractVector{T};kind::Integer=1) where {T<:fftwNumber} =
    plan_ichebyshevutransform!(x;kind=kind)*x

ichebyshevutransform(x;kind::Integer=1) = ichebyshevutransform!(copy(x);kind=kind)

*(P::IChebyshevUTransformPlan{T,k,false},x::AbstractVector{T}) where {T,k} = P.plan*copy(x)

## Code generation for integer inputs

for func in (:chebyshevutransform,:ichebyshevutransform)
    @eval $func(x::AbstractVector{T};kind::Integer=1) where {T<:Integer} = $func(convert(Float64,x);kind=kind)
end




## points

function chebyshevpoints(::Type{T}, n::Integer; kind::Int=1) where T<:Number
    if kind == 1
        T[sinpi((n-2k-one(T))/2n) for k=0:n-1]
    elseif kind == 2
        if n == 1
            zeros(T,1)
        else
            T[cospi(k/(n-one(T))) for k=0:n-1]
        end
    end
end
chebyshevpoints(n::Integer; kind::Int=1) = chebyshevpoints(Float64, n; kind=kind)


# sin(nθ) coefficients to values at Clenshaw-Curtis nodes except ±1
#
# struct DSTPlan{T,kind,inplace,P} <: Plan{T}
#     plan::P
# end
#
# DSTPlan{k,inp}(plan) where {k,inp} =
#     DSTPlan{eltype(plan),k,inp,typeof(plan)}(plan)
#
#
# plan_DSTI!(x) = length(x) > 0 ? DSTPlan{1,true}(FFTW.plan_r2r!(x, FFTW.RODFT00)) :
#                                 fill(one(T),1,length(x))
#
# function *(P::DSTPlan{T,1}, x::AbstractArray) where {T}
#     x = P.plan*x
#     rmul!(x,half(T))
# end

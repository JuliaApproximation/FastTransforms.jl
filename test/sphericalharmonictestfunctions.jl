function sphrand{T}(::Type{T}, m, n)
    A = zeros(T, m, 2n-1)
    for i = 1:m
        A[i,1] = rand(T)
    end
    for j = 1:n
        for i = 1:m-j
            A[i,2j] = rand(T)
            A[i,2j+1] = rand(T)
        end
    end
    A
end

function sphrandn{T}(::Type{T}, m, n)
    A = zeros(T, m, 2n-1)
    for i = 1:m
        A[i,1] = randn(T)
    end
    for j = 1:n
        for i = 1:m-j
            A[i,2j] = randn(T)
            A[i,2j+1] = randn(T)
        end
    end
    A
end

function normalizecolumns!(A::AbstractMatrix)
    m, n = size(A)
    @inbounds for j = 1:n
        nrm = zero(eltype(A))
        for i = 1:m
            nrm += abs2(A[i,j])
        end
        nrm = sqrt(nrm)
        for i = 1:m
            A[i,j] /= nrm
        end
    end
    A
end

function maxcolnorm(A::AbstractMatrix)
    m, n = size(A)
    nrm = zeros(n)
    @inbounds for j = 1:n
        nrm[j] = 0
        for i = 1:m
            nrm[j] += abs2(A[i,j])
        end
        nrm[j] = sqrt(nrm[j])
    end
    norm(nrm, Inf)
end

function sphevaluatepi(θ::Number,L::Integer,M::Integer)
    ret = one(θ)/sqrt(two(θ))
    if M < 0 M = -M end
    c, s = cospi(θ), sinpi(θ)
    for m = 1:M
        ret *= sqrt((m+half(θ))/m)*s
    end
    tc = two(c)*c

    if L == M
        return ret
    elseif L == M+1
        return sqrt(two(θ)*M+3)*c*ret
    else
        temp = ret
        ret *= sqrt(two(θ)*M+3)*c
        for l = M+1:L-1
            ret, temp = (sqrt(l+half(θ))*tc*ret - sqrt((l-M)*(l+M)/(l-half(θ)))*temp)/sqrt((l-M+1)*(l+M+1)/(l+3half(θ))), ret
        end
        return ret
    end
end

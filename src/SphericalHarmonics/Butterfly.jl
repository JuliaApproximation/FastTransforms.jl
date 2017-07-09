immutable Butterfly{T} <: Factorization{T}
    columns::Vector{Matrix{T}}
    factors::Vector{Vector{IDPackedV{T}}}
    permutations::Vector{Vector{ColumnPermutation}}
    indices::Vector{Vector{Int}}
    temp1::Vector{T}
    temp2::Vector{T}
    temp3::Vector{T}
    temp4::Vector{T}
end

function size(B::Butterfly, dim::Integer)
    if dim == 1
        m = 0
        for i = 1:length(B.columns)
            m += size(B.columns[i], 1)
        end
        return m
    elseif dim == 2
        n = 0
        for j = 1:length(B.factors[1])
            n += size(B.factors[1][j], 2)
        end
        return n
    else
        return 1
    end
end

size(B::Butterfly) = size(B, 1), size(B, 2)

function Butterfly{T}(A::AbstractMatrix{T}, L::Int; isorthogonal::Bool = false, opts...)
    m, n = size(A)
    tL = 2^L

    LRAOpts = LRAOptions(T; opts...)
    LRAOpts.rtol = eps(real(T))*max(m, n)

    columns = Vector{Matrix{T}}(tL)
    factors = Vector{Vector{IDPackedV{T}}}(L+1)
    permutations = Vector{Vector{ColumnPermutation}}(L+1)
    indices = Vector{Vector{Int}}(L+1)
    cs = Vector{Vector{Vector{Int}}}(L+1)

    factors[1] = Vector{IDPackedV{T}}(tL)
    permutations[1] = Vector{ColumnPermutation}(tL)
    indices[1] = Vector{Int}(tL+1)
    cs[1] = Vector{Vector{Int}}(tL)

    ninds = linspace(1, n+1, tL+1)
    indices[1][1] = 1
    for j = 1:tL
        nl = round(Int, ninds[j])
        nu = round(Int, ninds[j+1]) - 1
        nd = nu-nl+1
        if isorthogonal
            factors[1][j] = IDPackedV{T}(collect(1:nd),Int[],Array{T}(nd,0))
        else
            factors[1][j] = idfact!(A[:,nl:nu], LRAOpts)
        end
        permutations[1][j] = factors[1][j][:P]
        indices[1][j+1] = indices[1][j] + size(factors[1][j], 1)
        cs[1][j] = factors[1][j].sk+nl-1
    end

    ii, jj = 2, (tL>>1)
    for l = 2:L+1
        factors[l] = Vector{IDPackedV{T}}(tL)
        permutations[l] = Vector{ColumnPermutation}(tL)
        indices[l] = Vector{Int}(tL+1)
        cs[l] = Vector{Vector{Int}}(tL)

        ctr = 0
        minds = linspace(1, m+1, ii+1)
        indices[l][1] = 1
        for i = 1:ii
            shft = 2jj*div(ctr,2jj)
            ml = round(Int, minds[i])
            mu = round(Int, minds[i+1]) - 1
            for j = 1:jj
                cols = vcat(cs[l-1][2j-1+shft],cs[l-1][2j+shft])
                lc = length(cols)
                Av = A[ml:mu,cols]
                if maximum(abs, Av) < realmin(real(T))/eps(real(T))
                    factors[l][j+ctr] = IDPackedV{T}(Int[], collect(1:lc), Array{T}(0,lc))
                else
                    LRAOpts.rtol = eps(real(T))*max(mu-ml+1, lc)
                    factors[l][j+ctr] = idfact!(Av, LRAOpts)
                end
                permutations[l][j+ctr] = factors[l][j+ctr][:P]
                indices[l][j+ctr+1] = indices[l][j+ctr] + size(factors[l][j+ctr], 1)
                cs[l][j+ctr] = cols[factors[l][j+ctr].sk]
            end
            ctr += jj
        end
        ii <<= 1
        jj >>= 1
    end

    minds = linspace(1, m+1, tL+1)
    for i = 1:tL
        ml = round(Int, minds[i])
        mu = round(Int, minds[i+1]) - 1
        columns[i] = A[ml:mu, cs[L+1][i]]
    end

    kk = sumkmax(indices)

    Butterfly(columns, factors, permutations, indices, zeros(T, kk), zeros(T, kk), zeros(T, kk), zeros(T, kk))
end

function sumkmax(indices::Vector{Vector{Int}})
    ret = 0
    @inbounds for j = 1:length(indices)
        ret = max(ret, indices[j][end])
    end
    ret
end

#### Helper

function rowperm!(fwd::Bool, x::StridedVecOrMat, p::Vector{Int}, jstart::Int)
    n = length(p)
    jshift = jstart-1
    scale!(p, -1)
    @inbounds if (fwd)
        for i = 1:n
            p[i] > 0 && continue
            j    =    i
            p[j] = -p[j]
            k    =  p[j]
            while p[k] < 0
                x[jshift+j], x[jshift+k] = x[jshift+k], x[jshift+j]
                j    =    k
                p[j] = -p[j]
                k    =  p[j]
            end
        end
    else
        for i = 1:n
            p[i] > 0 && continue
            p[i] = -p[i]
            j    =  p[i]
            while p[j] < 0
                x[jshift+i], x[jshift+j] = x[jshift+j], x[jshift+i]
                p[j] = -p[j]
                j    =  p[j]
            end
        end
    end
    x
end

function rowperm!(fwd::Bool, y::StridedVector, x::StridedVector, p::Vector{Int}, jstart::Int)
    n = length(p)
    jshift = jstart-1
    @inbounds if (fwd)
        @simd for i = 1:n
            y[jshift+i] = x[jshift+p[i]]
        end
    else
        @simd for i = 1:n
            y[jshift+p[i]] = x[jshift+i]
        end
    end
    y
end

## ColumnPermutation
A_mul_B!(A::ColPerm, B::StridedVecOrMat, jstart::Int) = rowperm!(false, B, A.p, jstart)
At_mul_B!(A::ColPerm, B::StridedVecOrMat, jstart::Int) = rowperm!(true, B, A.p, jstart)
Ac_mul_B!(A::ColPerm, B::StridedVecOrMat, jstart::Int) = At_mul_B!(A, B, jstart)

A_mul_B!(y::StridedVector, A::ColPerm, x::StridedVector, jstart::Int) = rowperm!(false, y, x, A.p, jstart)
At_mul_B!(y::StridedVector, A::ColPerm, x::StridedVector, jstart::Int) = rowperm!(true, y, x, A.p, jstart)
Ac_mul_B!(y::StridedVector, A::ColPerm, x::StridedVector, jstart::Int) = At_mul_B!(y, x, A, jstart)

# Fast A_mul_B!, At_mul_B!, and Ac_mul_B! for an ID. These overwrite the output.

function A_mul_B!{T}(y::AbstractVecOrMat{T}, A::IDPackedV{T}, P::ColumnPermutation, x::AbstractVecOrMat{T}, istart::Int, jstart::Int)
    k, n = size(A)
    At_mul_B!(P, x, jstart)
    copy!(y, istart, x, jstart, k)
    A_mul_B!(y, A.T, x, istart, jstart+k)
    A_mul_B!(P, x, jstart)
    y
end

function A_mul_B!{T}(y::AbstractVector{T}, A::IDPackedV{T}, P::ColumnPermutation, x::AbstractVector{T}, temp::AbstractVector{T}, istart::Int, jstart::Int)
    k, n = size(A)
    At_mul_B!(temp, P, x, jstart)
    copy!(y, istart, temp, jstart, k)
    A_mul_B!(y, A.T, temp, istart, jstart+k)
    y
end

for f! in (:At_mul_B!, :Ac_mul_B!)
    @eval begin
        function $f!{T}(y::AbstractVecOrMat{T}, A::IDPackedV{T}, P::ColumnPermutation, x::AbstractVecOrMat{T}, istart::Int, jstart::Int)
            k, n = size(A)
            copy!(y, istart, x, jstart, k)
            $f!(y, A.T, x, istart+k, jstart)
            A_mul_B!(P, y, istart)
            y
        end

        function $f!{T}(y::AbstractVector{T}, A::IDPackedV{T}, P::ColumnPermutation, x::AbstractVector{T}, temp::AbstractVector{T}, istart::Int, jstart::Int)
            k, n = size(A)
            copy!(temp, istart, x, jstart, k)
            $f!(temp, A.T, x, istart+k, jstart)
            A_mul_B!(y, P, temp, istart)
            y
        end
    end
end

### A_mul_B!, At_mul_B!, and  Ac_mul_B! for a Butterfly factorization.

Base.A_mul_B!{T}(u::Vector{T}, B::Butterfly{T}, b::Vector{T}) = A_mul_B_col_J!(u, B, b, 1)
Base.At_mul_B!{T}(u::Vector{T}, B::Butterfly{T}, b::Vector{T}) = At_mul_B_col_J!(u, B, b, 1)
Base.Ac_mul_B!{T}(u::Vector{T}, B::Butterfly{T}, b::Vector{T}) = Ac_mul_B_col_J!(u, B, b, 1)

function A_mul_B_col_J!{T}(u::VecOrMat{T}, B::Butterfly{T}, b::VecOrMat{T}, J::Int)
    L = length(B.factors) - 1
    tL = 2^L

    M = size(b, 1)

    COLSHIFT = M*(J-1)

    temp1 = B.temp1
    temp2 = B.temp2
    temp3 = B.temp3
    fill!(temp1, zero(T))
    fill!(temp2, zero(T))

    factors = B.factors[1]
    permutations = B.permutations[1]
    inds = B.indices[1]
    nu = 0
    for j = 1:tL
        nl = nu+1
        nu += size(factors[j], 2)
        A_mul_B!(temp1, factors[j], permutations[j], b, inds[j], nl+COLSHIFT)
    end

    ii, jj = 2, (tL>>1)
    for l = 2:L+1
        factors = B.factors[l]
        permutations = B.permutations[l]
        indsout = B.indices[l]
        indsin = B.indices[l-1]
        ctr = 0
        for i = 1:ii
            shft = 2jj*div(ctr,2jj)
            for j = 1:jj
                A_mul_B!(temp2, factors[j+ctr], permutations[j+ctr], temp1, temp3, indsout[j+ctr], indsin[2j+shft-1])
            end
            ctr += jj
        end
        temp1, temp2 = temp2, fill!(temp1, zero(T))
        ii <<= 1
        jj >>= 1
    end

    columns = B.columns
    inds = B.indices[L+1]
    mu = 0
    for i = 1:tL
        ml = mu+1
        mu += size(columns[i], 1)
        A_mul_B!(u, columns[i], temp1, ml+COLSHIFT, inds[i])
    end

    u
end

for f! in (:At_mul_B!,:Ac_mul_B!)
    f_col_J! = parse(string(f!)[1:end-1]*"_col_J!")
    @eval begin
        function $f_col_J!{T}(u::VecOrMat{T}, B::Butterfly{T}, b::VecOrMat{T}, J::Int)
            L = length(B.factors) - 1
            tL = 2^L

            M = size(b, 1)

            COLSHIFT = M*(J-1)

            temp1 = B.temp1
            temp2 = B.temp2
            temp3 = B.temp3
            temp4 = B.temp4
            fill!(temp1, zero(T))
            fill!(temp2, zero(T))
            fill!(temp3, zero(T))

            columns = B.columns
            inds = B.indices[L+1]
            mu = 0
            for i = 1:tL
                ml = mu+1
                mu += size(columns[i], 1)
                $f!(temp1, columns[i], b, inds[i], ml+COLSHIFT)
            end

            ii, jj = tL, 1
            for l = L+1:-1:2
                factors = B.factors[l]
                permutations = B.permutations[l]
                indsout = B.indices[l-1]
                indsin = B.indices[l]
                ctr = 0
                for i = 1:ii
                    shft = 2jj*div(ctr,2jj)
                    fill!(temp4, zero(T))
                    for j = 1:jj
                        $f!(temp3, factors[j+ctr], permutations[j+ctr], temp1, temp4, indsout[2j+shft-1], indsin[j+ctr])
                        addtemp3totemp2!(temp2, temp3, indsout[2j+shft-1], indsout[2j+shft+1]-1)
                    end
                    ctr += jj
                end
                temp1, temp2 = temp2, fill!(temp1, zero(T))
                ii >>= 1
                jj <<= 1
            end

            factors = B.factors[1]
            permutations = B.permutations[1]
            inds = B.indices[1]
            nu = 0
            for j = 1:tL
                nl = nu+1
                nu += size(factors[j], 2)
                $f!(u, factors[j], permutations[j], temp1, nl+COLSHIFT, inds[j])
            end

            u
        end
    end
end

function addtemp3totemp2!(temp2::Vector, temp3::Vector, i1::Int, i2::Int)
    z = zero(eltype(temp3))
    @inbounds @simd for i = i1:i2
        temp2[i] += temp3[i]
        temp3[i] = z
    end
    temp2
end


function allranks(B::Butterfly)
    L = length(B.factors)-1
    tL = 2^L
    ret = zeros(Int, tL, L+1)
    @inbounds for l = 1:L+1
        for j = 1:tL
            ret[j,l] = size(B.factors[l][j], 1)
        end
    end

    ret
end

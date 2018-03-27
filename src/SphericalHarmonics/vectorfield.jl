function A_mul_B_vf!(P::RotationPlan, A::AbstractMatrix)
    N, M = size(A)
    snm = P.snm
    cnm = P.cnm
    @stepthreads for m = M÷2-1:-1:2
        @inbounds for j = m:-2:2
            for l = N-j:-1:1
                s = snm[l+(j-2)*(2*N+3-j)÷2]
                c = cnm[l+(j-2)*(2*N+3-j)÷2]
                a1 = A[l+N*(2*m+1)]
                a2 = A[l+2+N*(2*m+1)]
                a3 = A[l+N*(2*m+2)]
                a4 = A[l+2+N*(2*m+2)]
                A[l+N*(2*m+1)] = c*a1 + s*a2
                A[l+2+N*(2*m+1)] = c*a2 - s*a1
                A[l+N*(2*m+2)] = c*a3 + s*a4
                A[l+2+N*(2*m+2)] = c*a4 - s*a3
            end
        end
    end
    A
end

function At_mul_B_vf!(P::RotationPlan, A::AbstractMatrix)
    N, M = size(A)
    snm = P.snm
    cnm = P.cnm
    @stepthreads for m = M÷2-1:-1:2
        @inbounds for j = reverse(m:-2:2)
            for l = 1:N-j
                s = snm[l+(j-2)*(2*N+3-j)÷2]
                c = cnm[l+(j-2)*(2*N+3-j)÷2]
                a1 = A[l+N*(2*m+1)]
                a2 = A[l+2+N*(2*m+1)]
                a3 = A[l+N*(2*m+2)]
                a4 = A[l+2+N*(2*m+2)]
                A[l+N*(2*m+1)] = c*a1 - s*a2
                A[l+2+N*(2*m+1)] = c*a2 + s*a1
                A[l+N*(2*m+2)] = c*a3 - s*a4
                A[l+2+N*(2*m+2)] = c*a4 + s*a3
            end
        end
    end
    A
end


function Base.A_mul_B!(Y1::Matrix, Y2::Matrix, SP::SlowSphericalHarmonicPlan, X1::Matrix, X2::Matrix)
    RP, p1, p2, B = SP.RP, SP.p1, SP.p2, SP.B
    copy!(B, X1)
    A_mul_B_vf!(RP, B)
    M, N = size(X1)
    A_mul_B_col_J!!(Y1, p2, B, 1)
    for J = 2:4:N
        A_mul_B_col_J!!(Y1, p1, B, J)
        J < N && A_mul_B_col_J!!(Y1, p1, B, J+1)
    end
    for J = 4:4:N
        A_mul_B_col_J!!(Y1, p2, B, J)
        J < N && A_mul_B_col_J!!(Y1, p2, B, J+1)
    end
    copy!(B, X2)
    A_mul_B_vf!(RP, B)
    M, N = size(X2)
    A_mul_B_col_J!!(Y2, p2, B, 1)
    for J = 2:4:N
        A_mul_B_col_J!!(Y2, p1, B, J)
        J < N && A_mul_B_col_J!!(Y2, p1, B, J+1)
    end
    for J = 4:4:N
        A_mul_B_col_J!!(Y2, p2, B, J)
        J < N && A_mul_B_col_J!!(Y2, p2, B, J+1)
    end
    Y1
end

function Base.At_mul_B!(Y1::Matrix, Y2::Matrix, SP::SlowSphericalHarmonicPlan, X1::Matrix, X2::Matrix)
    RP, p1inv, p2inv, B = SP.RP, SP.p1inv, SP.p2inv, SP.B
    copy!(B, X1)
    M, N = size(X1)
    A_mul_B_col_J!!(Y1, p2inv, B, 1)
    for J = 2:4:N
        A_mul_B_col_J!!(Y1, p1inv, B, J)
        J < N && A_mul_B_col_J!!(Y1, p1inv, B, J+1)
    end
    for J = 4:4:N
        A_mul_B_col_J!!(Y1, p2inv, B, J)
        J < N && A_mul_B_col_J!!(Y1, p2inv, B, J+1)
    end
    sph_zero_spurious_modes_vf!(At_mul_B_vf!(RP, Y1))
    copy!(B, X2)
    M, N = size(X2)
    A_mul_B_col_J!!(Y2, p2inv, B, 1)
    for J = 2:4:N
        A_mul_B_col_J!!(Y2, p1inv, B, J)
        J < N && A_mul_B_col_J!!(Y2, p1inv, B, J+1)
    end
    for J = 4:4:N
        A_mul_B_col_J!!(Y2, p2inv, B, J)
        J < N && A_mul_B_col_J!!(Y2, p2inv, B, J+1)
    end
    sph_zero_spurious_modes_vf!(At_mul_B_vf!(RP, Y2))
    Y1
end

Base.Ac_mul_B!(Y1::Matrix, Y2::Matrix, SP::SlowSphericalHarmonicPlan, X1::Matrix, X2::Matrix) = At_mul_B!(Y1, Y2, SP, X1, X2)


function Base.A_mul_B!(Y1::Matrix{T}, Y2::Matrix{T}, P::SynthesisPlan{T}, X1::Matrix{T}, X2::Matrix{T}) where T
    M, N = size(X1)

    # Column synthesis
    PCe = P.planθ[1]
    PCo = P.planθ[2]

    A_mul_B_col_J!(Y1, PCo, X1, 1)

    for J = 2:4:N
        X1[1,J] *= two(T)
        J < N && (X1[1,J+1] *= two(T))
        A_mul_B_col_J!(Y1, PCe, X1, J)
        J < N && A_mul_B_col_J!(Y1, PCe, X1, J+1)
        X1[1,J] *= half(T)
        J < N && (X1[1,J+1] *= half(T))
    end
    for J = 4:4:N
        A_mul_B_col_J!(Y1, PCo, X1, J)
        J < N && A_mul_B_col_J!(Y1, PCo, X1, J+1)
    end
    scale!(half(T), Y1)

    # Row synthesis
    scale!(inv(sqrt(π)), Y1)
    invsqrttwo = inv(sqrt(2))
    @inbounds for i = 1:M Y1[i] *= invsqrttwo end

    temp = P.temp
    planφ = P.planφ
    C = P.C
    for I = 1:M
        copy_row_I!(temp, Y1, I)
        row_synthesis!(planφ, C, temp)
        copy_row_I!(Y1, temp, I)
    end

    M, N = size(X2)

    # Column synthesis
    PCe = P.planθ[1]
    PCo = P.planθ[2]

    A_mul_B_col_J!(Y2, PCo, X2, 1)

    for J = 2:4:N
        X2[1,J] *= two(T)
        J < N && (X2[1,J+1] *= two(T))
        A_mul_B_col_J!(Y2, PCe, X2, J)
        J < N && A_mul_B_col_J!(Y2, PCe, X2, J+1)
        X2[1,J] *= half(T)
        J < N && (X2[1,J+1] *= half(T))
    end
    for J = 4:4:N
        A_mul_B_col_J!(Y2, PCo, X2, J)
        J < N && A_mul_B_col_J!(Y2, PCo, X2, J+1)
    end
    scale!(half(T), Y2)

    # Row synthesis
    scale!(inv(sqrt(π)), Y2)
    invsqrttwo = inv(sqrt(2))
    @inbounds for i = 1:M Y2[i] *= invsqrttwo end

    temp = P.temp
    planφ = P.planφ
    C = P.C
    for I = 1:M
        copy_row_I!(temp, Y2, I)
        row_synthesis!(planφ, C, temp)
        copy_row_I!(Y2, temp, I)
    end
    Y1
end

function Base.A_mul_B!(Y1::Matrix{T}, Y2::Matrix{T}, P::AnalysisPlan{T}, X1::Matrix{T}, X2::Matrix{T}) where T
    M, N = size(X1)

    # Row analysis
    temp = P.temp
    planφ = P.planφ
    C = P.C
    for I = 1:M
        copy_row_I!(temp, X1, I)
        row_analysis!(planφ, C, temp)
        copy_row_I!(Y1, temp, I)
    end

    # Column analysis
    PCe = P.planθ[1]
    PCo = P.planθ[2]

    A_mul_B_col_J!(Y1, PCo, Y1, 1)
    for J = 2:4:N
        A_mul_B_col_J!(Y1, PCe, Y1, J)
        J < N && A_mul_B_col_J!(Y1, PCe, Y1, J+1)
        Y1[1,J] *= half(T)
        J < N && (Y1[1,J+1] *= half(T))
    end
    for J = 4:4:N
        A_mul_B_col_J!(Y1, PCo, Y1, J)
        J < N && A_mul_B_col_J!(Y1, PCo, Y1, J+1)
    end
    scale!(sqrt(π)*inv(T(M)), Y1)
    sqrttwo = sqrt(2)
    @inbounds for i = 1:M Y1[i] *= sqrttwo end

    M, N = size(X2)

    # Row analysis
    temp = P.temp
    planφ = P.planφ
    C = P.C
    for I = 1:M
        copy_row_I!(temp, X2, I)
        row_analysis!(planφ, C, temp)
        copy_row_I!(Y2, temp, I)
    end

    # Column analysis
    PCe = P.planθ[1]
    PCo = P.planθ[2]

    A_mul_B_col_J!(Y2, PCo, Y2, 1)
    for J = 2:4:N
        A_mul_B_col_J!(Y2, PCe, Y2, J)
        J < N && A_mul_B_col_J!(Y2, PCe, Y2, J+1)
        Y2[1,J] *= half(T)
        J < N && (Y2[1,J+1] *= half(T))
    end
    for J = 4:4:N
        A_mul_B_col_J!(Y2, PCo, Y2, J)
        J < N && A_mul_B_col_J!(Y2, PCo, Y2, J+1)
    end
    scale!(sqrt(π)*inv(T(M)), Y2)
    sqrttwo = sqrt(2)
    @inbounds for i = 1:M Y2[i] *= sqrttwo end

    Y1
end


function sph_zero_spurious_modes_vf!(A::AbstractMatrix)
    M, N = size(A)
    n = N÷2
    A[M, 1] = 0
    @inbounds for j = 2:n-1
        @simd for i = M-j+2:M
            A[i,2j] = 0
            A[i,2j+1] = 0
        end
    end
    @inbounds @simd for i = M-n+2:M
        A[i,2n] = 0
        2n < N && (A[i,2n+1] = 0)
    end
    A
end

function sphrandvf(::Type{T}, m::Int, n::Int) where T
    A = zeros(T, m, 2n-1)
    for i = 1:m-1
        A[i,1] = rand(T)
    end
    for j = 1:n-1
        for i = 1:m-j+1
            A[i,2j] = rand(T)
            A[i,2j+1] = rand(T)
        end
    end
    A
end

function sphrandnvf(::Type{T}, m::Int, n::Int) where T
    A = zeros(T, m, 2n-1)
    for i = 1:m-1
        A[i,1] = randn(T)
    end
    for j = 1:n-1
        for i = 1:m-j+1
            A[i,2j] = randn(T)
            A[i,2j+1] = randn(T)
        end
    end
    A
end

function sphonesvf(::Type{T}, m::Int, n::Int) where T
    A = zeros(T, m, 2n-1)
    for i = 1:m-1
        A[i,1] = one(T)
    end
    for j = 1:n-1
        for i = 1:m-j+1
            A[i,2j] = one(T)
            A[i,2j+1] = one(T)
        end
    end
    A
end

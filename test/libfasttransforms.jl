N = 2 .^(7:11)

for n in N
    A = sphrand(Float64, n, n)
    B = copy(A)

    @time P = plan_sph2fourier(n)
    @time PS = plan_sph_synthesis(size(A, 1), size(A, 2))
    @time PA = plan_sph_analysis(size(A, 1), size(A, 2))
    @time PSV = plan_sphv_synthesis(size(A, 1), size(A, 2))
    @time PAV = plan_sphv_analysis(size(A, 1), size(A, 2))

    @time sph2fourier!(P, B)
    @time sph_synthesis!(PS, B)
    @time sph_analysis!(PA, B)
    @time fourier2sph!(P, B)

    @show norm(A-B)/norm(A)

    @time sphv2fourier!(P, B)
    @time sphv_synthesis!(PS, B)
    @time sphv_analysis!(PA, B)
    @time fourier2sphv!(P, B)

    @show norm(A-B)/norm(A)

    println()
    println()
end

for n in N
    A = trirand(Float64, n, n)
    B = copy(A)

    α, β, γ = 0.0, 0.0, 0.0

    @time P = plan_tri2cheb(n, α, β, γ)
    @time PS = plan_tri_synthesis(size(A, 1), size(A, 2))
    @time PA = plan_tri_analysis(size(A, 1), size(A, 2))

    @time tri2cheb!(P, B)
    @time tri_synthesis!(PS, B)
    @time tri_analysis!(PA, B)
    @time cheb2tri!(P, B)

    @show norm(A-B)/norm(A)

    println()
    println()
end

function diskrand(::Type{T}, m::Int, n::Int) where T
    A = zeros(T, m, 4n-3)
    for i = 1:m
        A[i,1] = rand(T)
    end
    for j = 1:n-1
        for i = 1:m-j
            A[i,4j-2] = rand(T)
            A[i,4j-1] = rand(T)
            A[i,4j] = rand(T)
            A[i,4j+1] = rand(T)
        end
    end
    A
end


for n in N
    A = diskrand(Float64, n, n)
    B = copy(A)

    α, β, γ = 0.0, 0.0, 0.0

    @time P = plan_disk2cxf(n)
    @time PS = plan_disk_synthesis(size(A, 1), size(A, 2))
    @time PA = plan_disk_analysis(size(A, 1), size(A, 2))

    @time disk2cxf!(P, B)
    @time disk_synthesis!(PS, B)
    @time disk_analysis!(PA, B)
    @time cxf2disk!(P, B)

    @show norm(A-B)/norm(A)

    println()
    println()
end

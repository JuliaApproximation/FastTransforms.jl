using FastTransforms, ToeplitzMatrices

@testset "BigFloat TOeplitz" begin
    T = Toeplitz(BigFloat[1,2,3,4,5], BigFloat[1,6,7,8,0])
    @test T*ones(BigFloat,5) ≈ [22,24,19,16,15]

    let n = 512
        r = map(BigFloat,rand(n))
        T = Toeplitz(r,[r[1];map(BigFloat,rand(n-1))])
        @test T*ones(BigFloat,n) ≈ Matrix(T)*ones(BigFloat,n)

        T = TriangularToeplitz(BigFloat[1,2,3,4,5],:L)
        @test T*ones(BigFloat,5) ≈ Matrix(T)*ones(BigFloat,5)

        r = map(BigFloat,rand(n))
        T = TriangularToeplitz(r,:L)
        @test T*ones(BigFloat,n) ≈ Matrix(T)*ones(BigFloat,n)

        T = TriangularToeplitz(BigFloat[1,2,3,4,5],:U)
        @test T*ones(BigFloat,5) ≈ Matrix(T)*ones(BigFloat,5)

        r = map(BigFloat,rand(n))
        T = TriangularToeplitz(r,:U)
        @test T*ones(BigFloat,n) ≈ Matrix(T)*ones(BigFloat,n)
    end
end

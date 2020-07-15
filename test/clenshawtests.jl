using FastTransforms, Test
import FastTransforms: clenshaw, clenshaw!, forwardrecurrence!

@testset "clenshaw" begin
    @testset "Chebyshev" begin
        c = [1,2,3]
        cf = float(c)
        @test @inferred(clenshaw(c,1)) ≡ 1 + 2 + 3
        @test @inferred(clenshaw(c,0)) ≡ 1 + 0 - 3
        @test @inferred(clenshaw(c,0.1)) == 1 + 2*0.1 + 3*cos(2acos(0.1))
        @test @inferred(clenshaw(c,[-1,0,1])) == clenshaw!(c,[-1,0,1]) == [2,-2,6]
        @test clenshaw(c,[-1,0,1]) isa Vector{Int}
        @test @inferred(clenshaw(Float64[],1)) ≡ 0.0

        x = [1,0,0.1]
        @test @inferred(clenshaw(c,x)) ≈ @inferred(clenshaw!(c,copy(x))) ≈ 
            @inferred(clenshaw!(c,x,similar(x))) ≈
            @inferred(clenshaw(cf,x)) ≈ @inferred(clenshaw!(cf,copy(x))) ≈ 
            @inferred(clenshaw!(cf,x,similar(x))) ≈ [6,-2,-1.74]
    end

    @testset "general" begin
        @testset "forwardrecurrence!" begin
            N = 5
            A, B, C = [1; fill(2,N-2)], fill(0,N-1), fill(1,N-1)
            cf, Af, Bf, Cf = float(c), float(A), float(B), float(C)
            @test forwardrecurrence!(Vector{Int}(undef,N), A, B, C, 1) == ones(Int,N)
            @test forwardrecurrence!(Vector{Int}(undef,N), A, B, C, -1) == (-1) .^ (0:N-1)
            @test forwardrecurrence!(Vector{Float64}(undef,N), A, B, C, 0.1) ≈ cos.((0:N-1) .* acos(0.1))
        end

        @testset "Chebyshev-as-general" begin
            c, A, B, C = [1,2,3], [1,2,2], fill(0,3), fill(1,3)
            cf, Af, Bf, Cf = float(c), float(A), float(B), float(C)
            @test @inferred(clenshaw(c, A, B, C, 1)) ≡ 6
            @test @inferred(clenshaw(c, A, B, C, 0.1)) ≡ -1.74
            @test @inferred(clenshaw([1,2,3], A, B, C, [-1,0,1])) == clenshaw!([1,2,3],A, B, C, [-1,0,1]) == [2,-2,6]
            @test clenshaw(c, A, B, C, [-1,0,1]) isa Vector{Int}
            @test @inferred(clenshaw(Float64[], A, B, C, 1)) ≡ 0.0

            x = [1,0,0.1]
            @test @inferred(clenshaw(c,A,B,C,x)) ≈ @inferred(clenshaw!(c,A,B,C,copy(x))) ≈ 
                @inferred(clenshaw!(c,A,B,C,x,one.(x),similar(x))) ≈
                @inferred(clenshaw!(cf,Af,Bf,Cf,x,one.(x),similar(x))) ≈
                @inferred(clenshaw([1.,2,3],A,B,C,x)) ≈ 
                @inferred(clenshaw!([1.,2,3],A,B,C,copy(x))) ≈ [6,-2,-1.74]
        end
    end
end
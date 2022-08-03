using FastTransforms, FillArrays, Test
import FastTransforms: clenshaw, clenshaw!, forwardrecurrence!, forwardrecurrence

@testset "clenshaw" begin
    @testset "Chebyshev T" begin
        for elty in (Float64, Float32)
            c = [1,2,3]
            cf = elty.(c)
            @test @inferred(clenshaw(c,1)) ≡ 1 + 2 + 3
            @test @inferred(clenshaw(c,0)) ≡ 1 + 0 - 3
            @test @inferred(clenshaw(c,0.1)) == 1 + 2*0.1 + 3*cos(2acos(0.1))
            @test @inferred(clenshaw(c,[-1,0,1])) == clenshaw!(c,[-1,0,1]) == [2,-2,6]
            @test clenshaw(c,[-1,0,1]) isa Vector{Int}
            @test @inferred(clenshaw(elty[],1)) ≡ zero(elty)

            x = elty[1,0,0.1]
            @test @inferred(clenshaw(c,x)) ≈ @inferred(clenshaw!(c,copy(x))) ≈
                @inferred(clenshaw!(c,x,similar(x))) ≈
                @inferred(clenshaw(cf,x)) ≈ @inferred(clenshaw!(cf,copy(x))) ≈
                @inferred(clenshaw!(cf,x,similar(x))) ≈ elty[6,-2,-1.74]

            @testset "Strided" begin
                cv = view(cf,:)
                xv = view(x,:)
                @test clenshaw!(cv, xv, similar(xv)) == clenshaw!(cf,x,similar(x))

                cv2 = view(cf,1:2:3)
                @test clenshaw!(cv2, xv, similar(xv)) == clenshaw([1,3], x)

                # modifies x and xv
                @test clenshaw!(cv2, xv) == xv == x == clenshaw([1,3], elty[1,0,0.1])
            end

            @testset "matrix coefficients" begin
                c = [1 2; 3 4; 5 6]
                @test clenshaw(c,0.1) ≈ [clenshaw(c[:,1],0.1), clenshaw(c[:,2],0.1)]
                @test clenshaw(c,[0.1,0.2]) ≈ [clenshaw(c[:,1], 0.1) clenshaw(c[:,2], 0.1); clenshaw(c[:,1], 0.2) clenshaw(c[:,2], 0.2)]
            end
        end
    end

    @testset "Chebyshev U" begin
        N = 5
        A, B, C = Fill(2,N-1), Zeros{Int}(N-1), Ones{Int}(N)
        @testset "forwardrecurrence!" begin
            @test @inferred(forwardrecurrence(N, A, B, C, 1)) == @inferred(forwardrecurrence!(Vector{Int}(undef,N), A, B, C, 1)) == 1:N
            @test forwardrecurrence!(Vector{Int}(undef,N), A, B, C, -1) == (-1) .^ (0:N-1) .* (1:N)
            @test forwardrecurrence(N, A, B, C, 0.1) ≈ forwardrecurrence!(Vector{Float64}(undef,N), A, B, C, 0.1) ≈
                    sin.((1:N) .* acos(0.1)) ./ sqrt(1-0.1^2)
        end

        c = [1,2,3]
        @test c'forwardrecurrence(3, A, B, C, 0.1) ≈ clenshaw([1,2,3], A, B, C, 0.1) ≈
            1 + (2sin(2acos(0.1)) + 3sin(3acos(0.1)))/sqrt(1-0.1^2)

        @testset "matrix coefficients" begin
            c = [1 2; 3 4; 5 6]
            @test clenshaw(c,A,B,C,0.1) ≈ [clenshaw(c[:,1],A,B,C,0.1), clenshaw(c[:,2],A,B,C,0.1)]
            @test clenshaw(c,A,B,C,[0.1,0.2]) ≈ [clenshaw(c[:,1], A,B,C,0.1) clenshaw(c[:,2], A,B,C,0.1); clenshaw(c[:,1], A,B,C,0.2) clenshaw(c[:,2], A,B,C,0.2)]
        end
    end

    @testset "Chebyshev-as-general" begin
        @testset "forwardrecurrence!" begin
            N = 5
            A, B, C = [1; fill(2,N-2)], fill(0,N-1), fill(1,N)
            Af, Bf, Cf = float(A), float(B), float(C)
            @test forwardrecurrence(N, A, B, C, 1) == forwardrecurrence!(Vector{Int}(undef,N), A, B, C, 1) == ones(Int,N)
            @test forwardrecurrence!(Vector{Int}(undef,N), A, B, C, -1) == (-1) .^ (0:N-1)
            @test forwardrecurrence(N, A, B, C, 0.1) ≈ forwardrecurrence!(Vector{Float64}(undef,N), A, B, C, 0.1) ≈ cos.((0:N-1) .* acos(0.1))
        end

        c, A, B, C = [1,2,3], [1,2,2], fill(0,3), fill(1,4)
        cf, Af, Bf, Cf = float(c), float(A), float(B), float(C)
        @test @inferred(clenshaw(c, A, B, C, 1)) ≡ 6
        @test @inferred(clenshaw(c, A, B, C, 0.1)) ≡ -1.74
        @test @inferred(clenshaw([1,2,3], A, B, C, [-1,0,1])) == clenshaw!([1,2,3],A, B, C, [-1,0,1]) == [2,-2,6]
        @test clenshaw(c, A, B, C, [-1,0,1]) isa Vector{Int}
        @test @inferred(clenshaw(Float64[], A, B, C, 1)) ≡ 0.0

        x = [1,0,0.1]
        @test @inferred(clenshaw(c, A, B, C, x)) ≈ @inferred(clenshaw!(c, A, B, C, copy(x))) ≈
            @inferred(clenshaw!(c, A, B, C, x, one.(x), similar(x))) ≈
            @inferred(clenshaw!(cf, Af, Bf, Cf, x, one.(x),similar(x))) ≈
            @inferred(clenshaw([1.,2,3], A, B, C, x)) ≈
            @inferred(clenshaw!([1.,2,3], A, B, C, copy(x))) ≈ [6,-2,-1.74]
    end

    @testset "Legendre" begin
        @testset "Float64" begin
            N = 5
            n = 0:N-1
            A = (2n .+ 1) ./ (n .+ 1)
            B = zeros(N)
            C = n ./ (n .+ 1)
            v_1 = forwardrecurrence(N, A, B, C, 1)
            v_f = forwardrecurrence(N, A, B, C, 0.1)
            @test v_1 ≈ ones(N)
            @test forwardrecurrence(N, A, B, C, -1) ≈ (-1) .^ (0:N-1)
            @test v_f ≈ [1,0.1,-0.485,-0.1475,0.3379375]

            n = 0:N # need extra entry for C in Clenshaw
            C = n ./ (n .+ 1)
            for j = 1:N
                c = [zeros(j-1); 1]
                @test clenshaw(c, A, B, C, 1) ≈ v_1[j] # Julia code
                @test clenshaw(c, A, B, C, 0.1) ≈  v_f[j] # Julia code
                @test clenshaw!(c, A, B, C, [1.0,0.1], [1.0,1.0], [0.0,0.0])  ≈ [v_1[j],v_f[j]] # libfasttransforms
            end
        end

        @testset "BigFloat" begin
            N = 5
            n = BigFloat(0):N-1
            A = (2n .+ 1) ./ (n .+ 1)
            B = zeros(N)
            C = n ./ (n .+ 1)
            @test forwardrecurrence(N, A, B, C, parse(BigFloat,"0.1")) ≈ [1,big"0.1",big"-0.485",big"-0.1475",big"0.3379375"]
        end
    end

    @testset "Int" begin
        N = 10; A = 1:10; B = 2:11; C = range(3; step=2, length=N+1)
        v_i = forwardrecurrence(N, A, B, C, 1)
        v_f = forwardrecurrence(N, A, B, C, 0.1)
        @test v_i isa Vector{Int}
        @test v_f isa Vector{Float64}

        j = 3
        @test clenshaw([zeros(Int,j-1); 1; zeros(Int,N-j)], A, B, C, 1) == v_i[j]
    end

    @testset "Zeros diagonal" begin
        N = 10; A = randn(N); B = Zeros{Int}(N); C = randn(N+1)
        @test forwardrecurrence(N, A, B, C, 0.1) == forwardrecurrence(N, A, Vector(B), C, 0.1)
        c = randn(N)
        @test clenshaw(c, A, B, C, 0.1) == clenshaw(c, A, Vector(B), C, 0.1)
    end
end

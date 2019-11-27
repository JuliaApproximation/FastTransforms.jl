using FastTransforms, FastGaussQuadrature, Test

hermitepoints(n) = FastGaussQuadrature.unweightedgausshermite( n )[1]

@testset "Hermite" begin
    @test hermitepoints(1) == [0.0]
    @test hermitepoints(100_000)[end] ≈ 446.9720305443094

    @test weightedhermitetransform([1.0]) == [1.0]
    @test weightedhermitetransform(exp.(-hermitepoints(2).^2/2)) ≈ [1.0,0.0]
    @test weightedhermitetransform(exp.(-hermitepoints(3).^2/2)) ≈ [1.0,0.0,0.0]
    @test weightedhermitetransform(exp.(-hermitepoints(1000).^2/2)) ≈ [1.0; zeros(999)]
    @test weightedhermitetransform(exp.(-hermitepoints(3000).^2/2)) ≈ [1.0; zeros(2999)]

    for n in (1, 5,100)
        x = randn(n)
        @test iweightedhermitetransform(weightedhermitetransform(x)) ≈ x
        @test weightedhermitetransform(iweightedhermitetransform(x)) ≈ x
    end

    x = hermitepoints(100)
    @test iweightedhermitetransform([0.0; 1.0; zeros(98)]) ≈ (exp.(-x.^2 ./ 2) .* 2x/sqrt(2))
    @test iweightedhermitetransform([0.0; 0; 1.0; zeros(97)]) ≈ (exp.(-x.^2 ./ 2) .* (4x.^2 .- 2)/(sqrt(2)*2^(2/2)))
    @test iweightedhermitetransform([0.0; 0; 0; 1.0; zeros(96)]) ≈ (exp.(-x.^2 ./ 2) .* (-12x + 8x.^3) / (sqrt(2*3)*2^(3/2)))
    @test iweightedhermitetransform([0.0; 0; 0; 0; 1.0; zeros(95)]) ≈ (exp.(-x.^2 ./ 2) .* (12 .- 48x.^2 .+ 16x.^4) / (sqrt(2*3*4)*2^(4/2)))
end

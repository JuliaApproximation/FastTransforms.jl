using FastTransforms
using Compat.Test
import FastTransforms.pochhammer

@testset "Pochhammer" begin
    @test pochhammer(2,3) == 24
    @test pochhammer(0.5,3) == 0.5*1.5*2.5
    @test pochhammer(0.5,0.5) == 1/sqrtpi
    @test pochhammer(0,1) == 0
    @test pochhammer(-1,2) == 0
    @test pochhammer(-5,3) == -60
    @test pochhammer(-1,-0.5) == 0
    @test pochhammer(-0.5,-0.5) == Inf
end

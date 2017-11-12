const BigFloats = Union{BigFloat,Complex{BigFloat}}

if VERSION < v"0.7-"
    import Base.FFTW: fft, fft!, rfft, irfft, ifft, conv, dct, idct, dct!, idct!,
                        plan_fft!, plan_ifft!, plan_dct!, plan_idct!,
                        plan_fft, plan_ifft, plan_rfft, plan_irfft, plan_dct, plan_idct
else
    import FFTW: fft, fft!, rfft, irfft, ifft, conv, dct, idct, dct!, idct!,
                        plan_fft!, plan_ifft!, plan_dct!, plan_idct!,
                        plan_fft, plan_ifft, plan_rfft, plan_irfft, plan_dct, plan_idct
end

# The following implements Bluestein's algorithm, following http://www.dsprelated.com/dspbooks/mdft/Bluestein_s_FFT_Algorithm.html
# To add more types, add them in the union of the function's signature.
function fft{T<:BigFloats}(x::Vector{T})
    n = length(x)
    ispow2(n) && return fft_pow2(x)
    ks = linspace(zero(real(T)),n-one(real(T)),n)
    Wks = exp.((-im).*convert(T,π).*ks.^2 ./ n)
    xq, wq = x.*Wks, conj([exp(-im*convert(T,π)*n);reverse(Wks);Wks[2:end]])
    return Wks.*conv(xq,wq)[n+1:2n]
end

function fft!{T<:BigFloats}(x::Vector{T})
    x[:] = fft(x)
    return x
end

# add rfft for BigFloat, by calling fft
#  this creates ToeplitzMatrices.rfft, so avoids changing rfft
rfft{T<:BigFloats}(v::Vector{T})=fft(v)[1:div(length(v),2)+1]
function irfft{T<:BigFloats}(v::Vector{T},n::Integer)
    @assert n==2length(v)-1
    r = Vector{Complex{BigFloat}}(n)
    r[1:length(v)]=v
    r[length(v)+1:end]=reverse(conj(v[2:end]))
    real(ifft(r))
end

ifft{T<:BigFloats}(x::Vector{T}) = conj!(fft(conj(x)))/length(x)
function ifft!{T<:BigFloats}(x::Vector{T})
    x[:] = ifft(x)
    return x
end

function conv{T<:BigFloats}(u::StridedVector{T}, v::StridedVector{T})
    nu,nv = length(u),length(v)
    n = nu + nv - 1
    np2 = nextpow2(n)
    append!(u,zeros(T,np2-nu)),append!(v,zeros(T,np2-nv))
    y = ifft_pow2(fft_pow2(u).*fft_pow2(v))
    #TODO This would not handle Dual/ComplexDual numbers correctly
    y = T<:Real ? real(y[1:n]) : y[1:n]
end

# This is a Cooley-Tukey FFT algorithm inspired by many widely available algorithms including:
# c_radix2.c in the GNU Scientific Library and four1 in the Numerical Recipes in C.
# However, the trigonometric recurrence is improved for greater efficiency.
# The algorithm starts with bit-reversal, then divides and conquers in-place.
function fft_pow2!{T<:BigFloat}(x::Vector{T})
    n,big2=length(x),2one(T)
    nn,j=n÷2,1
    for i=1:2:n-1
        if j>i
            x[j], x[i] = x[i], x[j]
            x[j+1], x[i+1] = x[i+1], x[j+1]
        end
        m = nn
        while m ≥ 2 && j > m
            j -= m
            m = m÷2
        end
        j += m
    end
    logn = 2
    while logn < n
        θ=-big2/logn
        wtemp = sinpi(θ/2)
        wpr, wpi = -2wtemp^2, sinpi(θ)
        wr, wi = one(T), zero(T)
        for m=1:2:logn-1
            for i=m:2logn:n
                j=i+logn
                mixr, mixi = wr*x[j]-wi*x[j+1], wr*x[j+1]+wi*x[j]
                x[j], x[j+1] = x[i]-mixr, x[i+1]-mixi
                x[i], x[i+1] = x[i]+mixr, x[i+1]+mixi
            end
            wr = (wtemp=wr)*wpr-wi*wpi+wr
            wi = wi*wpr+wtemp*wpi+wi
        end
        logn = logn << 1
    end
    return x
end

function fft_pow2{T<:BigFloat}(x::Vector{Complex{T}})
    y = interlace(real(x),imag(x))
    fft_pow2!(y)
    return complex.(y[1:2:end],y[2:2:end])
end
fft_pow2{T<:BigFloat}(x::Vector{T}) = fft_pow2(complex(x))

function ifft_pow2{T<:BigFloat}(x::Vector{Complex{T}})
    y = interlace(real(x),-imag(x))
    fft_pow2!(y)
    return complex.(y[1:2:end],-y[2:2:end])/length(x)
end


function dct(a::AbstractArray{Complex{BigFloat}})
	N = big(length(a))
    c = fft([a; flipdim(a,1)])
    d = c[1:N]
    d .*= exp.((-im*big(pi)).*(0:N-1)./(2*N))
    d[1] = d[1] / sqrt(big(2))
    scale!(inv(sqrt(2N)), d)
end

dct(a::AbstractArray{BigFloat}) = real(dct(complex(a)))

function idct(a::AbstractArray{Complex{BigFloat}})
    N = big(length(a))
    b = a * sqrt(2*N)
    b[1] = b[1] * sqrt(big(2))
    b .*= exp.((im*big(pi)).*(0:N-1)./(2*N))
    b = [b; 0; conj(flipdim(b[2:end],1))]
    c = ifft(b)
    c[1:N]
end

idct(a::AbstractArray{BigFloat}) = real(idct(complex(a)))

dct!{T<:BigFloats}(a::AbstractArray{T}) = (b = dct(a); a[:] = b)
idct!{T<:BigFloats}(a::AbstractArray{T}) = (b = idct(a); a[:] = b)

# dummy plans
struct DummyFFTPlan{T,inplace} <: Plan{T} end
struct DummyiFFTPlan{T,inplace} <: Plan{T} end
struct DummyrFFTPlan{T,inplace} <: Plan{T} end
struct DummyirFFTPlan{T,inplace} <: Plan{T} end
struct DummyDCTPlan{T,inplace} <: Plan{T} end
struct DummyiDCTPlan{T,inplace} <: Plan{T} end

for (Plan,iPlan) in ((:DummyFFTPlan,:DummyiFFTPlan),
                     (:DummyrFFTPlan,:DummyirFFTPlan),
                     (:DummyDCTPlan,:DummyiDCTPlan))
   @eval begin
       Base.inv{T,inplace}(::$Plan{T,inplace})=$iPlan{T,inplace}()
       Base.inv{T,inplace}(::$iPlan{T,inplace})=$Plan{T,inplace}()
    end
end


for (Plan,ff,ff!) in ((:DummyFFTPlan,:fft,:fft!),
                      (:DummyiFFTPlan,:ifft,:ifft!),
                      (:DummyrFFTPlan,:rfft,:rfft!),
                      (:DummyirFFTPlan,:irfft,:irfft!),
                      (:DummyDCTPlan,:dct,:dct!),
                      (:DummyiDCTPlan,:idct,:idct!))
    @eval begin
        *{T,N}(p::$Plan{T,true}, x::StridedArray{T,N}) = $ff!(x)
        *{T,N}(p::$Plan{T,false}, x::StridedArray{T,N}) = $ff(x)
        function Base.A_mul_B!(C::StridedVector,p::$Plan,x::StridedVector)
            C[:]=$ff(x)
            C
        end
    end
end




plan_fft!{T<:BigFloats}(x::Vector{T}) = DummyFFTPlan{Complex{BigFloat},true}()
plan_ifft!{T<:BigFloats}(x::Vector{T}) = DummyiFFTPlan{Complex{BigFloat},true}()
#plan_rfft!{T<:BigFloats}(x::Vector{T}) = DummyrFFTPlan{Complex{BigFloat},true}()
#plan_irfft!{T<:BigFloats}(x::Vector{T},n::Integer) = DummyirFFTPlan{Complex{BigFloat},true}()
plan_dct!{T<:BigFloats}(x::Vector{T}) = DummyDCTPlan{T,true}()
plan_idct!{T<:BigFloats}(x::Vector{T}) = DummyiDCTPlan{T,true}()

plan_fft{T<:BigFloats}(x::Vector{T}) = DummyFFTPlan{Complex{BigFloat},false}()
plan_ifft{T<:BigFloats}(x::Vector{T}) = DummyiFFTPlan{Complex{BigFloat},false}()
plan_rfft{T<:BigFloats}(x::Vector{T}) = DummyrFFTPlan{Complex{BigFloat},false}()
plan_irfft{T<:BigFloats}(x::Vector{T},n::Integer) = DummyirFFTPlan{Complex{BigFloat},false}()
plan_dct{T<:BigFloats}(x::Vector{T}) = DummyDCTPlan{T,false}()
plan_idct{T<:BigFloats}(x::Vector{T}) = DummyiDCTPlan{T,false}()


function interlace{S<:Number,V<:Number}(a::Vector{S},b::Vector{V})
    na=length(a);nb=length(b)
    T=promote_type(S,V)
    if nb≥na
        ret=zeros(T,2nb)
        ret[1:2:1+2*(na-1)]=a
        ret[2:2:end]=b
        ret
    else
        ret=zeros(T,2na-1)
        ret[1:2:end]=a
        if !isempty(b)
            ret[2:2:2+2*(nb-1)]=b
        end
        ret
    end
end

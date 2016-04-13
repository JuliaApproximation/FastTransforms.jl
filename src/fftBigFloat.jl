typealias BigFloats Union{BigFloat,Complex{BigFloat}}

real(x...)=Base.real(x...)
real{T<:Real}(::Type{T})=T
real{T<:Real}(::Type{Complex{T}})=T

# The following implements Bluestein's algorithm, following http://www.dsprelated.com/dspbooks/mdft/Bluestein_s_FFT_Algorithm.html
# To add more types, add them in the union of the function's signature.
function Base.fft{T<:BigFloats}(x::Vector{T})
    n = length(x)
    ispow2(n) && return fft_pow2(x)
    ks = linspace(zero(real(T)),n-one(real(T)),n)
    Wks = exp(-im*convert(T,π)*ks.^2/n)
    xq,wq = x.*Wks,conj([exp(-im*convert(T,π)*n);reverse(Wks);Wks[2:end]])
    return Wks.*conv(xq,wq)[n+1:2n]
end

function Base.fft!{T<:BigFloats}(x::Vector{T})
    x[:] = fft(x)
    return x
end

# add rfft for BigFloat, by calling fft
#  this creates ToeplitzMatrices.rfft, so avoids changing Base.rfft
Base.rfft{T<:BigFloats}(v::Vector{T})=fft(v)[1:div(length(v),2)+1]
function Base.irfft{T<:BigFloats}(v::Vector{T},n::Integer)
    @assert n==2length(v)-1
    r=Array(Complex{BigFloat},n)
    r[1:length(v)]=v
    r[length(v)+1:end]=reverse(conj(v[2:end]))
    real(ifft(r))
end

Base.ifft{T<:BigFloats}(x::Vector{T}) = conj!(fft(conj(x)))/length(x)
function Base.ifft!{T<:BigFloats}(x::Vector{T})
    x[:] = ifft(x)
    return x
end

function Base.conv{T<:BigFloats}(u::StridedVector{T}, v::StridedVector{T})
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
    return complex(y[1:2:end],y[2:2:end])
end
fft_pow2{T<:BigFloat}(x::Vector{T}) = fft_pow2(complex(x))

function ifft_pow2{T<:BigFloat}(x::Vector{Complex{T}})
    y = interlace(real(x),-imag(x))
    fft_pow2!(y)
    return complex(y[1:2:end],-y[2:2:end])/length(x)
end


function Base.dct(a::AbstractArray{Complex{BigFloat}})
	N = big(length(a))
    c = fft([a; flipdim(a,1)])
    d = c[1:N] .* exp(-im*big(pi)*(0:N-1)/(2*N))
    d[1] = d[1] / sqrt(big(2))
    scale!(inv(sqrt(2N)), d)
end

Base.dct(a::AbstractArray{BigFloat}) = real(dct(complex(a)))

function Base.idct(a::AbstractArray{Complex{BigFloat}})
    N = big(length(a))
    b = a * sqrt(2*N)
    b[1] = b[1] * sqrt(big(2))
    b = b .* exp(im*big(pi)*(0:N-1)/(2*N))
    b = [b; 0; conj(flipdim(b[2:end],1))]
    c = ifft(b)
    c[1:N]
end

Base.idct(a::AbstractArray{BigFloat}) = real(idct(complex(a)))

Base.dct!{T<:BigFloats}(a::AbstractArray{T}) = (b = dct(a); a[:] = b)
Base.idct!{T<:BigFloats}(a::AbstractArray{T}) = (b = idct(a); a[:] = b)

# dummy plans
type DummyFFTPlan{T,inplace} <: Base.DFT.Plan{T} end
type DummyiFFTPlan{T,inplace} <: Base.DFT.Plan{T} end
type DummyrFFTPlan{T,inplace} <: Base.DFT.Plan{T} end
type DummyirFFTPlan{T,inplace} <: Base.DFT.Plan{T} end
type DummyDCTPlan{T,inplace} <: Base.DFT.Plan{T} end
type DummyiDCTPlan{T,inplace} <: Base.DFT.Plan{T} end

*{T,N}(p::DummyFFTPlan{T,true}, x::StridedArray{T,N})=fft!(x)
*{T,N}(p::DummyiFFTPlan{T,true}, x::StridedArray{T,N})=ifft!(x)
*{T,N}(p::DummyrFFTPlan{T,true}, x::StridedArray{T,N})=rfft!(x)
*{T,N}(p::DummyirFFTPlan{T,true}, x::StridedArray{T,N})=irfft!(x)
*{T,N}(p::DummyDCTPlan{T,true}, x::StridedArray{T,N})=dct!(x)
*{T,N}(p::DummyiDCTPlan{T,true}, x::StridedArray{T,N})=idct!(x)

*{T,N}(p::DummyFFTPlan{T,false}, x::StridedArray{T,N})=fft(x)
*{T,N}(p::DummyiFFTPlan{T,false}, x::StridedArray{T,N})=ifft(x)
*{T,N}(p::DummyrFFTPlan{T,false}, x::StridedArray{T,N})=rfft(x)
*{T,N}(p::DummyirFFTPlan{T,false}, x::StridedArray{T,N})=irfft(x)
*{T,N}(p::DummyDCTPlan{T,false}, x::StridedArray{T,N})=dct(x)
*{T,N}(p::DummyiDCTPlan{T,false}, x::StridedArray{T,N})=idct(x)

Base.plan_fft!{T<:BigFloats}(x::Vector{T}) = DummyFFTPlan{Complex{BigFloat},true}()
Base.plan_ifft!{T<:BigFloats}(x::Vector{T}) = DummyiFFTPlan{Complex{BigFloat},true}()
#Base.plan_rfft!{T<:BigFloats}(x::Vector{T}) = DummyrFFTPlan{Complex{BigFloat},true}()
#Base.plan_irfft!{T<:BigFloats}(x::Vector{T},n::Integer) = DummyirFFTPlan{Complex{BigFloat},true}()
Base.plan_dct!{T<:BigFloats}(x::Vector{T}) = DummyDCTPlan{T,true}()
Base.plan_idct!{T<:BigFloats}(x::Vector{T}) = DummyiDCTPlan{T,true}()

Base.plan_fft{T<:BigFloats}(x::Vector{T}) = DummyFFTPlan{Complex{BigFloat},false}()
Base.plan_ifft{T<:BigFloats}(x::Vector{T}) = DummyiFFTPlan{Complex{BigFloat},false}()
Base.plan_rfft{T<:BigFloats}(x::Vector{T}) = DummyrFFTPlan{Complex{BigFloat},false}()
Base.plan_irfft{T<:BigFloats}(x::Vector{T},n::Integer) = DummyirFFTPlan{Complex{BigFloat},false}()
Base.plan_dct{T<:BigFloats}(x::Vector{T}) = DummyDCTPlan{T,false}()
Base.plan_idct{T<:BigFloats}(x::Vector{T}) = DummyiDCTPlan{T,false}()


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

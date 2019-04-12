const AbstractFloats = Union{AbstractFloat,Complex{T} where T<:AbstractFloat}

if VERSION < v"0.7-"
    import Base.FFTW: fft, fft!, rfft, irfft, ifft, ifft!, conv, dct, idct, dct!, idct!,
                        plan_fft!, plan_ifft!, plan_dct!, plan_idct!,
                        plan_fft, plan_ifft, plan_rfft, plan_irfft, plan_dct, plan_idct
else
    import FFTW:	dct, dct!, idct, idct!,
					plan_fft!, plan_ifft!, plan_dct!, plan_idct!,
                    plan_fft, plan_ifft, plan_rfft, plan_irfft, plan_dct, plan_idct
    import DSP: conv
end

# The following implements Bluestein's algorithm, following http://www.dsprelated.com/dspbooks/mdft/Bluestein_s_FFT_Algorithm.html
# To add more types, add them in the union of the function's signature.

function generic_fft(x::Vector{T}) where T<:AbstractFloats
    n = length(x)
    ispow2(n) && return generic_fft_pow2(x)
    ks = range(zero(real(T)),stop=n-one(real(T)),length=n)
    Wks = exp.((-im).*convert(T,π).*ks.^2 ./ n)
    xq, wq = x.*Wks, conj([exp(-im*convert(T,π)*n);reverse(Wks);Wks[2:end]])
    return Wks.*conv(xq,wq)[n+1:2n]
end


function generic_fft!(x::Vector{T}) where T<:AbstractFloats
    x[:] = generic_fft(x)
    return x
end

# add rfft for AbstractFloat, by calling fft
#  this creates ToeplitzMatrices.rfft, so avoids changing rfft

generic_rfft(v::Vector{T}) where T<:AbstractFloats = generic_fft(v)[1:div(length(v),2)+1]

function generic_irfft(v::Vector{T},n::Integer) where T<:AbstractFloats
    @assert n==2length(v)-1
    r = Vector{Complex{real(T)}}(undef, n)
    r[1:length(v)]=v
    r[length(v)+1:end]=reverse(conj(v[2:end]))
    real(generic_ifft(r))
end

generic_ifft(x::Vector{T}) where {T<:AbstractFloats} = conj!(generic_fft(conj(x)))/length(x)
function generic_ifft!(x::Vector{T}) where T<:AbstractFloats
    x[:] = generic_ifft(x)
    return x
end

function conv(u::StridedVector{T}, v::StridedVector{T}) where T<:AbstractFloats
    nu,nv = length(u),length(v)
    n = nu + nv - 1
    np2 = nextpow(2,n)
    append!(u,zeros(T,np2-nu)),append!(v,zeros(T,np2-nv))
    y = generic_ifft_pow2(generic_fft_pow2(u).*generic_fft_pow2(v))
    #TODO This would not handle Dual/ComplexDual numbers correctly
    y = T<:Real ? real(y[1:n]) : y[1:n]
end

# This is a Cooley-Tukey FFT algorithm inspired by many widely available algorithms including:
# c_radix2.c in the GNU Scientific Library and four1 in the Numerical Recipes in C.
# However, the trigonometric recurrence is improved for greater efficiency.
# The algorithm starts with bit-reversal, then divides and conquers in-place.
function generic_fft_pow2!(x::Vector{T}) where T<:AbstractFloat
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

function generic_fft_pow2(x::Vector{Complex{T}}) where T<:AbstractFloat
    y = interlace(real(x),imag(x))
    generic_fft_pow2!(y)
    return complex.(y[1:2:end],y[2:2:end])
end
generic_fft_pow2(x::Vector{T}) where {T<:AbstractFloat} = generic_fft_pow2(complex(x))

function generic_ifft_pow2(x::Vector{Complex{T}}) where T<:AbstractFloat
    y = interlace(real(x),-imag(x))
    generic_fft_pow2!(y)
    return complex.(y[1:2:end],-y[2:2:end])/length(x)
end

function generic_dct(a::AbstractVector{Complex{T}}) where {T <: AbstractFloat}
	N = length(a)
	twoN = convert(T,2) * N
    c = generic_fft([a; flipdim(a,1)])
    d = c[1:N]
    d .*= exp.((-im*convert(T, pi)).*(0:N-1)./twoN)
    d[1] = d[1] / sqrt(convert(T, 2))
    lmul!(inv(sqrt(twoN)), d)
end

generic_dct(a::AbstractArray{T}) where {T <: AbstractFloat} = real(generic_dct(complex(a)))

function generic_idct(a::AbstractVector{Complex{T}}) where {T <: AbstractFloat}
	N = length(a)
	twoN = convert(T,2)*N
    b = a * sqrt(twoN)
    b[1] = b[1] * sqrt(convert(T,2))
    shift = exp.(-im * 2 * convert(T, pi) * (N - convert(T,1)/2) * (0:(2N-1)) / twoN)
    b = [b; 0; -flipdim(b[2:end],1)] .* shift
    c = ifft(b)
    flipdim(c[1:N],1)
end

generic_idct(a::AbstractArray{T}) where {T <: AbstractFloat} = real(generic_idct(complex(a)))

generic_dct!(a::AbstractArray{T}) where {T<:AbstractFloats} = (b = generic_dct(a); a[:] = b)
generic_idct!(a::AbstractArray{T}) where {T<:AbstractFloats} = (b = generic_idct(a); a[:] = b)

# These lines mimick the corresponding ones in FFTW/src/dct.jl, but with
# AbstractFloat rather than fftwNumber.
for f in (:dct, :dct!, :idct, :idct!)
    pf = Symbol("plan_", f)
    @eval begin
        $f(x::AbstractArray{<:AbstractFloats}) = $pf(x) * x
        $f(x::AbstractArray{<:AbstractFloats}, region) = $pf(x, region) * x
    end
end

# dummy plans
struct DummyFFTPlan{T,inplace} <: Plan{T} end
struct DummyiFFTPlan{T,inplace} <: Plan{T} end
struct DummyDCTPlan{T,inplace} <: Plan{T} end
struct DummyiDCTPlan{T,inplace} <: Plan{T} end
struct DummyrFFTPlan{T,inplace} <: Plan{T}
	n	::	Integer
end
struct DummyirFFTPlan{T,inplace} <: Plan{T}
	n	::	Integer
end

for (Plan,iPlan) in ((:DummyFFTPlan,:DummyiFFTPlan),
                     # (:DummyrFFTPlan,:DummyirFFTPlan),
                     (:DummyDCTPlan,:DummyiDCTPlan))
   @eval begin
       Base.inv(::$Plan{T,inplace}) where {T,inplace} = $iPlan{T,inplace}()
       Base.inv(::$iPlan{T,inplace}) where {T,inplace} = $Plan{T,inplace}()
    end
end

# Specific for rfft and irfft:
Base.inv(::DummyirFFTPlan{T,inplace}) where {T,inplace} = DummyrFFTPlan{T,Inplace}(p.n)
Base.inv(::DummyrFFTPlan{T,inplace}) where {T,inplace} = DummyirFFTPlan{T,Inplace}(p.n)


for (Plan,ff,ff!) in ((:DummyFFTPlan,:generic_fft,:generic_fft!),
                      (:DummyiFFTPlan,:generic_ifft,:generic_ifft!),
                      (:DummyrFFTPlan,:generic_rfft,:generic_rfft!),
#                      (:DummyirFFTPlan,:generic_irfft,:generic_irfft!),
                      (:DummyDCTPlan,:generic_dct,:generic_dct!),
                      (:DummyiDCTPlan,:generic_idct,:generic_idct!))
    @eval begin
        *(p::$Plan{T,true}, x::StridedArray{T,N}) where {T<:AbstractFloats,N} = $ff!(x)
        *(p::$Plan{T,false}, x::StridedArray{T,N}) where {T<:AbstractFloats,N} = $ff(x)
        function LAmul!(C::StridedVector, p::$Plan, x::StridedVector)
            C[:] = $ff(x)
            C
        end
    end
end

# Specific for irfft:
*(p::DummyirFFTPlan{T,true}, x::StridedArray{T,N}) where {T<:AbstractFloats,N} = generic_irfft!(x, p.n)
*(p::DummyirFFTPlan{T,false}, x::StridedArray{T,N}) where {T<:AbstractFloats,N} = generic_irfft(x, p.n)
function LAmul!(C::StridedVector, p::DummyirFFTPlan, x::StridedVector)
	C[:] = generic_irfft(x, p.n)
	C
end

# We override these for AbstractFloat, so that conversion from reals to
# complex numbers works for any AbstractFloat (instead of only BlasFloat's)
AbstractFFTs.complexfloat(x::StridedArray{Complex{<:AbstractFloat}}) = x
AbstractFFTs.realfloat(x::StridedArray{<:Real}) = x
# We override this one in order to avoid throwing an error that the type is
# unsupported (as defined in AbstractFFTs)
AbstractFFTs._fftfloat(::Type{T}) where {T <: AbstractFloat} = T

plan_fft!(x::StridedArray{T}, region) where {T <: AbstractFloats} = DummyFFTPlan{Complex{real(T)},true}()
plan_ifft!(x::StridedArray{T}, region) where {T <: AbstractFloats} = DummyiFFTPlan{Complex{real(T)},true}()

# plan_rfft!(x::StridedArray{T}) where {T <: AbstractFloat} = DummyrFFTPlan{Complex{real(T)},true}()
# plan_irfft!(x::StridedArray{T},n::Integer) where {T <: AbstractFloat} = DummyirFFTPlan{Complex{real(T)},true}()
plan_dct!(x::StridedArray{T}, region) where {T <: AbstractFloats} = DummyDCTPlan{T,true}()
plan_idct!(x::StridedArray{T}, region) where {T <: AbstractFloats} = DummyiDCTPlan{T,true}()

plan_fft(x::StridedArray{T}, region) where {T <: AbstractFloats} = DummyFFTPlan{Complex{real(T)},false}()
plan_ifft(x::StridedArray{T}, region) where {T <: AbstractFloats} = DummyiFFTPlan{Complex{real(T)},false}()
plan_rfft(x::StridedArray{T}, region) where {T <: AbstractFloats} = DummyrFFTPlan{Complex{real(T)},false}(length(x))
plan_irfft(x::StridedArray{T}, n::Integer, region) where {T <: AbstractFloats} = DummyirFFTPlan{Complex{real(T)},false}(n)
plan_dct(x::StridedArray{T}, region) where {T <: AbstractFloats} = DummyDCTPlan{T,false}()
plan_idct(x::StridedArray{T}, region) where {T <: AbstractFloats} = DummyiDCTPlan{T,false}()


function interlace(a::Vector{S},b::Vector{V}) where {S<:Number,V<:Number}
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

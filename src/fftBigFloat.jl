const AbstractFloats = Union{AbstractFloat,Complex{T} where T<:AbstractFloat}

# We use these type definitions for clarity
const RealFloats = T where T<:AbstractFloat
const ComplexFloats = Complex{T} where T<:AbstractFloat


# The following implements Bluestein's algorithm, following http://www.dsprelated.com/dspbooks/mdft/Bluestein_s_FFT_Algorithm.html
# To add more types, add them in the union of the function's signature.

function generic_fft(x::StridedVector{T}, region::Integer) where T<:AbstractFloats
    region == 1 && (ret = generic_fft(x))
    ret
end

function generic_fft!(x::StridedVector{T}, region::Integer) where T<:AbstractFloats
    region == 1 && (x[:] .= generic_fft(x))
    x
end

function generic_fft(x::StridedVector{T}, region::UnitRange{I}) where {T<:AbstractFloats, I<:Integer}
    region == 1:1 && (ret = generic_fft(x))
    ret
end

function generic_fft!(x::StridedVector{T}, region::UnitRange{I}) where {T<:AbstractFloats, I<:Integer}
    region == 1:1 && (x[:] .= generic_fft(x))
    x
end

function generic_fft(x::StridedMatrix{T}, region::Integer) where T<:AbstractFloats
    if region == 1
        ret = hcat([generic_fft(x[:, j]) for j in 1:size(x, 2)]...)
    end
    ret
end

function generic_fft!(x::StridedMatrix{T}, region::Integer) where T<:AbstractFloats
    if region == 1
        for j in 1:size(x, 2)
            x[:, j] .= generic_fft(x[:, j])
        end
    end
    x
end

function generic_fft(x::Vector{T}) where T<:AbstractFloats
    T <: FFTW.fftwNumber && (@warn("Using generic fft for FFTW number type."))
    n = length(x)
    ispow2(n) && return generic_fft_pow2(x)
    ks = range(zero(real(T)),stop=n-one(real(T)),length=n)
    Wks = exp.((-im).*convert(T,π).*ks.^2 ./ n)
    xq, wq = x.*Wks, conj([exp(-im*convert(T,π)*n);reverse(Wks);Wks[2:end]])
    return Wks.*conv(xq,wq)[n+1:2n]
end

generic_bfft(x::StridedArray{T, N}, region) where {T <: AbstractFloats, N} = conj!(generic_fft(conj(x), region))
generic_bfft!(x::StridedArray{T, N}, region) where {T <: AbstractFloats, N} = conj!(generic_fft!(conj!(x), region))
generic_ifft(x::StridedArray{T, N}, region) where {T<:AbstractFloats, N} = ldiv!(length(x), conj!(generic_fft(conj(x), region)))
generic_ifft!(x::StridedArray{T, N}, region) where {T<:AbstractFloats, N} = ldiv!(length(x), conj!(generic_fft!(conj!(x), region)))

generic_rfft(v::Vector{T}, region) where T<:AbstractFloats = generic_fft(v, region)[1:div(length(v),2)+1]
function generic_irfft(v::Vector{T}, n::Integer, region) where T<:ComplexFloats
    @assert n==2length(v)-1
    r = Vector{T}(undef, n)
    r[1:length(v)]=v
    r[length(v)+1:end]=reverse(conj(v[2:end]))
    real(generic_ifft(r, region))
end
generic_brfft(v::StridedArray, n::Integer, region) = generic_irfft(v, n, region)*n

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

function generic_dct(x::StridedVector{T}, region::Integer) where T<:AbstractFloats
    region == 1 && (ret = generic_dct(x))
    ret
end

function generic_dct!(x::StridedVector{T}, region::Integer) where T<:AbstractFloats
    region == 1 && (x[:] .= generic_dct(x))
    x
end

function generic_idct(x::StridedVector{T}, region::Integer) where T<:AbstractFloats
    region == 1 && (ret = generic_idct(x))
    ret
end

function generic_idct!(x::StridedVector{T}, region::Integer) where T<:AbstractFloats
    region == 1 && (x[:] .= generic_idct(x))
    x
end

function generic_dct(x::StridedVector{T}, region::UnitRange{I}) where {T<:AbstractFloats, I<:Integer}
    region == 1:1 && (ret = generic_dct(x))
    ret
end

function generic_dct!(x::StridedVector{T}, region::UnitRange{I}) where {T<:AbstractFloats, I<:Integer}
    region == 1:1 && (x[:] .= generic_dct(x))
    x
end

function generic_idct(x::StridedVector{T}, region::UnitRange{I}) where {T<:AbstractFloats, I<:Integer}
    region == 1:1 && (ret = generic_idct(x))
    ret
end

function generic_idct!(x::StridedVector{T}, region::UnitRange{I}) where {T<:AbstractFloats, I<:Integer}
    region == 1:1 && (x[:] .= generic_idct(x))
    x
end

function generic_dct(a::AbstractVector{Complex{T}}) where {T <: AbstractFloat}
    T <: FFTW.fftwNumber && (@warn("Using generic dct for FFTW number type."))
    N = length(a)
    twoN = convert(T,2) * N
    c = generic_fft([a; reverse(a, dims=1)]) # c = generic_fft([a; flipdim(a,1)])
    d = c[1:N]
    d .*= exp.((-im*convert(T, pi)).*(0:N-1)./twoN)
    d[1] = d[1] / sqrt(convert(T, 2))
    lmul!(inv(sqrt(twoN)), d)
end

generic_dct(a::AbstractArray{T}) where {T <: AbstractFloat} = real(generic_dct(complex(a)))

function generic_idct(a::AbstractVector{Complex{T}}) where {T <: AbstractFloat}
    T <: FFTW.fftwNumber && (@warn("Using generic idct for FFTW number type."))
    N = length(a)
    twoN = convert(T,2)*N
    b = a * sqrt(twoN)
    b[1] = b[1] * sqrt(convert(T,2))
    shift = exp.(-im * 2 * convert(T, pi) * (N - convert(T,1)/2) * (0:(2N-1)) / twoN)
    b = [b; 0; -reverse(b[2:end], dims=1)] .* shift # b = [b; 0; -flipdim(b[2:end],1)] .* shift
    c = ifft(b)
    reverse(c[1:N]; dims=1)#flipdim(c[1:N],1)
end

generic_idct(a::AbstractArray{T}) where {T <: AbstractFloat} = real(generic_idct(complex(a)))


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
abstract type DummyPlan{T} <: Plan{T} end
for P in (:DummyFFTPlan, :DummyiFFTPlan, :DummybFFTPlan, :DummyDCTPlan, :DummyiDCTPlan)
    # All plans need an initially undefined pinv field
    @eval begin
        mutable struct $P{T,inplace,G} <: DummyPlan{T}
            region::G # region (iterable) of dims that are transformed
            pinv::DummyPlan{T}
            $P{T,inplace,G}(region::G) where {T<:AbstractFloats, inplace, G} = new(region)
        end
    end
end
for P in (:DummyrFFTPlan, :DummyirFFTPlan, :DummybrFFTPlan)
    @eval begin
        mutable struct $P{T,inplace,G} <: DummyPlan{T}
            n::Integer
            region::G # region (iterable) of dims that are transformed
            pinv::DummyPlan{T}
            $P{T,inplace,G}(n::Integer, region::G) where {T<:AbstractFloats, inplace, G} = new(n, region)
        end
    end
end

for (Plan,iPlan) in ((:DummyFFTPlan,:DummyiFFTPlan),
                     (:DummyDCTPlan,:DummyiDCTPlan))
   @eval begin
       plan_inv(p::$Plan{T,inplace,G}) where {T,inplace,G} = $iPlan{T,inplace,G}(p.region)
       plan_inv(p::$iPlan{T,inplace,G}) where {T,inplace,G} = $Plan{T,inplace,G}(p.region)
    end
end

# Specific for rfft, irfft and brfft:
plan_inv(p::DummyirFFTPlan{T,inplace,G}) where {T,inplace,G} = DummyrFFTPlan{T,Inplace,G}(p.n, p.region)
plan_inv(p::DummyrFFTPlan{T,inplace,G}) where {T,inplace,G} = DummyirFFTPlan{T,Inplace,G}(p.n, p.region)



for (Plan,ff,ff!) in ((:DummyFFTPlan,:generic_fft,:generic_fft!),
                      (:DummybFFTPlan,:generic_bfft,:generic_bfft!),
                      (:DummyiFFTPlan,:generic_ifft,:generic_ifft!),
                      (:DummyrFFTPlan,:generic_rfft,:generic_rfft!),
                      (:DummyDCTPlan,:generic_dct,:generic_dct!),
                      (:DummyiDCTPlan,:generic_idct,:generic_idct!))
    @eval begin
        *(p::$Plan{T,true}, x::StridedArray{T,N}) where {T<:AbstractFloats,N} = $ff!(x, p.region)
        *(p::$Plan{T,false}, x::StridedArray{T,N}) where {T<:AbstractFloats,N} = $ff(x, p.region)
        function mul!(C::StridedVector, p::$Plan, x::StridedVector)
            C[:] = $ff(x, p.region)
            C
        end
    end
end

# Specific for irfft and brfft:
*(p::DummyirFFTPlan{T,true}, x::StridedArray{T,N}) where {T<:AbstractFloats,N} = generic_irfft!(x, p.n, p.region)
*(p::DummyirFFTPlan{T,false}, x::StridedArray{T,N}) where {T<:AbstractFloats,N} = generic_irfft(x, p.n, p.region)
function mul!(C::StridedVector, p::DummyirFFTPlan, x::StridedVector)
    C[:] = generic_irfft(x, p.n, p.region)
    C
end
*(p::DummybrFFTPlan{T,true}, x::StridedArray{T,N}) where {T<:AbstractFloats,N} = generic_brfft!(x, p.n, p.region)
*(p::DummybrFFTPlan{T,false}, x::StridedArray{T,N}) where {T<:AbstractFloats,N} = generic_brfft(x, p.n, p.region)
function mul!(C::StridedVector, p::DummybrFFTPlan, x::StridedVector)
    C[:] = generic_brfft(x, p.n, p.region)
    C
end


# We override these for AbstractFloat, so that conversion from reals to
# complex numbers works for any AbstractFloat (instead of only BlasFloat's)
AbstractFFTs.complexfloat(x::StridedArray{Complex{<:AbstractFloat}}) = x
AbstractFFTs.realfloat(x::StridedArray{<:Real}) = x
# We override this one in order to avoid throwing an error that the type is
# unsupported (as defined in AbstractFFTs)
AbstractFFTs._fftfloat(::Type{T}) where {T <: AbstractFloat} = T


# We intercept the calls to plan_X(x, region) below.
# In order not to capture any calls that should go to FFTW, we have to be
# careful about the typing, so that the calls to FFTW remain more specific.
# This is the reason for using StridedArray below. We also have to carefully
# distinguish between real and complex arguments.

plan_fft(x::StridedArray{T}, region) where {T <: ComplexFloats} = DummyFFTPlan{Complex{real(T)},false,typeof(region)}(region)
plan_fft!(x::StridedArray{T}, region) where {T <: ComplexFloats} = DummyFFTPlan{Complex{real(T)},true,typeof(region)}(region)

plan_bfft(x::StridedArray{T}, region) where {T <: ComplexFloats} = DummybFFTPlan{Complex{real(T)},false,typeof(region)}(region)
plan_bfft!(x::StridedArray{T}, region) where {T <: ComplexFloats} = DummybFFTPlan{Complex{real(T)},true,typeof(region)}(region)

# The ifft plans are automatically provided in terms of the bfft plans above.
# plan_ifft(x::StridedArray{T}, region) where {T <: ComplexFloats} = DummyiFFTPlan{Complex{real(T)},false,typeof(region)}(region)
# plan_ifft!(x::StridedArray{T}, region) where {T <: ComplexFloats} = DummyiFFTPlan{Complex{real(T)},true,typeof(region)}(region)

plan_dct(x::StridedArray{T}, region) where {T <: AbstractFloats} = DummyDCTPlan{T,false,typeof(region)}(region)
plan_dct!(x::StridedArray{T}, region) where {T <: AbstractFloats} = DummyDCTPlan{T,true,typeof(region)}(region)

plan_idct(x::StridedArray{T}, region) where {T <: AbstractFloats} = DummyiDCTPlan{T,false,typeof(region)}(region)
plan_idct!(x::StridedArray{T}, region) where {T <: AbstractFloats} = DummyiDCTPlan{T,true,typeof(region)}(region)

plan_rfft(x::StridedArray{T}, region) where {T <: RealFloats} = DummyrFFTPlan{Complex{real(T)},false,typeof(region)}(length(x), region)
plan_brfft(x::StridedArray{T}, n::Integer, region) where {T <: ComplexFloats} = DummybrFFTPlan{Complex{real(T)},false,typeof(region)}(n, region)

# A plan for irfft is created in terms of a plan for brfft.
# plan_irfft(x::StridedArray{T}, n::Integer, region) where {T <: ComplexFloats} = DummyirFFTPlan{Complex{real(T)},false,typeof(region)}(n, region)

# These don't exist for now:
# plan_rfft!(x::StridedArray{T}) where {T <: RealFloats} = DummyrFFTPlan{Complex{real(T)},true}()
# plan_irfft!(x::StridedArray{T},n::Integer) where {T <: RealFloats} = DummyirFFTPlan{Complex{real(T)},true}()

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

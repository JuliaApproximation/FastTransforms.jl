conv(u::AbstractArray{T, N}, v::AbstractArray{T, N}) where {T<:AbstractFloat, N} = GenericFFT._conv!(deepcopy(u), deepcopy(v))
conv(u::AbstractArray{T, N}, v::AbstractArray{Complex{T}, N}) where {T<:AbstractFloat, N} = GenericFFT._conv!(complex(deepcopy(u)), deepcopy(v))
conv(u::AbstractArray{Complex{T}, N}, v::AbstractArray{T, N}) where {T<:AbstractFloat, N} = GenericFFT._conv!(deepcopy(u), complex(deepcopy(v)))
conv(u::AbstractArray{Complex{T}, N}, v::AbstractArray{Complex{T}, N}) where {T<:AbstractFloat, N} = GenericFFT._conv!(deepcopy(u), deepcopy(v))

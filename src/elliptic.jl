"""
`FastTransforms` submodule for the computation of some elliptic integrals and functions.

Complete elliptic integrals of the first and second kinds:
```math
K(k) = \\int_0^{\\frac{\\pi}{2}} \\frac{{\\rm d}\\theta}{\\sqrt{1-k^2\\sin^2\\theta}},\\quad{\\rm and},
```
```math
E(k) = \\int_0^{\\frac{\\pi}{2}} \\sqrt{1-k^2\\sin^2\\theta} {\\rm\\,d}\\theta.
```

Jacobian elliptic functions:
```math
x = \\int_0^{\\operatorname{sn}(x,k)} \\frac{{\\rm d}t}{\\sqrt{(1-t^2)(1-k^2t^2)}},
```
```math
x = \\int_{\\operatorname{cn}(x,k)}^1 \\frac{{\\rm d}t}{\\sqrt{(1-t^2)[1-k^2(1-t^2)]}},
```
```math
x = \\int_{\\operatorname{dn}(x,k)}^1 \\frac{{\\rm d}t}{\\sqrt{(1-t^2)(t^2-1+k^2)}},
```
and the remaining nine are defined by:
```math
\\operatorname{pq}(x,k) = \\frac{\\operatorname{pr}(x,k)}{\\operatorname{qr}(x,k)} = \\frac{1}{\\operatorname{qp}(x,k)}.
```
"""
module Elliptic

import FastTransforms: libfasttransforms

export K, E,
       sn, cn, dn, ns, nc, nd,
       sc, cs, sd, ds, cd, dc

for (fC, elty) in ((:ft_complete_elliptic_integralf, :Float32), (:ft_complete_elliptic_integral, :Float64))
    @eval begin
        function K(k::$elty)
            return ccall(($(string(fC)), libfasttransforms), $elty, (Cint, $elty), '1', k)
        end
        function E(k::$elty)
            return ccall(($(string(fC)), libfasttransforms), $elty, (Cint, $elty), '2', k)
        end
    end
end

const SN = UInt(1)
const CN = UInt(2)
const DN = UInt(4)

for (fC, elty) in ((:ft_jacobian_elliptic_functionsf, :Float32), (:ft_jacobian_elliptic_functions, :Float64))
    @eval begin
        function sn(x::$elty, k::$elty)
            retsn = Ref{$elty}()
            ccall(($(string(fC)), libfasttransforms), Cvoid, ($elty, $elty, Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, UInt), x, k, retsn, C_NULL, C_NULL, SN)
            retsn[]
        end
        function cn(x::$elty, k::$elty)
            retcn = Ref{$elty}()
            ccall(($(string(fC)), libfasttransforms), Cvoid, ($elty, $elty, Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, UInt), x, k, C_NULL, retcn, C_NULL, CN)
            retcn[]
        end
        function dn(x::$elty, k::$elty)
            retdn = Ref{$elty}()
            ccall(($(string(fC)), libfasttransforms), Cvoid, ($elty, $elty, Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, UInt), x, k, C_NULL, C_NULL, retdn, DN)
            retdn[]
        end
        function ns(x::$elty, k::$elty)
            retsn = Ref{$elty}()
            ccall(($(string(fC)), libfasttransforms), Cvoid, ($elty, $elty, Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, UInt), x, k, retsn, C_NULL, C_NULL, SN)
            inv(retsn[])
        end
        function nc(x::$elty, k::$elty)
            retcn = Ref{$elty}()
            ccall(($(string(fC)), libfasttransforms), Cvoid, ($elty, $elty, Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, UInt), x, k, C_NULL, retcn, C_NULL, CN)
            inv(retcn[])
        end
        function nd(x::$elty, k::$elty)
            retdn = Ref{$elty}()
            ccall(($(string(fC)), libfasttransforms), Cvoid, ($elty, $elty, Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, UInt), x, k, C_NULL, C_NULL, retdn, DN)
            inv(retdn[])
        end
        function sc(x::$elty, k::$elty)
            retsn = Ref{$elty}()
            retcn = Ref{$elty}()
            ccall(($(string(fC)), libfasttransforms), Cvoid, ($elty, $elty, Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, UInt), x, k, retsn, retcn, C_NULL, SN & CN)
            retsn[]/retcn[]
        end
        function cs(x::$elty, k::$elty)
            retsn = Ref{$elty}()
            retcn = Ref{$elty}()
            ccall(($(string(fC)), libfasttransforms), Cvoid, ($elty, $elty, Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, UInt), x, k, retsn, retcn, C_NULL, SN & CN)
            retcn[]/retsn[]
        end
        function sd(x::$elty, k::$elty)
            retsn = Ref{$elty}()
            retdn = Ref{$elty}()
            ccall(($(string(fC)), libfasttransforms), Cvoid, ($elty, $elty, Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, UInt), x, k, retsn, C_NULL, retdn, SN & DN)
            retsn[]/retdn[]
        end
        function ds(x::$elty, k::$elty)
            retsn = Ref{$elty}()
            retdn = Ref{$elty}()
            ccall(($(string(fC)), libfasttransforms), Cvoid, ($elty, $elty, Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, UInt), x, k, retsn, C_NULL, retdn, SN & DN)
            retdn[]/retsn[]
        end
        function cd(x::$elty, k::$elty)
            retcn = Ref{$elty}()
            retdn = Ref{$elty}()
            ccall(($(string(fC)), libfasttransforms), Cvoid, ($elty, $elty, Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, UInt), x, k, C_NULL, retcn, retdn, CN & DN)
            retcn[]/retdn[]
        end
        function dc(x::$elty, k::$elty)
            retcn = Ref{$elty}()
            retdn = Ref{$elty}()
            ccall(($(string(fC)), libfasttransforms), Cvoid, ($elty, $elty, Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, UInt), x, k, C_NULL, retcn, retdn, CN & DN)
            retdn[]/retcn[]
        end
    end
end

end # module

function _stepthreadsfor(iter,lbody)
    lidx = iter.args[1]         # index
    range = iter.args[2]
    quote
        local stepthreadsfor_fun
        let range = $(esc(range))
        function stepthreadsfor_fun(onethread=false)
            r = range # Load into local variable
            lenr = length(r)
            # divide loop iterations among threads
            if onethread
                tid = 1
                len, rem = lenr, 0
            else
                tid = Threads.threadid()
                len, rem = divrem(lenr, Threads.nthreads())
            end
            # not enough iterations for all the threads?
            if len == 0
                if tid > rem
                    return
                end
                len, rem = 1, 0
            end
            # compute this thread's iterations
            f = tid
            m = Threads.nthreads()
            l = lenr
            # run this thread's iterations
            for i = f:m:l
                local $(esc(lidx)) = Base.unsafe_getindex(r,i)
                $(esc(lbody))
            end
        end
        end
        # Hack to make nested threaded loops kinda work
        if Threads.threadid() != 1 || Threads.in_threaded_loop[]
            # We are in a nested threaded loop
            stepthreadsfor_fun(true)
        else
            Threads.in_threaded_loop[] = true
            # the ccall is not expected to throw
            ccall(:jl_threading_run, Ref{Void}, (Any,), stepthreadsfor_fun)
            Threads.in_threaded_loop[] = false
        end
        nothing
    end
end
"""
    @stepthreads
A macro to parallelize a for-loop to run with multiple threads. This spawns `nthreads()`
number of threads, splits the iteration space amongst them, and iterates in parallel.
A barrier is placed at the end of the loop which waits for all the threads to finish
execution, and the loop returns.
"""
macro stepthreads(args...)
    na = length(args)
    if na != 1
        throw(ArgumentError("wrong number of arguments in @stepthreads"))
    end
    ex = args[1]
    if !isa(ex, Expr)
        throw(ArgumentError("need an expression argument to @stepthreads"))
    end
    if ex.head === :for
        return _stepthreadsfor(ex.args[1],ex.args[2])
    else
        throw(ArgumentError("unrecognized argument to @stepthreads"))
    end
end

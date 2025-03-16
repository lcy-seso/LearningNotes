xs = [fill(1.1, 3), fill(2.2, 3)];

# using Pkg
# Pkg.activate(".")

# zero(x::Array{Float64,1}) = [zero(x) for x in x]
#
# function case1(xs)
#     h = xs[1][1]
#     # sum(h)
# end
#
# Zygote.gradient(case1, xs)


using Pkg
Pkg.activate(".")
using Zygote

function case1(xs)
    h = xs[1]
    # for i in 2:length(xs)
    #     h = h .* xs[i]
    # end
    sum(h)
end
@show case1(xs)

Zygote.gradient(case1, xs)

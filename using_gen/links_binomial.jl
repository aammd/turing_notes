using Gen, CSV, StatsPlots;

import StatsFuns.logistic

xs = [-5., -4., -3., -.2, -1., 0., 1., 2., 3., 4., 5.];

## Calculate logistic curve -- either on my own or via Statsfuns.logistic
function inv_logit(xs::Vector{Float64})
    p = map(x -> exp(x) / (exp(x) + 1), xs)
    return p
end;

inv_logit([0.])

map(logistic, [5,0,-3])

## whoops, turns out there *isnt* a better 


linknode_data = CSV.read("mangal_data.dat")

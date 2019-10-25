import Distributions
import CSV
import DataFrames
import Turing
using Gadfly

linknode_data = CSV.read("mangal_data.dat")

# select just webs where predation is positive
webdata = linknode_data[linknode_data.predation .> 0, :]

webdata
nobig_web=filter(x -> x[:nodes]<700, webdata)
# or
webdata[webdata.nodes.<700,:]
####
# must use a truncated exponential, see example at https://github.com/StatisticalRethinkingJulia/TuringModels.jl/blob/master/chapters/11/m11.5t.jl
@model betabinomial_links_eq(nodes, links) = begin
    # intercept
    a ~ Normal(-2, 0.1)#Normal(1,0.2);
    # slope
    b ~ Normal(-0.2, 0.1);
    # dispersion
    #ϕ_0 ~ Normal(-2,0.5)
    #ϕ_1 ~ Normal(0.1,0.1)
    ϕ ~ Truncated(Exponential(6), 0, Inf)

    max_links = nodes .^ 2
    v = ((nodes .- 1) ./ nodes.^2) .+ ( (nodes.^2 .- nodes .+ 1) .* exp.(a) .* nodes .^(b .- 2) ) ./ ( 1 .+ exp.(a) .* nodes .^ b)

    #print(v)
    for i in 1:length(links)
        links[i] ~ BetaBinomial(max_links[i], v[i] * ϕ, (1 - v[i]) * ϕ )
    end
    return links
end


## prior predictive
link_gen = betabinomial_links_eq(nodes = webdata.nodes,
                          links = fill(missing, length(webdata.nodes)))
## add data to the function so it can make calculations
link_gen()
webdata_prior_predict = select(webdata, :nodes)

links_with_data = betabinomial_links_eq(nodes = nobig_web.nodes,
 links = nobig_web.links)
links_with_data()

chains_data = sample(links_with_data, HMCDA(200, 0.65, 0.3), 2000)
chains_data


# use cool mapping techniques to produce posterior

webdata_prior_predict.links = link_gen()
Gadfly.plot(webdata_prior_predict,
            x = :nodes, y = :links,
            Geom.point,  Scale.y_log10, Scale.x_log10())

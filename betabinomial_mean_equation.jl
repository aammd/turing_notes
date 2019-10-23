using Turing, Distributions, CSV

using DataFrames

using Gadfly

#using StatsFuns: logistic

linknode_data = CSV.read("mangal_data.dat")

# select just webs where predation is positive
webdata = linknode_data[linknode_data.predation .> 0, :]

####

@model betabinomial_links_eq(nodes, links) = begin
    # intercept
    a ~ Normal(-2, 0.1)#Normal(1,0.2);
    # slope
    b ~ Normal(-0.2, 0.1);
    # dispersion
    ϕ_0 ~ Normal(-2,0.5)
    ϕ_1 ~ Normal(0.1,0.1)

    max_links = nodes .^ 2
    v = ((nodes .- 1) ./ nodes.^2) .+ ( (nodes.^2 .- nodes .+ 1) .* exp.(a) .* nodes .^(b .- 2) ) ./ ( 1 .+ exp.(a) .* nodes .^ b)
    ϕ = ϕ_0 .+ ϕ_1 .* (nodes) 
    #print(v)
    for i in 1:length(links)
        links[i] ~ BetaBinomial(max_links[i], v[i] * exp(ϕ[i]), (1 - v[i]) * exp(ϕ[i]))
    end
    return links
end


## prior predictive
link_gen = betabinomial_links_eq(nodes = webdata.nodes,
                          links = fill(missing, length(webdata.nodes)))
## add data to the function so it can make calculations
link_gen()
webdata_prior_predict = select(webdata, :nodes)

webdata_prior_predict.links = link_gen()
Gadfly.plot(webdata_prior_predict,
            x = :nodes, y = :links,
            Geom.point,  Scale.y_log10, Scale.x_log10())


# sampling

links_with_data = betabinomial_links_eq(nodes = webdata.nodes,
 links = webdata.links)
links_with_data()

chains_data = sample(links_with_data, HMC(0.05, 10), 2000)

chains_data

function constrained_logistic(a,b)
 ((nodes .- 1) ./ nodes.^2) .+ ( (nodes.^2 .- nodes .+ 1) .* exp.(a) .* nodes .^(b .- 2) ) ./ ( 1 .+ exp.(a) .* nodes .^ b)
end

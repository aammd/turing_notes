using Turing, Distributions, CSV

using DataFrames, StatsPlots

using Gadfly

using StatsFuns: logistic

linknode_data = CSV.read("mangal_data.dat")

# select just webs where predation is positive
webdata = linknode_data[linknode_data.predation .> 0, :]

@model binomial_links(nodes, links) = begin
    # intercept
    β_0 ~ Normal(.57,0.09);
    # slope
    β_1 ~ Normal(1.73,0.09);

    # std of the response
    σ ~ Normal(0,1)

    max_links = nodes .^ 2

    for i in 1:length(nodes)
        links[i] ~ Binomial(400, logistic(β_0 * nodes[i] ^ β_1))
    end
    return links
end

## prior predictive
link_gen = binomial_links(nodes = webdata.nodes,
                          links = fill(missing, length(webdata.nodes)))
## add data to the function so it can make calculations
link_gen()
## No!! why are they all the same??

#### this was failing because the probabilities are too high!!
v = 0.3 .* webdata.nodes .^ -0.5
map(logistic, v)


@model binomial_links(nodes, links) = begin
    # intercept
    β_0 ~ Normal(1,0.5);
    # slope
    β_1 ~ Normal(0.5,0.5);

    max_links = nodes .^ 2

    for i in 1:length(nodes)
        links[i] ~ Binomial(max_links[i], logistic(β_0 * nodes[i] ^ β_1))
    end
    return links
end

## prior predictive
link_gen = binomial_links(nodes = webdata.nodes,
                          links = fill(missing, length(webdata.nodes)))
## add data to the function so it can make calculations
link_gen()
webdata_prior_predict = select(webdata, :nodes)
webdata_prior_predict.links = link_gen()
Gadfly.plot(webdata_prior_predict, x = :nodes, y = :links, Geom.point,  Scale.y_log10)

Gadfly.plot(webdata, x = :nodes, y = :links, Geom.point, Scale.y_log10)
## ugh now why is it a line??? perhaps because real data is too variable

#####################################
## try sampling posterior
links_with_data = binomial_links(nodes = webdata.nodes, links = webdata.links)
links_with_data()

chains_data = sample(links_with_data, HMC(0.05, 10), 2000)


##### Beta binomial

## trying with ϕ ~ Normal(0,0.1)
# μ[i] * 0.5, (1 - μ[i]) * 0.5
 # -- bounds error both times
rand(Beta(0.5, 0.5))

@model betabinomial_links(nodes, links) = begin
    # intercept
    β_0 ~ Normal(1,0.2);
    # slope
    β_1 ~ Normal(0.2,0.2);
    # dispersion
    ϕ ~ Exponential(6)

    max_links = nodes .^ 2
    for i in 1:length(links)
        v = logistic(β_0 * nodes[i] ^ β_1)
        links[i] ~ BetaBinomial(max_links[i], v * ϕ, (1 - v) * ϕ)
    end
    return links
end
## prior predictive
link_gen = betabinomial_links(nodes = webdata.nodes,
                          links = fill(missing, length(webdata.nodes)))
## add data to the function so it can make calculations
link_gen()
webdata_prior_predict = select(webdata, :nodes)
webdata_prior_predict.links = link_gen()
Gadfly.plot(webdata_prior_predict,
            x = :nodes, y = :links,
            Geom.point,  Scale.y_log10, Scale.x_log10())

Gadfly.plot(webdata, x = :nodes, y = :links, Geom.point, Scale.y_log10)
## ugh now why is it a line??? perhaps because real data is too variable

#####################################
## try sampling posterior
links_with_data = betabinomial_links(nodes = webdata.nodes,
 links = webdata.links)
links_with_data()

chains_data = sample(links_with_data, HMC(0.05, 10), 2000)

Gadfly.plot(webdata_prior_predict,
            x = :nodes, y = :links,
            Geom.point,  Scale.y_log10, Scale.x_log10())


β_1s = chains_data[:β_1].value[9]
β_0s = chains_data[:β_0].value[9]
ϕ_s    = chains_data[:ϕ].value[9]

# get only the observed values
webdata_post = select(webdata, [:nodes, :links])

webdata_post.mean_prediction = map(logistic, β_0s .* webdata_post.nodes .^ β_1s)
webdata_post.max_links = webdata_post.nodes .^ 2
webdata_post.prediction = map(x -> rand(BetaBinomial(x.max_links, x.mean_prediction * ϕ_s, (1 -x.mean_prediction) * ϕ_s)),
                             eachrow(webdata_post))
rand.(BetaBinomial.(webdata_post.max_links, webdata_post.mean_prediction * ϕ_s, (1 .- webdata_post.mean_prediction) * ϕ_s))
iszero.(webdata_post.max_links)
map(iszero, webdata_post.max_links)
webdata_post_stacked = stack(webdata_post[:, [:nodes, :links, :prediction]], [:links, :prediction])

Gadfly.plot(webdata_post_stacked,
            x = :nodes, y = :value,
             Geom.point,
            color = :variable,Scale.y_log10, Scale.x_log10)

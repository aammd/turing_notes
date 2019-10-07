using Turing, Distributions, CSV

using DataFrames, StatsPlots

linknode_data = CSV.read("mangal_data.dat")

# select just webs where predation is positive
webdata = linknode_data[linknode_data.predation .> 0, :]

@model links_nodes(nodes, links) = begin
    # intercept
    β_0 ~ Normal(-.57,0.09);
    # slope
    β_1 ~ Normal(1.73,0.09);

    # std of the response
    σ ~ Normal(0,1)

    for i in 1:length(links)
        links[i] ~ Normal(β_0 + β_1 * nodes[i], exp(σ))
    end
    return β_0, β_1, links
end

ff = rand(Uniform(-2,5), 100)
link_gen = links_nodes(nodes = ff, links = fill(missing, 100))

scatter(ff, link_gen()[3])

link_gen()[3]

links_fake = links_nodes(ff, link_gen()[3])
### conditioning
chain = sample(links_fake, HMC(0.05, 10), 2000)

## real data
node_data = map(log, webdata.nodes)
link_data = map(log, webdata.links)
node_data = node_data .- mean(node_data)

links_with_data = links_nodes(node_data, link_data)
chains_data = sample(links_with_data, HMC(0.05, 10), 2000)

## failure!

scatter(webdata.nodes, webdata.links)


### WHY does this produce the same vector every time??
prior_predictions = [link_gen() for _=1:10]

map(x -> x[3], prior_predictions)


link_data = p_p[:links].value.data

scatter(ff, link_data[38,:,1])


gen_links = links_nodes(nodes = 1:10, links = fill(missing, 10))
one_sim = gen_links()



links_sim = DataFrame(p_p[:links])

scatter(x = ff, y = links_sim[1,:])

## prior predictive checks

scatter(ff, one_sim[3])
plot(ff, one_sim[3], seriestype = :scatter)
##
link_data[18,:,1]
scatter(ff, link_data[12,:,1])


############

@model links_nodes_trunc(nodes, links) = begin
    # intercept
    β_0 ~ Normal(-.57,0.09);
    # slope
    β_1 ~ Normal(1.73,0.09);

    # std of the response
    σ ~ Normal(1,2)

    for i in 1:length(links)
        μ[i] = β_0 + β_1 * nodes[i]
        links[i] ~ TruncatedNormal(μ[i], exp(σ), nodes[i] - 1, nodes[i]^2)
    end
    return β_0, β_1, links
end

ff = rand(Uniform(-2,5), 100)
link_gen = links_nodes_trunc(nodes = ff, links = fill(missing, 100))

scatter(ff, link_gen()[3])

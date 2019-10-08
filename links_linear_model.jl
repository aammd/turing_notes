using Turing, Distributions, CSV

using DataFrames, StatsPlots

using Gadfly

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

## OK you can sample from the prior predictive distribution
## by including links as the return value of the function
ff = rand(Uniform(-2,5), 100)
link_gen = links_nodes(nodes = ff, links = fill(missing, 100))
## add data to the function so it can make calculations
link_gen() # samples the prior predictive distribution
scatter(ff, link_gen()[3])



links_fake = links_nodes(ff, link_gen()[3])
### conditioning
chain = sample(links_fake, HMC(0.05, 10), 2000)

## real data
node_data = map(log, webdata.nodes)
link_data = map(log, webdata.links)
node_data = node_data .- mean(node_data)

### prior predictions with real data

link_gen_real = links_nodes(nodes = node_data, links = fill(missing, 100))
link_gen_real()

## now add link data
links_with_data = links_nodes(node_data, link_data)
chains_data = sample(links_with_data, HMC(0.05, 10), 2000)


scatter(webdata.nodes, webdata.links)
scatter(node_data, link_data)




### QUESTION
### WHY does this produce the same vector every time??
prior_predictions = [link_gen() for _=1:10]
map(x -> x[3], prior_predictions)



#################################
### SIMPLE PLOT
Gadfly.plot(webdata, x = :nodes, y = :links, Geom.point, Scale.y_log, Scale.x_log)

insertcols!(webdata, 1, :lognode => map(log, webdata.nodes))
insertcols!(webdata, 2, :loglinks => map(log, webdata.links))

## simple gadfly plot based on http://gadflyjl.org/stable/gallery/geometries/#[Geom.abline](@ref)-1
Gadfly.plot(webdata, x = :lognode, y = :loglinks,
 Geom.point,
 intercept = [-.57], slope = [1.73],
 Geom.abline(color="red", style=:dash)
 )

# can the slope and intercept be vectors


β_1s = chains_data[:β_1].value[1:200]
β_0s = chains_data[:β_0].value[1:200]

#Gadfly.plot(webdata, x = :lognode, y = :loglinks,
# Geom.point,
# intercept = β_0s, slope = β_1s,
# Geom.abline()
# )

 # can you add predicted values easily??

stack()

 β_1s = chains_data[:β_1].value[4]
 β_0s = chains_data[:β_0].value[4]
 σ_s = chains_data[:σ].value[4]

# get only the observed values
webdata_post = select(webdata, [:lognode, :loglinks])

webdata_post.mean_prediction = β_0s .+ β_1s .* webdata.lognode
webdata_post.prediction = map(x -> rand(Normal(x, exp(σ_s))), webdata_post.mean_prediction)

webdata_post_stacked = stack(webdata_post, [:loglinks, :prediction])

Gadfly.plot(webdata_post_stacked,
 x = :lognode, y = :value,
 Geom.point,
 color = :variable
 )
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

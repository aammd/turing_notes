using Gen;
using PyPlot

xs = [-5., -4., -3., -.2, -1., 0., 1., 2., 3., 4., 5.];

@gen function line_model(xs::Vector{Float64})
    n = length(xs)

    β0 = @trace(normal(0, 1), :β0)
    β1 = @trace(normal(0, 2), :β1)
    for (i, x) in enumerate(xs)
        @trace(normal(β0 * x + β1, 0.1), (:y, i))
    end
    return n
end;

trace = Gen.simulate(line_model, (xs,))

println(Gen.get_choices(trace))

choices = Gen.get_choices(trace)
choices[:β1]

# extract all the ys into an Array
[trace[(:y, i)] for i=1:length(xs)]

# why does trace and choices give you the same thing?
trace[:β1]
choices[:β1]

ys = [trace[(:y, i)] for i=1:length(xs)]
scatter(xs, ys)

############# variable variance

@gen function line_model_var(xs::Vector{Float64})
    n = length(xs)

    β0 = @trace(normal(0, 1), :β0)
    β1 = @trace(normal(0, 2), :β1)
    σ  = @trace(exponential(3), :σ)
    for (i, x) in enumerate(xs)
        @trace(normal(β0 * x + β1, σ), (:y, i))
    end
    return n
end;

trace = Gen.simulate(line_model_var, (xs,))
ys = [trace[(:y, i)] for i=1:length(xs)]
scatter(xs, ys)
PyPlot.display_figs()

function plot_points(mod)
    trace = Gen.simulate(mod, (xs,))
    ys = [trace[(:y, i)] for i=1:length(xs)]
    scatter(xs, ys)
    PyPlot.display_figs()
end

###### poisson


@gen function poisson_line(xs::Vector{Float64})
    β0 = @trace(normal(1, 0.5), :β0)
    β1 = @trace(normal(1, 0.5), :β1)
    for (i, x) in enumerate(xs)
        @trace(poisson(exp(β0 + β1 * x)), (:y, i))
    end
end;

plot_points(poisson_line)


##### overdispersed poisson

@gen function poisson_od(xs::Vector{Float64})
    β0 = @trace(normal(0, 1), :β0)
    β1 = @trace(normal(0, 2), :β1)

    σ  = @trace(exponential(3), :σ)

    for (i, x) in enumerate(xs)
        α = @trace(normal(0, σ), (:α, i))
    end

    for (i, x) in enumerate(xs)
        @trace(poisson(exp(β0 * x + β1 + α)), (:y, i))
    end
end;
# alpha not defined == but why? it is in a loop
plot_points(poisson_od)

################################
@gen function poisson_od(xs::Vector{Float64})
    β0 = @trace(normal(2, 0.2), :β0)
    β1 = @trace(normal(0.4, 0.1), :β1)

    σ  = @trace(exponential(3), :σ)

    α = [@trace(normal(0, σ), (:α, i)) for i=1:length(xs)]

    for (i, x) in enumerate(xs)
        @trace(poisson(exp(β0 + β1 * x + α[i])), (:y, i))
    end
end;
plot_points(poisson_od)

#########################

### hierarchical groups
@gen function hier_groups(xs::Vector{Int64})

    ngrps = length(unique(xs))
    σ_block  = @trace(exponential(3), :σ_block)
    block = [@trace(normal(0, σ_block), (:block, i)) for i=1:ngrps]

    σ_response = @trace(exponential(6), :σ_response)
    for (i, x) in enumerate(xs)
        @trace(normal( block[xs[i]],  σ_response), (:y, i))
    end
end;

x_int = [1,1,1,2,2,2,3,3,3]
trace = Gen.simulate(hier_groups, (x_int,))
ys = [trace[(:y, i)] for i=1:9]
scatter(x_int, ys)
PyPlot.display_figs()

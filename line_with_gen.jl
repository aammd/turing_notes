using Gen;
using PyPlot

xs = [-5., -4., -3., -.2, -1., 0., 1., 2., 3., 4., 5.];

@gen function line_model(xs::Vector{Float64})
    n = length(xs)

    # We begin by sampling a slope and intercept for the line.
    # Before we have seen the data, we don't know the values of
    # these parameters, so we treat them as random choices. The
    # distributions they are drawn from represent our prior beliefs
    # about the parameters: in this case, that neither the slope nor the
    # intercept will be more than a couple points away from 0.
    β0 = @trace(normal(0, 1), :β0)
    β1 = @trace(normal(0, 2), :β1)

    # Given the slope and intercept, we can sample y coordinates
    # for each of the x coordinates in our input vector.
    for (i, x) in enumerate(xs)
        @trace(normal(β0 * x + β1, 0.1), (:y, i))
    end

    # The return value of the model is often not particularly important,
    # Here, we simply return n, the number of points.
    return n
end;

trace = Gen.simulate(line_model, (xs,));

println(Gen.get_choices(trace))

choices = Gen.get_choices(trace)
choices[:β1]

# extract all the ys into an Array
[trace[(:y, i)] for i=1:length(xs)]

function render_trace(trace; show_data=true)

    # Pull out xs from the trace
    xs = get_args(trace)[1]

    xmin = minimum(xs)
    xmax = maximum(xs)
    if show_data
        ys = [trace[(:y, i)] for i=1:length(xs)]

        # Plot the data set
        scatter(xs, ys, c="black")
    end

    # Pull out slope and intercept from the trace
    slope = trace[:slope]
    intercept = trace[:intercept]

    # Draw the line
    plot([xmin, xmax], slope *  [xmin, xmax] .+ intercept, color="black", alpha=0.5)
    ax = gca()
    ax[:set_xlim]((xmin, xmax))
    ax[:set_ylim]((xmin, xmax))
end;

function grid(renderer::Function, traces; ncols=6, nrows=3)
    figure(figsize=(16, 8))
    for (i, trace) in enumerate(traces)
        subplot(nrows, ncols, i)
        renderer(trace)
    end
end;

figure(figsize=(3,3))
render_trace(trace)
PyPlot.display_figs()

traces = [Gen.simulate(line_model, (xs,)) for _=1:12]
grid(render_trace, traces)
PyPlot.display_figs()

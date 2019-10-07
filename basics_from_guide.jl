using Turing, Distributions

@model gdemo(x, y) = begin
  s ~ InverseGamma(2, 3)
  m ~ Normal(0, sqrt(s))
  x ~ Normal(m, sqrt(s))
  y ~ Normal(m, sqrt(s))
  return x, y
end

# Samples from p(x,y)
g_prior_sample = gdemo()
g_prior_sample()


# Declare a model with a default value.
@model generative(x = Vector{Real}(undef, 10)) = begin
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    for i in 1:length(x)
        x[i] ~ Normal(m, sqrt(s))
    end
    return s, m
end

g = generative()
g()
some_samples = [g() for i in 1:10]

map(x -> x[1,], some_samples)

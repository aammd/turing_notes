## can't get my binomial to work! back to the example

# Bayesian logistic regression (LR)
@model ber_trials(n, y, σ) = begin
    intercept ~ Normal(0, σ)

    for i = 1:n
        v = logistic(intercept)
        y[i] ~ Bernoulli(v)
    end
    return y
end;

samp_ber = ber_trials(n = 20, y = fill(missing, 20), σ = 1)

samp_ber()

## okay so that works -- we get bernoulli results

@model bin_trials(n, y, σ) = begin
    intercept ~ Normal(0, σ)

    for i = 1:n
        v = logistic(intercept)
        y[i] ~ Binomial(35, v)
    end
    return y
end;

samp_ber = bin_trials(n = 20, y = fill(missing, 20), σ = 1)

samp_ber()

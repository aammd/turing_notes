# sampling

links_with_data = betabinomial_links_eq(nodes = webdata.nodes,
 links = webdata.links)
links_with_data()

chains_data = sample(links_with_data, HMC(0.05, 10), 2000)

chains_data

function constrained_logistic(nodes,a,b)
 ((nodes .- 1) ./ nodes.^2) .+ ( (nodes.^2 .- nodes .+ 1) .* exp.(a) .* nodes .^(b .- 2) ) ./ ( 1 .+ exp.(a) .* nodes .^ b)
end

# trying to riff on

function using_samples(a, b)
    return x->constrained_logistic(x, a, b)
end

aa = [1,4,6]
bb = [4,2,3]

function make_line(a)
    f = function(x) x .* a
    end
    return f
end

ffs=make_line.(aa)

map(x -> x(bb), ffs)
[f(bb) for f in ffs]

# this is my favourite solution to the problem
function make_line2(a, b)
    f = function(x) x .* a .+ b
    end
    return f
end

cc = collect(-1:1:1)
ffs2=make_line2.(aa, cc)

[f(bb) for f in ffs2]

function f(a, b, c)
           b .* a .+ c
end

[f(a, b, c) for a in aa, b in bb, c in cc]


function ffs3(x)
           x .* a .+ c
end

[f(bb) for f in ffs3]

ffs[3](bb) # yes
map(ffs, bb) #does not work
map(x->map(x, bb), ffs) #ffs, x-> print(x)) #also does not work

broadcast(ffs[1], bb)

fi(f, s) = map(f, s)


fi(make_line, aa)

f0(x) = 2x
f2(x) = x^2
fsf1(number) = fi(f0, number)
fsf1(5)
fsf2(s) = fs(f2, s)

using DrWatson
@quickactivate "DynCausal"
using LinearAlgebra, Statistics
using Plots
using JLD2
using MAT
## =====

data_path = "C:/Users/aresf/Desktop/OMM_archive_submission_final/OMM_archive_submission_final/data"

proj_meta = matread(data_path * "/OMM_1_meta.mat")["proj_meta"]["rd"][5]["act"][:, 6]

## ====

# animal, condition = 5, 4
# cond = proj_meta["rd"][animal]["Condition"]
# tp = findall(x -> x == 8.0, proj_meta["rd"][animal]["timepoint"])

# act = vcat(proj_meta["rd"][animal]["act"][:, 6]...)
act = vcat(proj_meta...)
inds = sortperm(mean(act, dims=2)[:])[end-10:end]

## ====

using DynamicalSystems
using MultivariateStats

function norm_std(x)
    μ, σ = mean(x), std(x)
    return (x .- μ) ./ σ
end


function get_embeddings(x)
    Ds, Ts, Es = [], [], []
    for cellind in eachrow(x)
        x_norm = norm_std(cellind)
        D, τ, e = optimal_separated_de(x_norm)
        push!(Ds, D)
        push!(Ts, τ)
        push!(Es, e)
    end
    return Ds, Ts, Es
end

@time Ds, Ts, Es = get_embeddings(act)

cs = 101, 36
using NearestNeighbors
kdtree = BallTree(Ds[101].data)
k = 10

Ds[101][100, :]

idxs, dists = knn(kdtree, Ds[101][100, :], k, true)

plot(Ds[101][:, 1], Ds[101][:, 2])
scatter!(Ds[101][idxs, 1], Ds[101][idxs, 2])

kdtree2 = BallTree(Ds[36].data)
idxs2, dists2 = knn(kdtree2, Ds[36][100, :], k, true)

d_ = Ds[36]
plot(d_[:, 1], d_[:, 2])
plot!(d_[idxs, 1], d_[idxs, 2])



begin
    ind = 120
    X = zeros(size(Ds[ind]))'
    for j in 1:size(X, 2)
        X[:, j] = Ds[ind].data[j]
    end
    M = fit(PCA, X)
    Yte = predict(M, X)
    plot(Yte[1, :], Yte[2, :])
end





# JLD2.save(datadir("exp_pro", "animal5_c4_delay_embedding.jld2"),
# Dict("Ds" => Ds, "Ts" => Ts, "Es" => Es))

begin
    D = Ds[1]
    ss = 1:3000
    # fig = Figure()
    # axs = Axis3D(fig[1, 1])
    lines(D[ss, 1], D[ss, 2], D[ss, 3])

    # fig
end


begin

    # ss = 1:4:length(D)
    # ss = 10_000:4:20_000
    ss = 10_000:20_000
    lines(D[ss, 1], D[ss, 2], D[ss, 3])
end



## =====

using LinearAlgebra, Flux, Statistics


function loss_false(code_batch::AbstractMatrix{T}, k::Int=1) where {T<:AbstractFloat}
    """
    An activity regularizer based on the False-Nearest-Neighbor
    Algorithm of Kennel, Brown, and Arbanel. Phys Rev A. 1992

    Parameters:
    - code_batch: Matrix
        (Batch size, Embedding Dimension) matrix of encoded inputs
    - k: Int 
        The number of nearest neighbors used to compute neighborhoods.
    """
    batch_size, n_latent = size(code_batch)

    # Fixed parameters
    rtol = 20.0
    atol = 2.0

    # Distance matrix calculation
    tri_mask = tril(ones(T, n_latent, n_latent), -1)
    batch_masked = tri_mask .* reshape(code_batch, (1, batch_size, n_latent))
    X_sq = sum(batch_masked .^ 2, dims=3)
    pdist_vector = X_sq .+ permutedims(X_sq, (1, 3, 2)) .- 2 .* batched_mul(batch_masked, permutedims(batch_masked, (1, 3, 2)))
    all_dists = pdist_vector

    # Average distances calculation
    all_ra = sqrt.((1 ./ (1:n_latent)) .* vec(sum(std(batch_masked, dims=2) .^ 2, dims=3)))

    # Clip distances to avoid singularities
    all_dists = clamp.(all_dists, 1e-14, maximum(all_dists))

    # Find k nearest neighbors
    _, inds = Flux.topk(-all_dists, k + 1, dims=3)

    # Gather neighbor distances
    neighbor_dists_d = gather(all_dists, inds)
    neighbor_new_dists = gather(all_dists[2:end, :, :], inds[1:end-1, :, :])

    # Calculate scaled distances
    scaled_dist = sqrt.((neighbor_new_dists .- neighbor_dists_d[1:end-1, :, :]) ./ neighbor_dists_d[1:end-1, :, :])

    # Apply FNN conditions
    is_false_change = scaled_dist .> rtol
    is_large_jump = neighbor_new_dists .> atol .* reshape(all_ra[1:end-1], (:, 1, 1))
    is_false_neighbor = is_false_change .| is_large_jump

    # Count false neighbors
    total_false_neighbors = convert.(Int, is_false_neighbor[:, :, 2:k+1])

    # Calculate regularization weights
    reg_weights = 1 .- mean(convert.(Float64, total_false_neighbors), dims=(2, 3))
    reg_weights = vcat(0, vec(reg_weights))

    # Calculate batch-averaged activations
    activations_batch_averaged = sqrt.(mean(code_batch .^ 2, dims=1))

    # Compute final loss
    loss = sum(reg_weights .* vec(activations_batch_averaged))

    convert(T, loss)
end

# Helper function to mimic TensorFlow's gather operation
function gather(x::AbstractArray, indices::AbstractArray)
    output = similar(x, size(indices)...)
    for i in CartesianIndices(indices)
        output[i] = x[indices[i], i[2], i[3]]
    end
    output
end

code_batch = cumsum(ones(Float32, 32, 6), dims=2)
T = Float32

batch_size, n_latent = size(code_batch)

# Fixed parameters
rtol = 20.0
atol = 2.0

using Flux: unsqueeze
using Tullio
# Distance matrix calculation
tri_mask = tril(ones(T, n_latent, n_latent), -1)

batch_masked = unsqueeze(tri_mask, 2) .* unsqueeze(code_batch, 1)
X_sq = sum(batch_masked .^ 2, dims=3)
pdist_vector = X_sq .+ permutedims(X_sq, (1, 3, 2)) .- 2 .* batched_mul(batch_masked, permutedims(batch_masked, (1, 3, 2)))

batched_mul(batch_masked, permutedims(batch_masked, (1, 3, 2)))

bm = permutedims(batch_masked, (1, 3, 2))
batched_mul(bm, permutedims(bm, (1, 3, 2)))


batch_masked
permutedims(batch_masked, (1, 3, 2))


all_dists = pdist_vector

# Average distances calculation
all_ra = sqrt.((1 ./ (1:n_latent)) .* vec(sum(std(batch_masked, dims=2) .^ 2, dims=3)))

# Clip distances to avoid singularities
all_dists = clamp.(all_dists, 1e-14, maximum(all_dists))

# Find k nearest neighbors
_, inds = Flux.topk(-all_dists, k + 1, dims=3)

# Gather neighbor distances
neighbor_dists_d = gather(all_dists, inds)
neighbor_new_dists = gather(all_dists[2:end, :, :], inds[1:end-1, :, :])

# Calculate scaled distances
scaled_dist = sqrt.((neighbor_new_dists .- neighbor_dists_d[1:end-1, :, :]) ./ neighbor_dists_d[1:end-1, :, :])

# Apply FNN conditions
is_false_change = scaled_dist .> rtol
is_large_jump = neighbor_new_dists .> atol .* reshape(all_ra[1:end-1], (:, 1, 1))
is_false_neighbor = is_false_change .| is_large_jump

# Count false neighbors
total_false_neighbors = convert.(Int, is_false_neighbor[:, :, 2:k+1])

# Calculate regularization weights
reg_weights = 1 .- mean(convert.(Float64, total_false_neighbors), dims=(2, 3))
reg_weights = vcat(0, vec(reg_weights))

# Calculate batch-averaged activations
activations_batch_averaged = sqrt.(mean(code_batch .^ 2, dims=1))

# Compute final loss
loss = sum(reg_weights .* vec(activations_batch_averaged))


loss_false(e, 1)
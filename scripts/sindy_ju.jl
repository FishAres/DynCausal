using DrWatson
@quickactivate "DynCausal"
using Flux, Zygote, CUDA
using DifferentialEquations
using LinearAlgebra, Statistics
using NPZ
using Plots
## ====

data = npzread(datadir("exp_pro", "v1_anim5_tp6_actwvelp.npy"))

## =====

# Define the SINDy autoencoder structure
struct SINDyAutoencoder
    encoder::Chain
    decoder::Chain
    dynamics::Matrix{Float32}
end

# Function to create the SINDy autoencoder
function create_sindy_autoencoder(input_dim, latent_dim, library_dim)
    encoder = Chain(
        Dense(input_dim, 64, relu),
        Dense(64, 32, relu),
        Dense(32, latent_dim)
    )

    decoder = Chain(
        Dense(latent_dim, 32, relu),
        Dense(32, 64, relu),
        Dense(64, input_dim)
    )

    dynamics = randn(Float32, library_dim, latent_dim)

    return SINDyAutoencoder(encoder, decoder, dynamics)
end

# Function to compute the library of candidate functions
function compute_library(z)
    library = [ones(Float32, size(z, 2))]
    for i in 1:size(z, 1)
        push!(library, z[i, :])
        for j in i:size(z, 1)
            push!(library, z[i, :] .* z[j, :])
            for k in j:size(z, 1)
                push!(library, z[i, :] .* z[j, :] .* z[k, :])
            end
        end
    end
    return hcat(library...)
end

# Loss function for the SINDy autoencoder
function sindy_loss(model, x, λ=0.1)
    z = model.encoder(x)
    x_recon = model.decoder(z)

    # Compute time derivatives
    dz = (z[:, 2:end] - z[:, 1:end-1]) ./ 0.01  # Assuming dt = 0.01

    # Compute library
    Θ = Zygote.ignore() do
        compute_library(z[:, 1:end-1])
    end

    # SINDy loss
    sindy_loss = sum(abs2, dz' - Θ * model.dynamics)

    # Reconstruction loss
    recon_loss = sum(abs2, x - x_recon)

    # Sparsity regularization
    sparsity_loss = λ * sum(abs, model.dynamics)

    return recon_loss + sindy_loss + sparsity_loss
end

# Function to generate example data (Lorenz system)
function generate_lorenz_data(n_samples, dt)
    function lorenz(du, u, p, t)
        σ, ρ, β = p
        du[1] = σ * (u[2] - u[1])
        du[2] = u[1] * (ρ - u[3]) - u[2]
        du[3] = u[1] * u[2] - β * u[3]
    end

    u0 = [1.0, 0.0, 0.0]
    tspan = (0.0, dt * (n_samples - 1))
    p = [10.0, 28.0, 8 / 3]
    prob = ODEProblem(lorenz, u0, tspan, p)
    sol = solve(prob, Tsit5(), dt=dt)
    return Array(sol)
end

# Training function
function train_sindy_autoencoder(model, data, n_epochs, learning_rate)
    opt = ADAM(learning_rate)

    for epoch in 1:n_epochs
        gs = gradient(Flux.params(model.encoder, model.decoder, model.dynamics)) do
            sindy_loss(model, data)
        end
        Flux.Optimise.update!(opt, Flux.params(model.encoder, model.decoder, model.dynamics), gs)

        if epoch % 100 == 0
            loss = sindy_loss(model, data)
            println("Epoch $epoch, Loss: $loss")
        end
    end
end

# Generate example data
n_samples = 1000
dt = 0.01
data = generate_lorenz_data(n_samples, dt)

# Create and train the SINDy autoencoder
input_dim = 3
latent_dim = 2
library_dim = 20  # Adjust based on the number of library functions

model = create_sindy_autoencoder(input_dim, latent_dim, library_dim)
train_sindy_autoencoder(model, data, 1000, 1e-3)

sindy_loss(model, data)

# After training, you can use the model to make predictions or analyze the learned dynamics

x = data

z = model.encoder(x)
x_recon = model.decoder(z)

plot(x_recon')
plot!(x')


# Compute time derivatives
dz = (z[:, 2:end] - z[:, 1:end-1]) ./ 0.01f0  # Assuming dt = 0.01

# Compute library
Θ = compute_library(z[:, 1:end-1])

Θ * model.dynamics


# SINDy loss
sum(abs2, dz' - Θ * model.dynamics)

# Reconstruction loss
recon_loss = sum(abs2, x - x_recon)

# Sparsity regularization
sparsity_loss = λ * sum(abs, model.dynamics)

## ======

using LinearAlgebra

function extract_delays_from_hankel(data, max_delay; num_delays=2)
    # Construct Hankel matrix
    m = length(data) - max_delay + 1
    H = zeros(m, max_delay)
    for i in 1:m
        H[i, :] = data[i:i+max_delay-1]
    end

    # Perform SVD
    U, S, V = svd(H)

    # Extract delays from the first few columns of U
    delays = []
    for col in 1:num_delays
        # Find peaks in the column
        peaks = findlocalmaxima(U[:, col])
        if !isempty(peaks)
            push!(delays, peaks[1] - 1)  # Subtract 1 because Julia uses 1-based indexing
        end
    end

    return delays
end

# Helper function to find local maxima
function findlocalmaxima(v)
    maxima = Int[]
    for i in 2:length(v)-1
        if v[i-1] < v[i] && v[i] > v[i+1]
            push!(maxima, i)
        end
    end
    return maxima
end

# Example usage
data = sin.(0.1 .* (1:1000)) .+ 0.1 .* randn(1000)  # Example time series
max_delay = 200
delays = extract_delays_from_hankel(data, max_delay; num_delays=3)
println("Extracted delays: ", delays)

findlocalmaxima(U[:, 2])

using Plots

function pareto_analysis(S, threshold=0.95)
    cumulative_sum = cumsum(S) / sum(S)
    n_components = findfirst(cumulative_sum .>= threshold)

    plot(cumulative_sum, marker=:circle, label="Cumulative Variance")
    plot!(xlabel="Number of Components", ylabel="Cumulative Variance Explained")
    vline!([n_components], label="Threshold at $threshold")
    title!("Pareto Analysis of Singular Values")

    return n_components
end

# Usage in the context of the Hankel matrix SVD
function analyze_hankel_svd(data, max_delay)
    # Construct Hankel matrix
    m = length(data) - max_delay + 1
    H = zeros(m, max_delay)
    for i in 1:m
        H[i, :] = data[i:i+max_delay-1]
    end

    # Perform SVD
    U, S, V = svd(H)

    # Perform Pareto analysis
    n_components = pareto_analysis(S)

    println("Number of significant components: ", n_components)

    # Return the first n_components columns of U as the significant eigen-time-delays
    return U[:, 1:n_components]
end

# Example usage
data = sin.(0.1 .* (1:1000)) .+ 0.1 .* randn(1000)  # Example time series
max_delay = 100
significant_eigen_time_delays = analyze_hankel_svd(data, max_delay)
import torch
import matplotlib.pyplot as plt
from cheetah.particles import ParticleBeam
from gpsr.analysis import compute_fractional_emittance_curve

def compute_halo_metric(
    distribution,
    slice_,
    visualize = False,
):
    """
    Compute halo metric, which is calculated as the 
    normalized total error between the distribution
    fractional emittance curve and a fractional emittance
    curve for a gaussian distribution.
    """
    
    benchmark_beam = ParticleBeam.from_twiss(
        emittance_x = torch.tensor(1e-8),
        emittance_y = torch.tensor(1e-8),
        beta_x = torch.tensor(1.0),
        beta_y = torch.tensor(1.0),
        sigma_p = torch.tensor(1.0),
        sigma_tau = torch.tensor(1.0),
    )
    
    fractions = torch.linspace(0.5,1.0,100)
    emit = compute_fractional_emittance_curve(fractions, distribution, slice_)
    gaussian = compute_fractional_emittance_curve(fractions, benchmark_beam, slice_)

    if visualize:
        fig,ax = plt.subplots()
        ax.plot(fractions, emit / emit[0], label="distribution")
        ax.plot(fractions, gaussian / gaussian[0], label="gaussian")
        ax.legend()
        ax.set_xlabel("Beam fraction")
        ax.set_ylabel(r"$\epsilon / \epsilon_{50\%}$")

    return (emit / emit[0] - gaussian / gaussian[0]).abs().sum() / (gaussian / gaussian[0]).sum()
    

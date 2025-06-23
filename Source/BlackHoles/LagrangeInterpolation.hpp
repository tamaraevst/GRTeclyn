#ifndef LAGRANGEINTERPOLATION_HPP_
#define LAGRANGEINTERPOLATION_HPP_

#include <AMReX_Gpu.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>
#include <AMReX_Array4.H>
#include "AMReX_LOUtil_K.H"
#include <cmath>

namespace amrex {

template <typename P>
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void lagrange4_interpolate_to_particle(
    const P& p, //pointer to the physical coordinates of the particle
    amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& plo, //position in the cell
    amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxi, // 1/dx, 1/dy and so on
    const Array4<amrex::Real const>* data_arr, //data we are iterpolating from
    amrex::ParticleReal * val, //here we store the interpolated values
    const amrex::IntVect* is_nodal, //nodal or cell-centered?
    int start_comp, int ncomp, int num_arrays) //starting components, number of components of the field data and an option to interpolate from several fields at the same time
{
	
// Stencil nodes for uniform 4th-order Lagrange 
constexpr int N = 5;
constexpr int radius = N / 2;
constexpr amrex::Real x_stencil[N] = {-2., -1., 0., 1., 2.};

int ctr = 0;

// Iterate over num_arrays
 for (int d = 0; d < num_arrays; ++d)
{
    AMREX_D_TERM(amrex::Real lx = (Real(p.pos(0))-plo[0])*dxi[0] - static_cast<Real>(!is_nodal[d][0])*Real(0.5);,
                     amrex::Real ly = (Real(p.pos(1))-plo[1])*dxi[1] - static_cast<Real>(!is_nodal[d][1])*Real(0.5);,
                     amrex::Real lz = (Real(p.pos(2))-plo[2])*dxi[2] - static_cast<Real>(!is_nodal[d][2])*Real(0.5));

    //(i0, j0, k0) -- lower corner of the box needed for interpolation 
    AMREX_D_TERM(int i0 = static_cast<int>(amrex::Math::floor(lx));,
                 int j0 = static_cast<int>(amrex::Math::floor(ly));,
                 int k0 = static_cast<int>(amrex::Math::floor(lz)));

    AMREX_D_TERM(amrex::Real const xint = lx - static_cast<Real>(i0);,
                     amrex::Real const yint = ly - static_cast<Real>(j0);,
                     amrex::Real const zint = lz - static_cast<Real>(k0));

    // Compute interpolation weights using poly_interp_coeff
    amrex::Real wx[N], wy[N], wz[N]; //weights along x,y,z
    poly_interp_coeff(xint, x_stencil, N, wx);
    
#if AMREX_SPACEDIM >= 2
        poly_interp_coeff(yint, x_stencil, N, wy);
#endif
#if AMREX_SPACEDIM == 3
        poly_interp_coeff(zint, x_stencil, N, wz);
#endif

//Iterate over components
for (int comp = start_comp; comp < start_comp+ncomp; ++comp)
        {
            val[ctr] = ParticleReal(0.0);

#if AMREX_SPACEDIM == 1
            for (int i = -radius; i <= radius; ++i)
            {
                val[ctr] += wx[i + radius] *
                       (data_arr[d])(i0 + i, 0, 0, comp);
            }
#elif AMREX_SPACEDIM == 2
            for (int j = -radius; j <= radius; ++j)
            for (int i = -radius; i <= radius; ++i)
            {
                val[ctr] += wx[i + radius] * wy[j + radius] *
                       (data_arr[d])(i0 + i, j0 + j, 0, comp);
            }
#elif AMREX_SPACEDIM == 3
            for (int k = -radius; k <= radius; ++k)
            for (int j = -radius; j <= radius; ++j)
            for (int i = -radius; i <= radius; ++i)
            {
                val[ctr] += wx[i + radius] * wy[j + radius] * wz[k + radius] *
                       (data_arr[d])(i0 + i, j0 + j, k0 + k, comp);
            }
#endif

            ctr++;
        }
    }
} 
}

#endif /* LAGRANGEINTERPOLATION_HPP_ */

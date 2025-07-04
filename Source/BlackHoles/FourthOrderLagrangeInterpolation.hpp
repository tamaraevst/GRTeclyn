#ifndef FOURTHORDERLAGRANGEINTERPOLATION_HPP_
#define FOURTHORDERLAGRANGEINTERPOLATION_HPP_

#include <AMReX_Gpu.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>
#include <AMReX_Array4.H>
#include "AMReX_LOUtil_K.H"
#include <cmath>

//Class for 4th order interpolation of the mesh data onto the particle using Lagrange polynomials. 
//Currently, it allows to interpolate only one field at a time.
//Assumes uniform grids 

class FourthOrderLagrangeInterpolator
{

private:    
    static constexpr int N = 5; //number of stencil points
    inline static constexpr amrex::Real stencil[N] = {-2, -1, 0, 1, 2};
    //inline static constexpr amrex::Real stencil[N] = {0., 1.};
    double i0, j0, k0; // indices of the lower left corner of the stencil in the grid

public:
    amrex::Real wx[N], wy[N], wz[N]; //where we store the weights for each dimension

    FourthOrderLagrangeInterpolator() {};

    template <typename P>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void compute_weights(
        const P& p,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& plo,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dxi,
        const amrex::IntVect& is_nodal)
    {

        // Compute the grid index of the position
        AMREX_D_TERM(amrex::Real lx = (amrex::Real(p.pos(0)) - plo[0]) * dxi[0] - static_cast<amrex::Real>(!is_nodal[0]) * amrex::Real(0.5);,
                     amrex::Real ly = (amrex::Real(p.pos(1)) - plo[1]) * dxi[1] - static_cast<amrex::Real>(!is_nodal[1]) * amrex::Real(0.5);,
                     amrex::Real lz = (amrex::Real(p.pos(2)) - plo[2]) * dxi[2] - static_cast<amrex::Real>(!is_nodal[2]) * amrex::Real(0.5););

        std::cout << "lx: " << lx << ", ly: " << ly << ", lz: " << lz << std::endl;
        std::cout << "plo[0]: " << plo[0] << ", plo[1]: " << plo[1] << ", plo[2]: " << plo[2] << std::endl;

        // // Compute the floor for the index of the position
        // AMREX_D_TERM(int i0 = static_cast<int>(amrex::Math::floor(lx));,
        //              int j0 = static_cast<int>(amrex::Math::floor(ly));,
        //              int k0 = static_cast<int>(amrex::Math::floor(lz)););

        // Shift i0, j0, k0 back by 2 to start from the lower-left of the 5-point centered stencil
        i0 = static_cast<int>(amrex::Math::floor(lx)) - 2;
        j0 = static_cast<int>(amrex::Math::floor(ly)) - 2;
        k0 = static_cast<int>(amrex::Math::floor(lz)) - 2;


        std::cout << "i0: " << i0 << ", j0: " << j0 << ", k0: " << k0 << std::endl;

        // // Compute the position w.r.t. to the lower corner
        // AMREX_D_TERM(amrex::Real xint = lx - static_cast<amrex::Real>(i0);,
        //              amrex::Real yint = ly - static_cast<amrex::Real>(j0);,
        //              amrex::Real zint = lz - static_cast<amrex::Real>(k0););

        // xint should be relative to the **center** of the stencil (i0 + 2)
        amrex::Real xint = lx - (i0 + 2);
        amrex::Real yint = ly - (j0 + 2);
        amrex::Real zint = lz - (k0 + 2);

        std::cout << "xint: " << xint << ", yint: " << yint << ", zint: " << zint << std::endl;

        amrex::poly_interp_coeff(xint, stencil, N, wx);

        std::cout << "wx: " << wx[0] << ", " << wx[1] << ", " << wx[2] << ", " << wx[3] << ", " << wx[4] << std::endl;

#if AMREX_SPACEDIM >= 2
        amrex::poly_interp_coeff(yint, stencil, N, wy);
        std::cout << "wy: " << wy[0] << ", " << wy[1] << ", " << wy[2] << ", " << wy[3] << ", " << wy[4] << std::endl;
#endif
#if AMREX_SPACEDIM == 3
        amrex::poly_interp_coeff(zint, stencil, N, wz);
        std::cout << "wz: " << wz[0] << ", " << wz[1] << ", " << wz[2] << ", " << wz[3] << ", " << wz[4] << std::endl;
#endif
    }

    // Function to perform the interpolation 
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void interpolate(const amrex::Array4<amrex::Real const>* data_arr,
        amrex::ParticleReal* val,
        int start_comp,
        int ncomp) const
    {
        int ctr = 0;
        auto const& data = data_arr[0];

        for (int comp = start_comp; comp < start_comp + ncomp; ++comp) {
            val[ctr] = amrex::ParticleReal(0.0);
#if AMREX_SPACEDIM == 3
            for (int kk = 0; kk < N; ++kk) {
#endif
#if AMREX_SPACEDIM >= 2
                for (int jj = 0; jj < N; ++jj) {
#endif
                    for (int ii = 0; ii < N; ++ii) {
                        val[ctr] += data(amrex::IntVect(AMREX_D_DECL(i0 + ii, j0 + jj, k0 + kk)), comp) * AMREX_D_TERM(wx[ii], * wy[jj], * wz[kk]);
                        }
#if AMREX_SPACEDIM >= 2
                    }
#endif
#if AMREX_SPACEDIM == 3
                }
#endif
                ++ctr;
            } // end of for comp loop

    std::cout << "z at i = " << i0+0 << " j = " << j0+0 << " k = " << k0+0 << " : " << data(amrex::IntVect(AMREX_D_DECL(i0 + 0, j0 + 0, k0 + 0)), 0) << std::endl;
    std::cout << "z at i = " << i0+0 << " j = " << j0+0 << " k = " << k0+1 << " : " << data(amrex::IntVect(AMREX_D_DECL(i0 + 0, j0 + 0, k0 + 1)), 0) << std::endl;
    std::cout << "z at i = " << i0+0 << " j = " << j0+0 << " k = " << k0+2 << " : " << data(amrex::IntVect(AMREX_D_DECL(i0 + 0, j0 + 0, k0 + 2)), 0) << std::endl;
    std::cout << "z at i = " << i0+0 << " j = " << j0+0 << " k = " << k0+3 << " : " << data(amrex::IntVect(AMREX_D_DECL(i0 + 0, j0 + 0, k0 + 3)), 0) << std::endl;
    std::cout << "z at i = " << i0+0 << " j = " << j0+0 << " k = " << k0+4 << " : " << data(amrex::IntVect(AMREX_D_DECL(i0 + 0, j0 + 0, k0 + 4)), 0) << std::endl;
    }

};

#endif /* FOURTHORDERLAGRANGEINTERPOLATION_HPP_ */

#ifndef LINEARINTERPOLATION_HPP_
#define LINEARINTERPOLATION_HPP_

#include <AMReX_Gpu.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>
#include <AMReX_Array4.H>
#include "AMReX_LOUtil_K.H"
#include <cmath>

//Linear interpolation for testing

class LinearInterpolator
{

private:    
    static constexpr int N = 2; //number of stencil points
    //inline static constexpr amrex::Real stencil[N] = {-2., -1., 0., 1., 2.};
    inline static constexpr amrex::Real stencil[N] = {0., 1.};
    int i0, j0, k0; // indices of the lower left corner of the stencil in the grid

public:
    amrex::Real wx[N], wy[N], wz[N]; //where we store the weights for each dimension

    LinearInterpolator() {};

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

        // Compute the floor of the position
        AMREX_D_TERM(int i0_floor = static_cast<int>(amrex::Math::floor(lx));,
                     int j0_floor = static_cast<int>(amrex::Math::floor(ly));,
                     int k0_floor = static_cast<int>(amrex::Math::floor(lz)););
	
	AMREX_D_TERM(i0 = i0_floor;,
                     j0 = j0_floor;,
                     k0 = k0_floor;);

        std::cout << "i0: " << i0 << ", j0: " << j0 << ", k0: " << k0 << std::endl;
        
        // Compute the position w.r.t. to the lower corner
        AMREX_D_TERM(amrex::Real xint = lx - static_cast<amrex::Real>(i0_floor);,
                     amrex::Real yint = ly - static_cast<amrex::Real>(j0_floor);,
                     amrex::Real zint = lz - static_cast<amrex::Real>(k0_floor););

        amrex::Real sx[] = {amrex::Real(1.0) - xint, xint};
        amrex::Real sy[] = {amrex::Real(1.0) - yint, yint};
        amrex::Real sz[] = {amrex::Real(1.0) - zint, zint};

        std::cout << "lx: " << lx << ", ly: " << ly << ", lz: " << lz << std::endl;
        std::cout << "xint: " << xint << ", yint: " << yint << ", zint: " << zint << std::endl;

        amrex::poly_interp_coeff(xint, stencil, N, wx);

        // std::cout << "sx: " << sx[0] << ", " << sx[1] << std::endl;
        // std::cout << "wx: " << wx[0] << ", " << wx[1] << std::endl;
        std::cout << "Abs value of x : " << fabs(wx[0] - sx[0]) << " " << fabs(wx[1] - sx[1]) << std::endl;

#if AMREX_SPACEDIM >= 2
        amrex::poly_interp_coeff(yint, stencil, N, wy);
        // std::cout << "sy: " << sy[0] << ", " << sy[1] << std::endl;
        // std::cout << "wy: " << wy[0] << ", " << wy[1] << std::endl;
        std::cout << "Abs value of y : " << fabs(wy[0] - sy[0]) << " " << fabs(wy[1] - sy[1]) << std::endl;
#endif
#if AMREX_SPACEDIM == 3
        amrex::poly_interp_coeff(zint, stencil, N, wz);
        // std::cout << "sz: " << sz[0] << ", " << sz[1] << std::endl;
        // std::cout << "wz: " << wz[0] << ", " << wz[1] << std::endl;
        std::cout << "Abs value of z : " << fabs(wz[0] - sz[0]) << " " << fabs(wz[1] - sz[1]) << std::endl;
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
	        std::cout << "val[" << ctr << "] = " << val[ctr] << std::endl;
                ++ctr;
            } // end of for comp loop

    // std::cout << "z at i = " << i0+0 << " j = " << j0+0 << " k = " << k0+0 << " : " << data(amrex::IntVect(AMREX_D_DECL(i0 + 0, j0 + 0, k0 + 0)), 0) << std::endl;
    // std::cout << "z at i = " << i0+0 << " j = " << j0+0 << " k = " << k0+1 << " : " << data(amrex::IntVect(AMREX_D_DECL(i0 + 0, j0 + 0, k0 + 1)), 0) << std::endl;
    }

};

#endif /* LINEARINTERPOLATION_HPP_ */

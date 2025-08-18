#ifndef PARTICLEINTERPOLATORS_HPP_
#define PARTICLEINTERPOLATORS_HPP_

#include <AMReX_Array.H>
#include <AMReX_Particles.H>
#include <AMReX_AmrParGDB.H>
#include <AMReX_AmrLevel.H>
#include <AMReX_ParallelDescriptor.H>

#include "GRAMR.hpp"
#include "LagrangeInterpolation.hpp"
#include "InterpolationQuery.hpp"        
#include "BoundaryConditions.hpp"

// This class interpolates one variable (possibly multi-component) at arbitrary
// coordinates provided via InterpolationQuery, using AMReX Particles.
//
// Layout:
// - NStructReal = 0, NStructInt = 1   (idata(0) = query point index)
// - NArrayReal  = AMREX_SPACEDIM      (SoA slots used to store up to AMREX_SPACEDIM comps)
// - NArrayInt   = 0
//
// Usage (typical):
//   ParticleInterpolators interp(bc_params, c_chi, /*ncomp=*/1);
//   interp.set_gramr_ptr(&gr);
//   interp.populate_from_query(query);   // seeds particles (rank 0) then Redistribute()
//   interp.interpolate_to_particle();          // fills SoA with interpolated values (all ranks)
//   interp.interp(query);                // MPI-reduces and writes into query outs
//
class ParticleInterpolators
  : public amrex::ParticleContainer</*NStructReal*/0,  // for positions
                                    /*NStructInt*/1,               // particle index
                                    /*NArrayReal*/AMREX_SPACEDIM,  // SOA slots to store interpolated values, cannot have more than AMREX_SPACEDIM for one variable 
                                    /*NArrayInt*/0>
{
  private:
    GRAMR* m_gr_amr{nullptr};
    bool   m_initialized{false};
    int    m_start_comp{0};                 // first component 
    int    m_ncomp{1};                      // number of components 

    // physical domain corners on level 0 for parity logic
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_prob_lo{};
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_prob_hi{};

    // reflective BC flags per side on the low and hihh sides
    std::array<bool, AMREX_SPACEDIM> m_lo_boundary_reflective{ {false} };
    std::array<bool, AMREX_SPACEDIM> m_hi_boundary_reflective{ {false} };

    // copy of BC params 
    BoundaryConditions::params_t m_bc_params{};

  public:
    using amrex::ParticleContainer<0,1,AMREX_SPACEDIM,0>::ParticleContainer;

    ParticleInterpolators(const BoundaryConditions::params_t& a_bc_params,
                          int a_start_comp, int a_ncomp)
      : m_gr_amr(nullptr)
      , m_initialized(false)
      , m_start_comp(a_start_comp)
      , m_ncomp(a_ncomp)
      , m_bc_params(a_bc_params)
    {}

    // initialise everything and perform some sanity checks
    void set_gramr_ptr(GRAMR* gr_amr_ptr)
    {
      // is GRAMR properly set?
      AMREX_ASSERT(gr_amr_ptr != nullptr);
      m_gr_amr = gr_amr_ptr;

      this->Define(dynamic_cast<amrex::ParGDBBase*>(m_gr_amr->GetParGDB()));
      m_initialized = true;

      AMREX_ALWAYS_ASSERT(m_ncomp >= 1 && m_ncomp <= AMREX_SPACEDIM);

      // read in the physical bounds for reflective BC checks (it is sufficient to do this on lev = 0)
      const amrex::Geometry& geom0 = m_gr_amr->getLevel(0).Geom();
      const auto plo = geom0.ProbLoArray();
      const auto phi = geom0.ProbHiArray();
      for (int d = 0; d < AMREX_SPACEDIM; ++d) {
          m_prob_lo[d] = plo[d];
          m_prob_hi[d] = phi[d];
      }

      // set the reflective flags from BC params
      for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        m_lo_boundary_reflective[dir] =
            (m_bc_params.lo_boundary[dir] == BoundaryConditions::REFLECTIVE_BC);
        m_hi_boundary_reflective[dir] =
            (m_bc_params.hi_boundary[dir] == BoundaryConditions::REFLECTIVE_BC);
      }
    }

    // a parity helper (the same way as it was defined in the AMRInterpolator)
    int get_state_var_parity(int comp,
                             int point_idx,
                             const InterpolationQuery& query,
                             const Derivative& deriv) const
    {
        int parity = 1;
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
        {
            // get the coords
            const double x = query.m_coords[dir][point_idx];

            // check where we are w.r.t to the prob domain
            const bool beyond_lo = (m_lo_boundary_reflective[dir] && x <  m_prob_lo[dir]);
            const bool beyond_hi = (m_hi_boundary_reflective[dir] && x >  m_prob_hi[dir]);

            if (beyond_lo || beyond_hi)
            {
                parity *= BoundaryConditions::get_state_var_parity(comp, dir); // Is there a boundary conditions function for derived vars? TODO!!!
                // invert parity for first derivatives
                if (deriv[dir] == 1) 
                  parity *= -1;
            }
        }
        return parity;
    }

    // a function to reflect a particle back into the valid domain, when symmetry BCs are used 
    amrex::Real reflect_particle(amrex::Real x,
                                 amrex::Real lo, amrex::Real hi,
                                 amrex::Real dx_half,
                                 bool lo_reflect, bool hi_reflect) const
    {
        // enforce a new particle position if needed
        amrex::Real xl = x;
        if (lo_reflect && xl < lo)         
        {
          xl = lo + (lo - xl);   // reflect across lo
          // amrex::Print() << "Particle position " << x << " reflected to " << xl << " across lo boundary.\n";
        }
        if (hi_reflect && xl > hi)
        {
          xl = hi - (xl - hi);   // reflect across hi
          // amrex::Print() << "Particle position " << x << " reflected to " << xl << " across hi boundary.\n";
        }

        // what to do at the boundary? TODO!!
        // xl = amrex::max(xl, lo + dx_half);
        // xl = amrex::min(xl, hi - dx_half);
        return xl;
    }


    // allocate particles at the query points
    void populate_from_query(const InterpolationQuery& query)
{
    AMREX_ASSERT(m_initialized);

    //////////////////////////////////
    // this is not used so far. TODO!!
    const auto& geom0 = m_gr_amr->getLevel(0).Geom();
    const auto dx     = geom0.CellSizeArray();

    const amrex::Real hx2 = 0.5*dx[0];
#if AMREX_SPACEDIM >= 2
    const amrex::Real hy2 = 0.5*dx[1];
#endif
#if AMREX_SPACEDIM == 3
    const amrex::Real hz2 = 0.5*dx[2];
#endif
    //////////////////////////////////

    const int lev = 0;
    const int n   = static_cast<int>(query.m_num_points);

    // // find a local (grid,tile) on this rank
    // int grid = -1, tile = -1;
    // for (amrex::MFIter mfi = this->MakeMFIter(lev); mfi.isValid(); ++mfi) {
    //     grid = mfi.index();
    //     tile = mfi.LocalTileIndex();
    //     break;
    // }
    // if (grid < 0) {
    //     amrex::AllPrint() << "[populate] rank " << amrex::ParallelDescriptor::MyProc()
    //                       << " has no local tiles at lev " << lev << " — skipping.\n";
    //     return;
    // }

    // it does not matter on which grid/tile we place the particles initially as long as we call Redistribute() after.
    auto& ptile = this->DefineAndReturnParticleTile(lev, 0, 0);
    ptile.resize(n);
    const auto& particle_data = ptile.getParticleTileData();

    // get coords from query
    const double* x = query.m_coords[0];
    const double* y = query.m_coords[1];
#if AMREX_SPACEDIM == 3
    const double* z = query.m_coords[2];
#endif

    // loop over particles and place them at the required points
    amrex::ParallelFor(n, [=] AMREX_GPU_DEVICE(int ip)
    {
        auto& p = particle_data[ip];   // this will grow SOA
        p.id()  = ip + 1; // particle id starts from 1
        p.cpu() = 0; // CPU id, not used here

        // reflect into valid interior for seeding
        const amrex::Real xr = reflect_particle(static_cast<amrex::Real>(x[ip]),
                                            m_prob_lo[0], m_prob_hi[0], hx2,
                                            m_lo_boundary_reflective[0],
                                            m_hi_boundary_reflective[0]);
        const amrex::Real yr = reflect_particle(static_cast<amrex::Real>(y[ip]),
                                            m_prob_lo[1], m_prob_hi[1], hy2,
                                            m_lo_boundary_reflective[1],
                                            m_hi_boundary_reflective[1]);
#if AMREX_SPACEDIM == 3
        const amrex::Real zr = reflect_particle(static_cast<amrex::Real>(z[ip]),
                                            m_prob_lo[2], m_prob_hi[2], hz2,
                                            m_lo_boundary_reflective[2],
                                            m_hi_boundary_reflective[2]);
#endif
        // set position 
        p.pos(0) = xr;
        p.pos(1) = yr;
    #if AMREX_SPACEDIM == 3
        p.pos(2) = zr;
    #endif
        p.idata(0) = ip;
    });

    // set zero SOA slots for now: it did not work inside GPU loop. TODO!!!
    {
        auto soa = ptile.getParticleTileData();
        const int np = ptile.numParticles();
        for (int s = 0; s < AMREX_SPACEDIM; ++s) {
            auto* rs = soa.rdata(s);
            for (int i = 0; i < np; ++i) rs[i] = 0.0;
        }
    }

    // // some debugging stuff 
    // long np_tile = ptile.numParticles();
    // long np_lev  = this->NumberOfParticlesAtLevel(lev);
    // amrex::AllPrint() << "on rank " << amrex::ParallelDescriptor::MyProc()
    //                   << " tile count=" << np_tile
    //                   << " level count=" << np_lev << "\n";

    this->Redistribute();

    // // more debugs
    // long local_total = 0;
    // for (int L = 0; L <= this->finestLevel(); ++L)
    //     local_total += this->NumberOfParticlesAtLevel(L);
    // long global_total = local_total;
    // // amrex::ParallelDescriptor::ReduceLongSum(global_total);

    // if (amrex::ParallelDescriptor::IOProcessor()) {
    //     amrex::Print() << "have global #particles after Redistribute = "
    //                    << global_total << "\n";
    // }
}

    // interpolate variables into SOA slots
    void interpolate_to_particle()
{
    AMREX_ASSERT(m_initialized);
    AMREX_ALWAYS_ASSERT(m_ncomp >= 1 && m_ncomp <= AMREX_SPACEDIM);
    AMREX_ASSERT(m_start_comp >= 0);

    for (int lev = 0; lev <= m_gr_amr->finestLevel(); ++lev)
    {
        // if particles are only on the coarsest level, skip the looping over levels
        if (this->NumberOfParticlesAtLevel(lev) == 0) continue;

        amrex::AmrLevel&       level = m_gr_amr->getLevel(lev);
        const amrex::Geometry& geom  = level.Geom();
        amrex::MultiFab&       state = level.get_new_data(State_Type);

        AMREX_ASSERT(m_start_comp + m_ncomp <= state.nComp());

        amrex::IntVect nghost(AMREX_D_DECL(2,2,2));
        state.FillBoundary(m_start_comp, m_ncomp, nghost, geom.periodicity());

        const auto plo = geom.ProbLoArray();
        const auto dxi = geom.InvCellSizeArray();

        // loop over tiles and interpolate now
        for (ParIterType it(*this, lev); it.isValid(); ++it)
        {
            auto& ptile = this->ParticlesAt(lev, it);
            auto  ptd   = ptile.getParticleTileData();  
            const int np = it.numParticles();
            auto  fab_array = state[it].const_array();  

            amrex::ParallelFor(np, [=] AMREX_GPU_DEVICE (int ip) {

                auto& sp = ptd[ip];

                amrex::IntVect is_nodal = amrex::IntVect::TheZeroVector();
                LagrangeInterpolator<5> interp;
                interp.compute_weights(sp, plo, dxi, is_nodal);

                amrex::ParticleReal vals[AMREX_SPACEDIM];
                interp.interpolate(&fab_array, vals, m_start_comp, m_ncomp);

                // write results to SOA
                for (int k = 0; k < m_ncomp; ++k) {
                    ptd.rdata(k)[ip] = vals[k];
                }
            });
        }

        // synchronize GPU streams to ensure all particles are updated
        amrex::Gpu::streamSynchronize();
    }
}

    // mirror of AMRInterpolator::interp(); assembles all particle data and writes parity * value into the query out arrays
    void interp(InterpolationQuery& query)
    {
      AMREX_ASSERT(m_initialized);
      // get total query points here
      const int npts = static_cast<int>(query.numPoints());

      // value_at_point[k][ip] : k in [0..m_ncomp-1]
      std::vector<std::vector<double>> value_at_point(m_ncomp, std::vector<double>(npts, 0.0));
      std::vector<int> have(npts, 0);

      long local = 0;
      // gather all particle values
      for (int lev = 0; lev <= this->finestLevel(); ++lev)
      {
        for (ParIterType it(*this, lev); it.isValid(); ++it)
        {
          local += it.numParticles();

          const auto& ptile = this->ParticlesAt(lev, it);
          const auto  aos   = ptile.GetArrayOfStructs();
          const auto* P     = aos.data();
          const auto  soa   = ptile.getConstParticleTileData(); // read-only SoA
          const int   np    = it.numParticles();

          for (int i = 0; i < np; ++i)
          {
            auto& p = P[i];
            const int q = p.idata(0);                
            if (0 <= q && q < npts) {
              for (int k = 0; k < m_ncomp; ++k) {
                value_at_point[k][q] = static_cast<double>(soa.rdata(k)[i]);
              }
              have[q] = 1; // mark that we have a value for this point
            }
          }
        }
      }

      amrex::AllPrint() << "rank " << amrex::ParallelDescriptor::MyProc()
                  << " holds " << local << " particles\n";

      // reduce across ranks 
      for (int k = 0; k < m_ncomp; ++k) {
        amrex::ParallelDescriptor::ReduceRealSum(value_at_point[k].data(), npts);
      }
      amrex::ParallelDescriptor::ReduceIntSum(have.data(), npts);

      // loop for each component
      for (auto deriv_it = query.compsBegin(); deriv_it != query.compsEnd(); ++deriv_it)
      {
        using comps_t = std::vector<typename InterpolationQuery::out_t>;
        comps_t& comps = deriv_it->second;

        for (auto& entry : comps)
        {
          const int       comp   = std::get<0>(entry);
          double*         out    = std::get<1>(entry); // this is where interpolated values are written into 
          const Derivative& dkey = deriv_it->first;

          const int k = comp - m_start_comp;     // reindex the variable component from 0
          AMREX_ALWAYS_ASSERT(k >= 0 && k < m_ncomp);

          for (int ip = 0; ip < npts; ++ip)
          {
            if (!have[ip]) 
            {
              amrex::Abort("interp(): no data for query point " + std::to_string(ip));
            }

            const int parity = get_state_var_parity(comp, ip, query, dkey);
            const double v   = have[ip] ? value_at_point[k][ip] : 0.0;
            out[ip] = parity * v;
          }
        }
      }
    }
};

#endif // PARTICLEINTERPOLATORS_HPP_

/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#ifndef SPHERICALPARTICLES_HPP_
#define SPHERICALPARTICLES_HPP_

#include <AMReX_Array.H>
#include <AMReX_Particles.H>
#include <AMReX_AmrParGDB.H>
#include <AMReX_AmrLevel.H>

#include "GRAMR.hpp"
#include "GRAMRLevel.hpp"
#include "SphericalGeometry.hpp"
#include "LagrangeInterpolation.hpp"
#include "StateVariables.hpp"
#include "SmallDataIO.hpp"
#include "Parameters.hpp"

// Recall that the ParticleContainer is templeted over <NStructReal, NStructInt, NArrayReal, NArrayInt>, where 
// NStructReal	The number of extra Real components in the particle struct
// NStructInt	The number of extra integer components in the particle struct
// NArrayReal	The number of extra Real components stored in struct-of-array form
// NArrayInt	The number of extra integer components stored in struct-of-array form

class SphericalParticles : public amrex::ParticleContainer</*NStructReal*/3,
                                                          /*NStructInt*/0,
                                                          /*NArrayReal*/0,
                                                          /*NArrayInt*/0>
{ 
protected:
    GRAMR* m_gr_amr{nullptr};
    bool m_initialized{false}; // for a check whether GRAMR is set properly
    const amrex::Real m_dt{};
    const amrex::Real m_time{};
    const amrex::Real m_restart_time{};
    Spherical_params_t m_params;
    SphericalGeometry m_sph_geom;
    int m_start_comp;
    int m_ncomp; 
 
public:
    using amrex::ParticleContainer<3, 0, 0, 0>::ParticleContainer;

    SphericalParticles(const Spherical_params_t& a_params, int a_start_comp, int a_ncomp, amrex::Real a_dt, amrex::Real a_time, amrex::Real a_restart_time)
        : m_params(a_params), m_start_comp(a_start_comp), m_ncomp(a_ncomp), m_sph_geom(m_params.extraction_center), m_dt(a_dt), m_time(a_time), m_restart_time(a_restart_time) {}

    void set_gramr_ptr(GRAMR* gr_amr_ptr)
    {
        m_gr_amr = gr_amr_ptr;
        m_initialized = true;
        
        Define(dynamic_cast<amrex::ParGDBBase*>(m_gr_amr->GetParGDB()));
    }
 
    void initialize_particles_on_sphere()
    {
        AMREX_ASSERT(m_initialized);
        AMREX_ASSERT(m_gr_amr != nullptr);

        // Do stuff on rank 0 only
        if (amrex::ParallelDescriptor::MyProc() != 0) return;
 
        const int lev = 0; 
        int total_particles = m_params.num_points_theta * m_params.num_points_phi;

        // It does not matter on which grid/tile we place the particles initially as long as we call Redistribute() after.
        auto& particle_tile = DefineAndReturnParticleTile(lev, 0, 0);
        particle_tile.resize(total_particles);
        const auto& particle_data = particle_tile.getParticleTileData();

        // Loop over particles and place them at required (u, v) points
        amrex::ParallelFor(total_particles, [=] AMREX_GPU_DEVICE(int ip)
        {
            int i_theta = ip / m_params.num_points_phi;
            int i_phi = ip % m_params.num_points_phi;
            double theta = SphericalGeometry::u(i_theta, m_params.num_points_theta);
            double phi = SphericalGeometry::v(i_phi, m_params.num_points_phi);

            auto& p = particle_data[ip];
            p.id() = ip + 1;
            p.cpu() = 0;
 
            p.pos(0) = m_sph_geom.get_grid_coord(0, m_params.extraction_radii, theta, phi);
            p.pos(1) = m_sph_geom.get_grid_coord(1, m_params.extraction_radii, theta, phi);
#if AMREX_SPACEDIM == 3
            p.pos(2) = m_sph_geom.get_grid_coord(2, m_params.extraction_radii, theta, phi);
#endif
            p.rdata(0) = 0.0; // this is for \chi (will be filled in later)
            p.rdata(1) = theta;
            p.rdata(2) = phi;
        });

        // std::cout << "Total number of particles after init " << TotalNumberOfParticles() << std::endl;
 
        amrex::Gpu::streamSynchronize();
        Redistribute();

        // std::cout << "Total number of particles after init " << TotalNumberOfParticles() << std::endl;

    }

    // Interpolation function (for now the interpolation is hardcoded to chi)
    void interpolate()
    {
        AMREX_ASSERT(m_initialized);
        AMREX_ASSERT(m_gr_amr != nullptr);
 
        for (int ilevel = 0; ilevel <= m_gr_amr->finestLevel(); ilevel++)
        {
            // if particles are only on the coarsest level, skip the looping over levels
            if (this->NumberOfParticlesAtLevel(ilevel) == 0L) continue;
 
            amrex::AmrLevel &amr_level = m_gr_amr->getLevel(ilevel);
            const amrex::Geometry &geom = amr_level.Geom();
            amrex::MultiFab &state_level = amr_level.get_new_data(State_Type);
 
            amrex::IntVect ghosts_to_fill(2, 2, 2); // this should be changed based on the var we are interpolating above
            state_level.FillBoundary(c_chi, GR_SPACEDIM, ghosts_to_fill, geom.periodicity());
 
            const auto problem_domain_lo = geom.ProbLoArray();
            const auto dxi = geom.InvCellSizeArray();
            
            // Loop over tiles and interpolate now
            for (ParIterType iter(*this, ilevel); iter.isValid(); ++iter)
            {
                ParticleTileType& punc_tile = ParticlesAt(ilevel, iter);
                auto& punc_particles = punc_tile.GetArrayOfStructs();
                auto* punc_particles_data = punc_particles.data();
                int num_punc_tile = iter.numParticles();
                const auto& fab_array = state_level[iter].const_array();
 
                amrex::ParallelFor(
                    num_punc_tile,
                    [=] AMREX_GPU_DEVICE(int ipunc)
                    {
                        auto& p = punc_particles_data[ipunc];
                        amrex::ParticleReal chi;
                        amrex::IntVect is_nodal = amrex::IntVect::TheZeroVector();
                        LagrangeInterpolator<5> interp;
                        interp.compute_weights(p, problem_domain_lo, dxi, is_nodal);
                        interp.interpolate(&fab_array, &chi, m_start_comp, m_ncomp);
                        p.rdata(0) = chi;
                    });
            }
        }
    }
    
    // Write out the extracted data at (u, v) points in the file
    void write_file(std::string a_file_prefix) 
    {
        AMREX_ASSERT(m_initialized);
        if (!m_params.write_extraction || amrex::ParallelDescriptor::MyProc() != 0) return;
    
        SmallDataIO out_file(m_params.chi_filename + a_file_prefix,
                            m_dt, m_time, m_restart_time,
                            SmallDataIO::NEW, /* first_step = */ true);
    
        // Header data
        std::vector<std::string> header_info = {
            "time = " + std::to_string(m_time),
            "radius = " + std::to_string(m_params.extraction_radii)
        };
        out_file.write_header_line(header_info, "");
    
        std::vector<std::string> components = {"chi"};
        std::vector<std::string> coords = {"theta", "phi"};
        out_file.write_header_line(components, coords);
    
        // Collect interpolated particle data: (theta, phi, chi)
        std::vector<std::tuple<double, double, double>> all_particles;
    
        for (int ilevel = 0; ilevel <= this->finestLevel(); ilevel++)
        {
            for (ParIterType iter(*this, ilevel); iter.isValid(); ++iter)
            {
                const auto& p_tile = ParticlesAt(ilevel, iter);
                const auto& p_arr = p_tile.GetArrayOfStructs();
                const auto* particles = p_arr.data();
                int num_particles = iter.numParticles();
    
                for (int i = 0; i < num_particles; ++i)
                {
                    const auto& p = particles[i];
                    all_particles.emplace_back(p.rdata(1), p.rdata(2), p.rdata(0)); // theta, phi, chi
                }
            }
        }
    
        // Sort by (theta, phi)
        std::sort(all_particles.begin(), all_particles.end(),
                [](const auto& a, const auto& b)
                {
                    return std::tie(std::get<0>(a), std::get<1>(a)) <
                            std::tie(std::get<0>(b), std::get<1>(b));
                });
    
        // Write data rows
        for (const auto& p : all_particles)
        {
            double theta = std::get<0>(p);
            double phi   = std::get<1>(p);
            double chi   = std::get<2>(p);
            out_file.write_data_line({chi}, {theta, phi});
        }
    
        out_file.line_break();
    }

    void write_plotfile(const std::string &a_dir)
    {
        // chi, theta, phi
        amrex::Vector<std::string> real_names = {"chi", "theta", "phi"};
        // none
        amrex::Vector<std::string> int_names{}; 

        // plotfile particle type name
        std::string ptype = "particles";
        
        // Redistribute(); // do I need redistribute here?

        this->WritePlotFile(a_dir, ptype, real_names, int_names);
    }
};

#endif /* SPHERICALPARTICLES_HPP_ */
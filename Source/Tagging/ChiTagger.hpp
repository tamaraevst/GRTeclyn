/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#ifndef CHITAGGER_HPP_
#define CHITAGGER_HPP_

#include "Cell.hpp"
#include "Coordinates.hpp"
#include "DimensionDefinitions.hpp"
#include "FourthOrderDerivatives.hpp"
#include "Tensor.hpp"
#include "VarsTools.hpp"

//! This class tags cells based on two criteria - the
//! value of the second derivs and the extraction regions
class ChiTagger
{
  protected:
    double m_dx;
    FourthOrderDerivatives m_deriv;
    amrex::Real m_threshold;

  public:
    template <class data_t> struct Vars
    {
        data_t chi; //!< Conformal factor

        template <typename mapping_function_t>
        AMREX_GPU_DEVICE void enum_mapping(mapping_function_t mapping_function)
        {
            using namespace VarsTools; // define_enum_mapping is part of
                                       // VarsTools
            define_enum_mapping(mapping_function, c_chi, chi);
        }
    };

    // The constructor
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    ChiTagger(const double dx,
                        const amrex::Real a_threshold)
        : m_dx(dx), m_deriv(dx), m_threshold(a_threshold)
    // NOLINTEND(bugprone-easily-swappable-parameters)
    {}

    template <class data_t>
    AMREX_GPU_DEVICE void
    operator()(int i, int j, int k,
               const amrex::Array4<amrex::TagBox::TagType> &tags,
               const amrex::Array4<data_t const> &state) const
    {
        // first test the gradients for regions of high curvature
        const auto d2     = m_deriv.template diff2<Vars>(i, j, k, state);
        data_t mod_d2_chi = 0;
        FOR (idir, jdir)
        {
            mod_d2_chi += d2.chi[idir][jdir] * d2.chi[idir][jdir];
        }
        data_t criterion = m_dx * std::sqrt(mod_d2_chi);
        if (criterion >= m_threshold)
        {
            tags(i, j, k) = amrex::TagBox::SET;
        }

    }
};

#endif /* CHITAGGER_HPP_ */

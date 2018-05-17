/*BHEADER**********************************************************************
 *
 * Copyright (c) 2018, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * LLNL-CODE-745247. All Rights reserved. See file COPYRIGHT for details.
 *
 * This file is part of smoothG. For more information and source code
 * availability, see https://www.github.com/llnl/smoothG.
 *
 * smoothG is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 ***********************************************************************EHEADER*/

/** @file

    @brief Contains implementation of NonlinearMG
*/

#include "NonlinearMG.hpp"

namespace smoothg
{

NonlinearMG::NonlinearMG(Hierarchy& hierarchy, Cycle cycle)
    : comm_(hierarchy.GetComm()), hierarchy_(hierarchy), cycle_(cycle),
      num_levels_(hierarchy.GetNumLevels()), rhs_(num_levels_), sol_(num_levels_),
      help_(num_levels_)
{
}

void NonlinearMG::Solve(const mfem::Vector& rhs, mfem::Vector& sol) const
{
    rhs_[0].SetDataAndSize(rhs.GetData(), rhs.Size());
    sol_[0].SetDataAndSize(sol.GetData(), sol.Size());

    double norm = mfem::ParNormlp(rhs, 2, comm_); // TODO: add hierarchy source norm
    for (iter_ = 0; iter_ < max_num_iter_; iter_++)
    {
        if (cycle_ == Cycle::FMG)
        {
            FAS_FMG();
        }
        else // V-cycle
        {
            FAS_VCycle(0);
        }

        double resid_norm = ResidualNorm(sol, rhs);
        if (resid_norm < atol_ || resid_norm / norm < rtol_)
        {
            break;
        }
    }
}

void NonlinearMG::Mult(const mfem::Vector& rhs, mfem::Vector& sol) const
{
    Solve(rhs, sol);
}

void NonlinearMG::FAS_FMG() const
{
    // TODO: add smoothing step
    for (int level = 0; level < num_levels_ - 1; level++)
    {
        hierarchy_.Restrict(level, rhs_[level], rhs_[level + 1]);
    }

    for (int level = num_levels_ - 1; level > 0; level--)
    {
        hierarchy_.Solve(level, rhs_[level], sol_[level]);
        hierarchy_.Interpolate(level, sol_[level], sol_[level - 1]);
    }
    hierarchy_.Solve(0, rhs_[0], sol_[0]);
}

void NonlinearMG::FAS_VCycle(int level) const
{
    if (level == num_levels_ - 1)
    {
        hierarchy_.Solve(level, rhs_[level], sol_[level]);
    }
    else
    {
        // Pre-smoothing
        hierarchy_.Smoothing(level, rhs_[level], sol_[level]);

        // Compute FAS coarser level rhs
        hierarchy_.Mult(level, sol_[level], help_[level]);
        help_[level] -= rhs_[level];
        hierarchy_.Restrict(level, help_[level], help_[level + 1]);
        hierarchy_.Restrict(level, sol_[level], sol_[level + 1]);
        hierarchy_.Mult(level+1, sol_[level + 1], rhs_[level + 1]);
        rhs_[level + 1] -= help_[level + 1];

        // Store approximate coarse solution
        help_[level + 1] = sol_[level + 1];

        // Go to coarser level (sol_[level + 1] will be updated)
        FAS_VCycle(level+1);

        // Compute correction
        help_[level + 1] -= sol_[level + 1];
        hierarchy_.Interpolate(level+1, help_[level + 1], help_[level]);
        sol_[level] -= help_[level];

        // Post-smoothing
        hierarchy_.Smoothing(level, rhs_[level], sol_[level]);
    }
}

double NonlinearMG::ResidualNorm(const mfem::Vector& sol, const mfem::Vector& rhs) const
{
    hierarchy_.Mult(0, sol, help_[0]);
    help_[0] -= rhs;
    return mfem::ParNormlp(help_[0], 2, comm_);
}


} // namespace smoothg

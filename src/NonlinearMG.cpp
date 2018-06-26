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
    MPI_Comm_rank(comm_, &myid_);

    help_[0].SetSize(hierarchy_.DomainSize(0));
    for (int level = 1; level < num_levels_; level++)
    {
        int size = hierarchy_.DomainSize(level);
        rhs_[level].SetSize(size);
        sol_[level].SetSize(size);
        help_[level].SetSize(size);
    }
}

void NonlinearMG::Solve(const mfem::Vector& rhs, mfem::Vector& sol)
{
    mfem::Vector zero_vec(sol);
    zero_vec = 0.0;
    double norm = ResidualNorm(zero_vec, rhs);

    rhs_[0].SetDataAndSize(rhs.GetData(), rhs.Size());
    sol_[0].SetDataAndSize(sol.GetData(), sol.Size());
    converged_ = false;

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

        double resid = ResidualNorm(sol, rhs);
        double rel_resid = resid / norm;

        if (myid_ == 0 && print_level_ > 0)
        {
            std::cout << "Nonlinear MG iter " << iter_ << ":  rel resid = "
                      << rel_resid << "  abs resid = " << resid << "\n";
        }

        if (resid < atol_ || rel_resid < rtol_)
        {
            converged_ = true;
            iter_++;
            break;
        }
    }

    if (myid_ == 0 && !converged_ && print_level_ >= 0)
    {
        std::cout << "Warning: Nonlinear MG reached maximum number of iterations!\n";
    }
}

void NonlinearMG::FAS_FMG()
{
    // TODO: add smoothing step
    for (int level = 0; level < num_levels_ - 1; level++)
    {
//        hierarchy_.Restrict(level, rhs_[level], rhs_[level + 1]);
//        hierarchy_.Project(level, sol_[level], sol_[level + 1]);

        hierarchy_.Mult(level, sol_[level], help_[level]);
        help_[level] -= rhs_[level];
        hierarchy_.Restrict(level, help_[level], help_[level + 1]);
        hierarchy_.Project(level, sol_[level], sol_[level + 1]);
        hierarchy_.Mult(level+1, sol_[level + 1], rhs_[level + 1]);
        rhs_[level + 1] -= help_[level + 1];
    }

    for (int level = num_levels_ - 1; level > 0; level--)
    {
//        hierarchy_.Solve(level, rhs_[level], sol_[level]);
//        hierarchy_.Interpolate(level, sol_[level], sol_[level - 1]);

        help_[level] = sol_[level];
        hierarchy_.Solve(level, rhs_[level], sol_[level]);
        help_[level] -= sol_[level];
        hierarchy_.Interpolate(level, help_[level], help_[level - 1]);
        sol_[level-1] -= help_[level-1];
    }
    hierarchy_.Smoothing(0, rhs_[0], sol_[0]);
}

void NonlinearMG::FAS_VCycle(int level)
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
        hierarchy_.Project(level, sol_[level], sol_[level + 1]);
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

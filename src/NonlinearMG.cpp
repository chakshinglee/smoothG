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

NonlinearSolver::NonlinearSolver(MPI_Comm comm, int size)
    : comm_(comm), size_(size), residual_(size)
{
    MPI_Comm_rank(comm_, &myid_);
}

double NonlinearSolver::ResidualNorm(const VectorView& sol, const VectorView& rhs)
{
    residual_ = 0.0;
    Mult(sol, residual_);

    residual_ -= rhs;

    Vector true_resid = AssembleTrueVector(residual_);

    return parlinalgcpp::ParL2Norm(comm_, true_resid);
}

NonlinearMG::NonlinearMG(Hierarchy& hierarchy, Cycle cycle)
    : NonlinearSolver(hierarchy.GetComm(), hierarchy.DomainSize(0)),
      hierarchy_(hierarchy), cycle_(cycle), num_levels_(hierarchy.GetNumLevels()),
      rhs_(num_levels_), sol_(num_levels_), help_(num_levels_)
{
    for (int level = 0; level < num_levels_; level++)
    {
        int size = hierarchy_.DomainSize(level);
        rhs_[level].SetSize(size, 0.0);
        sol_[level].SetSize(size, 0.0);
        help_[level].SetSize(size, 0.0);
    }
}

void NonlinearMG::Mult(const VectorView& x, VectorView& Rx)
{
    hierarchy_.Mult(0, x, Rx);
}

void NonlinearMG::Solve(const VectorView& rhs, VectorView& sol)
{
    Vector zero_vec(sol);
    zero_vec = 0.0;
    double norm = ResidualNorm(zero_vec, rhs);

    rhs_[0] = rhs;
    sol_[0] = sol;
    converged_ = false;

    for (iter_ = 0; iter_ < max_num_iter_; iter_++)
    {
        double resid = ResidualNorm(sol_[0], rhs);
        double rel_resid = resid / norm;

        if (myid_ == 0 && print_level_ > 0)
        {
            std::cout << "Nonlinear MG iter " << iter_ << ":  rel resid = "
                      << rel_resid << "  abs resid = " << resid << "\n";
        }

        if (resid < atol_ || rel_resid < rtol_)
        {
            converged_ = true;
            break;
        }

        FAS_Cycle(0);
    }

    if (myid_ == 0 && !converged_ && print_level_ >= 0)
    {
        std::cout << "Warning: Nonlinear MG reached maximum number of iterations!\n";
    }

    sol = sol_[0];
}

void NonlinearMG::FAS_Cycle(int level)
{
    if (level == num_levels_ - 1)
    {
        hierarchy_.Solve(level, rhs_[level], sol_[level]);
    }
    else
    {
        // Pre-smoothing
        if (cycle_ == V_CYCLE)
        {
            hierarchy_.Smoothing(level, rhs_[level], sol_[level]);
        }

        // Compute FAS coarser level rhs
        // f_{l+1} = P^T( f_l - A_l(x_l) ) + A_{l+1}(pi x_l)
        hierarchy_.Mult(level, sol_[level], help_[level]);
        help_[level] -= rhs_[level];

        // TODO: consider to work with true dofs directly
//        Vector true_help = hierarchy_.AssembleTrueVector(level, help_[level]);
//        help_[level] = hierarchy_.RestrictTrueVector(level, true_help);
        hierarchy_.Restrict(level, help_[level], help_[level + 1]);

        hierarchy_.Project(level, sol_[level], sol_[level + 1]);
//        Vector true_help = hierarchy_.SelectTrueVector(level+1, sol_[level+1]);
//        sol_[level+1] = hierarchy_.DistributeTrueVector(level+1, true_help);

        hierarchy_.Mult(level + 1, sol_[level + 1], rhs_[level + 1]);
        rhs_[level + 1] -= help_[level + 1];

        // Store approximate coarse solution
//        help_[level + 1] = sol_[level + 1];
        Vector help_2 = sol_[level + 1];

        // Go to coarser level (sol_[level + 1] will be updated)
        FAS_Cycle(level + 1);

        // Compute correction
//        help_[level + 1] -= sol_[level + 1];
//        hierarchy_.Interpolate(level + 1, help_[level + 1], help_[level]);
        help_2 -= sol_[level + 1];
        hierarchy_.Interpolate(level + 1, help_2, help_[level]);
        sol_[level] -= help_[level];

        // Post-smoothing
        hierarchy_.Smoothing(level, rhs_[level], sol_[level]);
    }
}

Vector NonlinearMG::AssembleTrueVector(const VectorView& vec_dof) const
{
    return hierarchy_.AssembleTrueVector(0, vec_dof);
}


} // namespace smoothg

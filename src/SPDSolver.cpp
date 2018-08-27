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

/**
   @file

   @brief Implements SPDSolver object.
*/

#include "SPDSolver.hpp"

namespace smoothg
{

SPDSolver::SPDSolver(const MixedMatrix& mgl)
    : SPDSolver(mgl, {})
{
}

SPDSolver::SPDSolver(const MixedMatrix& mgl, const std::vector<int>& ess_vdofs)
    : MGLSolver(mgl, ess_vdofs)
{
    std::vector<double> M_diag(mgl.GlobalM().GetDiag().GetDiag());

    std::vector<double> diag(mgl.LocalD().Rows(), 0.0);

    SparseMatrix D_elim = mgl.LocalD();

    // TODO: revisit this function once ParMatrix::EliminateRowCol is available
    if (ess_vdofs_.size() == 0 && myid_ == 0 && !use_w_)
    {
        ess_vdofs_.push_back(0);
    }
    for (auto&& dof : ess_vdofs_)
    {
        diag[dof] = 1.0;
        D_elim.EliminateRow(dof);
    }

    ParMatrix D_elim_global(comm_, mgl.GlobalD().GetRowStarts(),
                            mgl.EdgeTrueEdge().GetRowStarts(), std::move(D_elim));

    ParMatrix D = D_elim_global.Mult(mgl.EdgeTrueEdge());
    ParMatrix MinvDT = D.Transpose();
//    Print(M_diag);
//    std::cout<<" size = "<< M_diag.size()<<"\n";
    MinvDT.InverseScaleRows(M_diag);

    if (use_w_)
    {
        A_ = parlinalgcpp::ParSub(D.Mult(MinvDT), mgl.GlobalW());
        // need elimination in general. Now only works for diagonal W and 0 essential condition
    }
    else
    {
        A_ = D.Mult(MinvDT);
    }

    A_.AddDiag(diag);
    MinvDT_ = mgl.EdgeTrueEdge().Mult(MinvDT);

    prec_ = parlinalgcpp::BoomerAMG(A_);
    pcg_ = linalgcpp::PCGSolver(A_, prec_, max_num_iter_, rtol_,
                                atol_, 0, parlinalgcpp::ParMult);

    if (myid_ == 0)
    {
        SetPrintLevel(print_level_);
    }

    nnz_ = A_.nnz();
}


SPDSolver::SPDSolver(const SPDSolver& other) noexcept
    : MGLSolver(other), A_(other.A_),
      MinvDT_(other.MinvDT_),
      prec_(other.prec_), pcg_(other.pcg_)
{

}

SPDSolver::SPDSolver(SPDSolver&& other) noexcept
{
    swap(*this, other);
}

SPDSolver& SPDSolver::operator=(SPDSolver other) noexcept
{
    swap(*this, other);

    return *this;
}

void swap(SPDSolver& lhs, SPDSolver& rhs) noexcept
{
    swap(static_cast<MGLSolver&>(lhs),
         static_cast<MGLSolver&>(rhs));

    swap(lhs.A_, rhs.A_);
    swap(lhs.MinvDT_, rhs.MinvDT_);

    swap(lhs.prec_, rhs.prec_);
    swap(lhs.pcg_, rhs.pcg_);
}

void SPDSolver::Solve(const BlockVector& rhs, BlockVector& sol) const
{
    Timer timer(Timer::Start::True);

    rhs_.GetBlock(1) = rhs.GetBlock(1);
    rhs_.GetBlock(1) *= -1.0;

    for (auto&& dof : ess_vdofs_)
    {
        rhs_.GetBlock(1)[dof] = sol.GetBlock(1)[dof]; // only correct when sol[dof] == 0
    }

    pcg_.Mult(rhs_.GetBlock(1), sol.GetBlock(1));

    MinvDT_.Mult(sol.GetBlock(1), sol.GetBlock(0)); // should this be negative too?
    sol.GetBlock(0) *= -1.0;

    timer.Click();
    timing_ = timer.TotalTime();

    num_iterations_ = pcg_.GetNumIterations();
}

void SPDSolver::SetPrintLevel(int print_level)
{
    MGLSolver::SetPrintLevel(print_level);

    if (myid_ == 0)
    {
        pcg_.SetVerbose(print_level_);
    }
}

void SPDSolver::SetMaxIter(int max_num_iter)
{
    MGLSolver::SetMaxIter(max_num_iter);

    pcg_.SetMaxIter(max_num_iter);
}

void SPDSolver::SetRelTol(double rtol)
{
    MGLSolver::SetRelTol(rtol);

    pcg_.SetRelTol(rtol);
}

void SPDSolver::SetAbsTol(double atol)
{
    MGLSolver::SetAbsTol(atol);

    pcg_.SetAbsTol(atol);
}

} // namespace smoothg



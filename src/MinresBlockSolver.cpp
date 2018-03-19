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

   @brief Implements MinresBlockSolver object.
*/

#include "MinresBlockSolver.hpp"

namespace smoothg
{

MinresBlockSolver::MinresBlockSolver(const MixedMatrix& mgl)
    : M_(mgl.M_global_), D_(mgl.D_global_), W_(mgl.W_global_), op_(mgl.true_offsets_), prec_(mgl.true_offsets_),
      edge_true_edge_(mgl.edge_true_edge_), true_rhs_(mgl.true_offsets_), true_sol_(mgl.true_offsets_)
{
    int myid;
    MPI_Comm comm = M_.GetComm();
    MPI_Comm_rank(comm, &myid);

    if (myid == 0)
    {
        D_.EliminateRow(0);
    }

    DT_ = D_.Transpose();

    CooMatrix elim_dof(D_.Rows(), D_.Rows());
    elim_dof.Add(0, 0, 1.0);

    SparseMatrix W = elim_dof.ToSparse();
    W_ = ParMatrix(D_.GetComm(), D_.GetRowStarts(), std::move(W));

    SparseMatrix M_diag(M_.GetDiag().GetDiag());
    ParMatrix MinvDT = DT_;
    MinvDT.InverseScaleRows(M_diag);
    ParMatrix schur_block = D_.Mult(MinvDT);

    ParMatrix M_diag_d(comm, M_.GetRowStarts(), std::move(M_diag));

    M_prec_ = parlinalgcpp::ParDiagScale(std::move(M_diag_d));
    schur_prec_ = parlinalgcpp::BoomerAMG(std::move(schur_block));

    op_.SetBlock(0, 0, M_);
    op_.SetBlock(0, 1, DT_);
    op_.SetBlock(1, 0, D_);
    op_.SetBlock(1, 1, W_);

    prec_.SetBlock(0, 0, M_prec_);
    prec_.SetBlock(1, 1, schur_prec_);
}

void MinresBlockSolver::Mult(const BlockVector& rhs, BlockVector& sol) const
{
    linalgcpp::PMINRESSolver pminres(op_, prec_, 100000, 1e-9, true, parlinalgcpp::ParMult);

    VectorView true_rhs_0 = true_rhs_.GetBlock(0);
    edge_true_edge_.MultAT(rhs.GetBlock(0), true_rhs_0);

    //VectorView true_rhs_1 = true_rhs_.GetBlock(1);
    true_rhs_.GetBlock(1) = rhs.GetBlock(1);

    true_sol_ = 0.0;

    rhs.Print("Input rhs:");
    true_rhs_.Print("True rhs:");

    pminres.Mult(true_rhs_, true_sol_);

    VectorView sol_0 = sol.GetBlock(0);
    edge_true_edge_.Mult(true_sol_.GetBlock(0), sol_0);

    sol.GetBlock(1) = true_sol_.GetBlock(1);
}

void MinresBlockSolver::SetPrintLevel(int print_level)
{
}

void MinresBlockSolver::SetMaxIter(int max_num_iter)
{
}

void MinresBlockSolver::SetRelTol(double rtol)
{
}

void MinresBlockSolver::SetAbsTol(double atol)
{
}


} // namespace smoothg


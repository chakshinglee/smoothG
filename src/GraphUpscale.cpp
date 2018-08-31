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

    @brief Contains GraphUpscale class
*/

#include "GraphUpscale.hpp"

namespace smoothg
{

GraphUpscale::GraphUpscale(const Graph& graph, const UpscaleParams& params)
    : Operator(graph.vertex_edge_local_.Rows()),
      mgl_(params.max_levels),
      coarsener_(params.max_levels - 1),
      solver_(params.max_levels),
      rhs_(params.max_levels),
      sol_(params.max_levels),
      constant_rep_(params.max_levels),
      ess_vdofs_(params.max_levels),
      comm_(graph.edge_true_edge_.GetComm()),
      myid_(graph.edge_true_edge_.GetMyId()),
      setup_time_(0),
      hybridization_(params.hybridization),
      proj_pw1_(params.max_levels)
{
    Timer timer(Timer::Start::True);

    // Compute Topology
    std::vector<GraphTopology> gts;
    gts.emplace_back(graph);
    ess_vdofs_[0] = params.ess_vdofs;

    for (int level = 1; level < params.max_levels - 1; ++level)
    {
        CoarsenEssVdofs(level, gts.back().NumAggs());
        gts.emplace_back(gts.back(), params.coarsen_factor, ess_vdofs_[level]);
    }

    // Fine Level
    int level = 0;
    {
        mgl_[level] = MixedMatrix(graph, ess_vdofs_[level].size());
        mgl_[level].AssembleM();

        proj_pw1_[level] = SparseIdentity(gts[level].NumVertices());

        MakeSolver(level);
        MakeVectors(level);
    }

    // Coarse Levels
    for (level = 1; level < params.max_levels; ++level)
    {
        auto& gt_i = gts[level - 1];
        auto& spect_pair_i = params.spectral_pair[level - 1];

        double spect_tol_i = spect_pair_i.first;
        int num_evects_i = level == 1 ? spect_pair_i.second : spect_pair_i.second;

        int num_vert = gt_i.GlobalNumVertices();
        int num_agg = gt_i.GlobalNumAggs();

        ParPrint(myid_, printf("Coarsening: %d / %d = %.2f, evects: %d\n",
                 num_vert, num_agg, num_vert / (double) num_agg, num_evects_i));

        SparseMatrix Q_pw1 = gt_i.agg_vertex_local_;
        SparseMatrix scale = Q_pw1.Mult(Q_pw1.Transpose());
        assert(IsDiag(scale));
        Q_pw1.InverseScaleRows(scale.GetDiag());

        coarsener_[level - 1] = GraphCoarsen(std::move(gt_i), mgl_[level - 1],
                                             num_evects_i, spect_tol_i);
        mgl_[level] = MixedMatrix(coarsener_[level - 1].Coarsen(mgl_[level - 1]));
        mgl_[level].AssembleM();

        CoarsenEssVdofs(level, mgl_[level].LocalD().Rows());
        auto tmp = proj_pw1_[level - 1].Mult(coarsener_[level - 1].Pvertex());
        proj_pw1_[level] = Q_pw1.Mult(tmp);

        MakeSolver(level);
        MakeVectors(level);
    }

    do_ortho_ = !GetMatrix(0).CheckW() && params.ess_vdofs.size() == 0;

    timer.Click();
    setup_time_ += timer.TotalTime();
}

GraphUpscale::GraphUpscale(const Graph& graph,
                           const std::vector<Vector>& local_weight,
                           const UpscaleParams& params)
    : Operator(graph.vertex_edge_local_.Rows()),
      mgl_(params.max_levels),
      coarsener_(params.max_levels - 1),
      solver_(params.max_levels),
      rhs_(params.max_levels),
      sol_(params.max_levels),
      constant_rep_(params.max_levels),
      ess_vdofs_(params.max_levels),
      comm_(graph.edge_true_edge_.GetComm()),
      myid_(graph.edge_true_edge_.GetMyId()),
      setup_time_(0),
      hybridization_(params.hybridization),
      proj_pw1_(params.max_levels)
{
    Timer timer(Timer::Start::True);

    // Compute Topology
    std::vector<GraphTopology> gts;
    gts.emplace_back(graph);
    ess_vdofs_[0] = params.ess_vdofs;

    for (int level = 1; level < params.max_levels - 1; ++level)
    {
        CoarsenEssVdofs(level, gts.back().NumAggs());
        gts.emplace_back(gts.back(), params.coarsen_factor, ess_vdofs_[level]);
    }

    // Fine Level
    int level = 0;
    {
        mgl_[level] = MixedMatrix(graph, local_weight, ess_vdofs_[level].size());
        mgl_[level].AssembleM();

        proj_pw1_[level] = SparseIdentity(gts[level].NumVertices());

        MakeSolver(level);
        MakeVectors(level);
    }

    // Coarse Levels
    for (level = 1; level < params.max_levels; ++level)
    {
        auto& gt_i = gts[level - 1];
        auto& spect_pair_i = params.spectral_pair[level - 1];

        double spect_tol_i = spect_pair_i.first;
        int num_evects_i = level == 1 ? spect_pair_i.second : spect_pair_i.second;
//        num_evects_i = std::max(num_evects_i - (level -1), 1);

        int num_vert = gt_i.GlobalNumVertices();
        int num_agg = gt_i.GlobalNumAggs();

        ParPrint(myid_, printf("Coarsening: %d / %d = %.2f, evects: %d\n",
                 num_vert, num_agg, num_vert / (double) num_agg, num_evects_i));

        SparseMatrix Q_pw1 = gt_i.agg_vertex_local_;
        SparseMatrix scale = Q_pw1.Mult(Q_pw1.Transpose());
        assert(IsDiag(scale));
        Q_pw1.InverseScaleRows(scale.GetDiag());

        coarsener_[level - 1] = GraphCoarsen(std::move(gt_i), mgl_[level - 1],
                                             num_evects_i, spect_tol_i);
        mgl_[level] = MixedMatrix(coarsener_[level - 1].Coarsen(mgl_[level - 1]));
        mgl_[level].AssembleM();

        CoarsenEssVdofs(level, mgl_[level].LocalD().Rows());
        auto tmp = proj_pw1_[level - 1].Mult(coarsener_[level - 1].Pvertex());
        proj_pw1_[level] = Q_pw1.Mult(tmp);

        MakeSolver(level);
        MakeVectors(level);
    }

    do_ortho_ = !GetMatrix(0).CheckW() && params.ess_vdofs.size() == 0;

    timer.Click();
    setup_time_ += timer.TotalTime();
}

void GraphUpscale::MakeSolver(int level)
{
    auto& mm = GetMatrix(level);

//    if (level == 0)
//    {
//        solver_[level] = make_unique<MinresBlockSolver>(mm, ess_vdofs_[level]);
//    }
//    else
        if (hybridization_)
    {
        solver_[level] = make_unique<HybridSolver>(mm);
    }
    else
    {
        solver_[level] = make_unique<MinresBlockSolver>(mm, ess_vdofs_[level]);
    }
}

void GraphUpscale::MakeSolver(int level, const std::vector<double>& agg_weights)
{
    auto& mm = GetMatrix(level);

//    if (level == 0)
//    {
//        mm.AssembleM(agg_weights);
//        solver_[level] = make_unique<MinresBlockSolver>(mm, ess_vdofs_[level]);
//    }
//    else
        if (hybridization_)
    {
        if (!solver_[level])
        {
            solver_[level] = make_unique<HybridSolver>(mm);
        }

        auto& hb = dynamic_cast<HybridSolver&>(*solver_[level]);
        hb.UpdateAggScaling(agg_weights);
    }
    else
    {
        mm.AssembleM(agg_weights);
        solver_[level] = make_unique<MinresBlockSolver>(mm, ess_vdofs_[level]);
    }
}

std::vector<BlockVector> GraphUpscale::MultMultiLevel(const BlockVector& x) const
{
    std::vector<BlockVector> sols;

    MultMultiLevel(x, sols);

    return sols;
}

void GraphUpscale::MultMultiLevel(const BlockVector& x, std::vector<BlockVector>& sols) const
{
    int num_levels = NumLevels();

    sols.resize(num_levels, GetBlockVector(0));

    for (int i = 0; i < num_levels; ++i)
    {
        Solve(i, x, sols[i]);
    }
}

BlockVector GraphUpscale::MultMultiGrid(const BlockVector& x) const
{
    BlockVector sol = GetBlockVector(0);
    sol = 0.0;

    MultMultiGrid(x, sol);

    return sol;
}

void GraphUpscale::MultMultiGrid(const BlockVector& x, BlockVector& sol) const
{
    rhs_[0] = x;
    sol_[0] = sol;

    int num_coarse = coarsener_.size();

    for (int i = 0; i < num_coarse; ++i)
    {
        coarsener_[i].Restrict(rhs_[i], rhs_[i + 1]);
        coarsener_[i].Restrict(sol_[i], sol_[i + 1]);
    }

    int num_levels = solver_.size();

    for (int i = num_levels - 1; i >= 0; --i)
    {
        rhs_[i].GetBlock(1) *= -1.0;
        sol_[i].GetBlock(1) *= -1.0;

        solver_[i]->Solve(rhs_[i], sol_[i]);

        if (do_ortho_)
        {
            OrthoConstant(comm_, sol_[i].GetBlock(1), constant_rep_[i]);
        }

        if (i != 0)
        {
            coarsener_[i - 1].Interpolate(sol_[i], sol_[i - 1]);
        }
    }

    sol = sol_[0];

    if (do_ortho_)
    {
        OrthoConstant(comm_, sol.GetBlock(1), constant_rep_[0]);
    }
}

void GraphUpscale::Mult(const VectorView& x, VectorView y) const
{
    Solve(1, x, y);
}

void GraphUpscale::Mult(int level, const VectorView& x, VectorView y) const
{
    mgl_[level].Mult(x, y);
}

void GraphUpscale::Mult(int level, const std::vector<double>& scale,
                        const VectorView& x, VectorView y) const
{
    mgl_[level].Mult(scale, x, y);
}

void GraphUpscale::Solve(const VectorView& x, VectorView y) const
{
    Solve(1, x, y);
}

Vector GraphUpscale::Solve(const VectorView& x) const
{
    return Solve(1, x);
}

void GraphUpscale::Solve(const BlockVector& x, BlockVector& y) const
{
    Solve(1, x, y);
}

BlockVector GraphUpscale::Solve(const BlockVector& x) const
{
    return Solve(1, x);
}

void GraphUpscale::Solve(int level, const VectorView& x, VectorView y) const
{
    rhs_[0].GetBlock(0) = 0.0;
    rhs_[0].GetBlock(1) = x;

    for (int i = 0; i < level; ++i)
    {
        coarsener_[i].Restrict(rhs_[i], rhs_[i + 1]);
    }

    rhs_[level].GetBlock(1) *= -1.0;
    sol_[level] = 0.0;

    solver_[level]->Solve(rhs_[level], sol_[level]);

    if (do_ortho_)
    {
        Orthogonalize(level, sol_[level].GetBlock(1));
    }

    for (int i = level - 1; i >= 0; --i)
    {
        coarsener_[i].Interpolate(sol_[i + 1], sol_[i]);
    }

    y = sol_[0].GetBlock(1);
}

Vector GraphUpscale::Solve(int level, const VectorView& x) const
{
    Vector y = GetVector(0);

    Solve(level, x, y);

    return y;
}

void GraphUpscale::Solve(int level, const BlockVector& x, BlockVector& y) const
{
    rhs_[0] = x;

    for (int i = 0; i < level; ++i)
    {
        coarsener_[i].Restrict(rhs_[i], rhs_[i + 1]);
    }

//    rhs_[level].GetBlock(1) *= -1.0;
    sol_[level] = 0.0;

    solver_[level]->Solve(rhs_[level], sol_[level]);

    if (do_ortho_)
    {
        Orthogonalize(level, sol_[level]);
    }

    for (int i = level - 1; i >= 0; --i)
    {
        coarsener_[i].Interpolate(sol_[i + 1], sol_[i]);
    }

    y = sol_[0];
}

BlockVector GraphUpscale::Solve(int level, const BlockVector& x) const
{
    BlockVector y = GetBlockVector(0);

    Solve(level, x, y);

    return y;
}

void GraphUpscale::SolveLevel(int level, const VectorView& x, VectorView y) const
{
    solver_.at(level)->Solve(x, y);
    y *= -1.0;

    if (do_ortho_)
    {
        Orthogonalize(level, y);
    }
}

Vector GraphUpscale::SolveLevel(int level, const VectorView& x) const
{
    Vector y = GetVector(level);
    SolveLevel(level, x, y);

    return y;
}

void GraphUpscale::SolveLevel(int level, const BlockVector& x, BlockVector& y) const
{
    solver_.at(level)->Solve(x, y);
//    y.GetBlock(1) *= -1.0;

    if (do_ortho_)
    {
        Orthogonalize(level, y);
    }
}

BlockVector GraphUpscale::SolveLevel(int level, const BlockVector& x) const
{
    BlockVector y = GetBlockVector(level);
    SolveLevel(level, x, y);

    return y;
}

void GraphUpscale::Interpolate(const VectorView& x, VectorView y) const
{
    int x_level = size_to_level_.at(x.size());
    int y_level = size_to_level_.at(y.size());

    // Don't need temp space and copy if only between consecutive levels
    if (x_level - y_level == 1)
    {
        coarsener_[y_level].Interpolate(x, y);

        return;
    }

    sol_[x_level].GetBlock(1) = x;

    for (int i = x_level - 1; i >= y_level; --i)
    {
        coarsener_[i].Interpolate(sol_[i + 1].GetBlock(1), sol_[i].GetBlock(1));
    }

    y = sol_[y_level].GetBlock(1);
}

Vector GraphUpscale::Interpolate(const VectorView& x, int level) const
{
    Vector y = GetVector(level);

    Interpolate(x, y);

    return y;
}

void GraphUpscale::Interpolate(const BlockVector& x, BlockVector& y) const
{
    int x_level = size_to_level_.at(x.GetBlock(1).size());
    int y_level = size_to_level_.at(y.GetBlock(1).size());

    // Don't need temp space and copy if only between consecutive levels
    if (x_level - y_level == 1)
    {
        coarsener_[y_level].Interpolate(x, y);

        return;
    }

    sol_[x_level] = x;

    for (int i = x_level - 1; i >= y_level; --i)
    {
        coarsener_[i].Interpolate(sol_[i + 1], sol_[i]);
    }

    y = sol_[y_level];
}

BlockVector GraphUpscale::Interpolate(const BlockVector& x, int level) const
{
    BlockVector y = GetBlockVector(level);

    Interpolate(x, y);

    return y;
}

void GraphUpscale::Restrict(const VectorView& x, VectorView y) const
{
    int x_level = size_to_level_.at(x.size());
    int y_level = size_to_level_.at(y.size());

    // Don't need temp space and copy if only between consecutive levels
    if (y_level - x_level == 1)
    {
        coarsener_[x_level].Restrict(x, y);

        return;
    }

    sol_[x_level].GetBlock(1) = x;

    for (int i = x_level; i < y_level; ++i)
    {
        coarsener_[i].Restrict(sol_[i].GetBlock(1), sol_[i + 1].GetBlock(1));
    }

    y = sol_[y_level].GetBlock(1);
}

Vector GraphUpscale::Restrict(const VectorView& x, int level) const
{
    Vector y = GetVector(level);

    Restrict(x, y);

    return y;
}

void GraphUpscale::Restrict(const BlockVector& x, BlockVector& y) const
{
    int x_level = size_to_level_.at(x.GetBlock(1).size());
    int y_level = size_to_level_.at(y.GetBlock(1).size());

    // Don't need temp space and copy if only between consecutive levels
    if (y_level - x_level == 1)
    {
        coarsener_[x_level].Restrict(x, y);

        return;
    }

    sol_[x_level] = x;

    for (int i = x_level; i < y_level; ++i)
    {
        coarsener_[i].Restrict(sol_[i], sol_[i + 1]);
    }

    y = sol_[y_level];
}

BlockVector GraphUpscale::Restrict(const BlockVector& x, int level) const
{
    BlockVector y = GetBlockVector(level);

    Restrict(x, y);

    return y;
}

void GraphUpscale::Project(const BlockVector& x, BlockVector& y) const
{
    coarsener_[0].Project(x, y);
}

BlockVector GraphUpscale::Project(const BlockVector& x) const
{
    return coarsener_[0].Project(x);
}

const std::vector<int>& GraphUpscale::BlockOffsets(int level) const
{
    return GetMatrix(level).Offsets();
}

const std::vector<int>& GraphUpscale::TrueBlockOffsets(int level) const
{
    return GetMatrix(level).TrueOffsets();
}

void GraphUpscale::Orthogonalize(int level, VectorView vect) const
{
    OrthoConstant(comm_, vect, constant_rep_.at(level));
}

void GraphUpscale::Orthogonalize(int level, BlockVector& vect) const
{
    Orthogonalize(level, vect.GetBlock(1));
}

Vector GraphUpscale::GetVector(int level) const
{
    return Vector(GetMatrix(level).LocalD().Rows());
}

BlockVector GraphUpscale::GetBlockVector(int level) const
{
    return BlockVector(GetMatrix(level).Offsets());
}

BlockVector GraphUpscale::GetTrueBlockVector(int level) const
{
    return BlockVector(GetMatrix(level).TrueOffsets());
}

std::vector<Vector> GraphUpscale::GetMLVectors() const
{
    int num_levels = NumLevels();

    std::vector<Vector> vects(num_levels);

    for (int i = 0; i < num_levels; ++i)
    {
        vects[i] = GetVector(i);
    }

    return vects;
}

std::vector<BlockVector> GraphUpscale::GetMLBlockVector() const
{
    int num_levels = NumLevels();

    std::vector<BlockVector> vects(num_levels);

    for (int i = 0; i < num_levels; ++i)
    {
        vects[i] = GetBlockVector(i);
    }

    return vects;
}

MixedMatrix& GraphUpscale::GetMatrix(int level)
{
    assert(level >= 0 && level < static_cast<int>(mgl_.size()));

    return mgl_[level];
}

const MixedMatrix& GraphUpscale::GetMatrix(int level) const
{
    assert(level >= 0 && level < static_cast<int>(mgl_.size()));

    return mgl_[level];
}

int GraphUpscale::GlobalRows() const
{
    return GetMatrix(0).GlobalD().GlobalRows();
}

int GraphUpscale::GlobalCols() const
{
    return GetMatrix(0).GlobalD().GlobalRows();
}

void GraphUpscale::PrintInfo(std::ostream& out) const
{
    int num_procs;
    MPI_Comm_size(comm_, &num_procs);

    if (myid_ == 0)
    {
        int old_precision = out.precision();
        out.precision(3);

        out << "\n";

        if (num_procs > 1)
        {
            out << "Processors: " << num_procs << "\n";
            out << "---------------------\n";
        }

        out << "\n";

        for (size_t i = 0; i < mgl_.size(); ++i)
        {
            out << "Level " << i << " Matrix\n";
            out << "---------------------\n";
            out << "M Size\t\t" << mgl_[i].GlobalM().GlobalRows() << "\n";
            out << "D Size\t\t" << mgl_[i].GlobalD().GlobalRows() << "\n";
            out << "+ Size\t\t" << mgl_[i].GlobalRows() << "\n";
            out << "NonZeros:\t" << mgl_[i].GlobalNNZ() << "\n";
            out << "\n";

            if (i != 0)
            {
                double op_comp = 1.0 + (solver_[i]->GetNNZ() / (double) solver_[0]->GetNNZ());

                out << "Op Comp:\t" << op_comp << "\n";
                out << "\n";
            }
        }

        double total_oc = 1.0;
        for (size_t i = 1; i < mgl_.size(); ++i)
        {
            total_oc += (solver_[i]->GetNNZ() / (double) solver_[0]->GetNNZ());
        }
        out << "Overall OC: " << total_oc << "\n";

        out.precision(old_precision);
    }
}

double GraphUpscale::OperatorComplexity() const
{
    assert(solver_[1]);

    double nnz_levels = 0.0;

    for (auto&& solver : solver_)
    {
        nnz_levels += solver->GetNNZ();
    }

    double nnz_fine;

    if (solver_[0])
    {
        nnz_fine = solver_[0]->GetNNZ();
    }
    else
    {
        nnz_fine = GetMatrix(0).GlobalNNZ();
    }

    return nnz_levels / nnz_fine;
}

void GraphUpscale::SetPrintLevel(int print_level)
{
    for (auto& solver : solver_)
    {
        if (solver)
        {
            solver->SetPrintLevel(print_level);
        }
    }
}

void GraphUpscale::SetMaxIter(int max_num_iter)
{
    for (auto& solver : solver_)
    {
        if (solver)
        {
            solver->SetMaxIter(max_num_iter);
        }
    }
}

void GraphUpscale::SetRelTol(double rtol)
{
    for (auto& solver : solver_)
    {
        if (solver)
        {
            solver->SetRelTol(rtol);
        }
    }
}

void GraphUpscale::SetAbsTol(double atol)
{
    for (auto& solver : solver_)
    {
        if (solver)
        {
            solver->SetAbsTol(atol);
        }
    }
}

void GraphUpscale::ShowCoarseSolveInfo(std::ostream& out) const
{
    assert(solver_[1]);

    if (myid_ == 0)
    {

        int num_solver = solver_.size();

        for (int i = 1; i < num_solver; ++i)
        {
            out << "\n";
            out << "Level " << i << " Solve Time:       " << solver_[i]->GetTiming() << "\n";
            out << "Level " << i << " Solve Iterations: " << solver_[i]->GetNumIterations() << "\n";
        }
    }
}

void GraphUpscale::ShowFineSolveInfo(std::ostream& out) const
{
    assert(solver_[0]);

    if (myid_ == 0)
    {
        out << "\n";
        out << "Fine Solve Time:         " << solver_[0]->GetTiming() << "\n";
        out << "Fine Solve Iterations:   " << solver_[0]->GetNumIterations() << "\n";
    }
}

void GraphUpscale::ShowSolveInfo(int level, std::ostream& out) const
{
    assert(solver_[level]);

    if (myid_ == 0)
    {
        out << "\n";
        out << "Level " << level << " Solve Time:       " << solver_[level]->GetTiming() << "\n";
        out << "Level " << level << " Solve Iterations: " << solver_[level]->GetNumIterations() << "\n";
    }
}

void GraphUpscale::ShowSetupTime(std::ostream& out) const
{
    if (myid_ == 0)
    {
        out << "\n";
        out << "GraphUpscale Setup Time:      " << setup_time_ << "\n";
    }
}

double GraphUpscale::SolveTime(int level) const
{
    return solver_.at(level)->GetTiming();
}

int GraphUpscale::SolveIters(int level) const
{
    return solver_.at(level)->GetNumIterations();
}

double GraphUpscale::GetSetupTime() const
{
    return setup_time_;
}

std::vector<double> GraphUpscale::ComputeErrors(const BlockVector& upscaled_sol,
                                                const BlockVector& fine_sol) const
{
    const SparseMatrix& M = GetMatrix(0).LocalM();
    const SparseMatrix& D = GetMatrix(0).LocalD();

    auto info = smoothg::ComputeErrors(comm_, M, D, upscaled_sol, fine_sol);
    info.push_back(OperatorComplexity());

    return info;
}

void GraphUpscale::ShowErrors(const BlockVector& upscaled_sol,
                              const BlockVector& fine_sol) const
{
    auto info = ComputeErrors(upscaled_sol, fine_sol);

    if (myid_ == 0)
    {
        smoothg::ShowErrors(info);
    }
}

ParMatrix GraphUpscale::ToPrimal() const
{
    const auto& mgl = GetMatrix(0);

    std::vector<double> M_diag(mgl.GlobalM().GetDiag().GetDiag());

    SparseMatrix D_elim = mgl.LocalD();

    bool use_w = mgl.CheckW();

    if (ess_vdofs_[0].size() > 0)
    {
        std::vector<int> marker(D_elim.Cols(), 0);

        for (auto&& dof : ess_vdofs_[0])
        {
            marker[dof] = 1;
        }

        D_elim.EliminateCol(marker);
    }

    ParMatrix D_elim_global(comm_, mgl.GlobalD().GetRowStarts(),
                            mgl.EdgeTrueEdge().GetRowStarts(), std::move(D_elim));

    ParMatrix D = D_elim_global.Mult(mgl.EdgeTrueEdge());
    ParMatrix MinvDT = D.Transpose();
    MinvDT.InverseScaleRows(M_diag);

    ParMatrix A;

    if (use_w)
    {
        A = parlinalgcpp::ParSub(D.Mult(MinvDT), mgl.GlobalW());
    }
    else
    {
        A = D.Mult(MinvDT);
    }

    return A;
}

void GraphUpscale::MakeVectors(int level)
{
    if (level == 0)
    {
        constant_rep_[level] = Vector(Rows(), 1.0 / std::sqrt(GlobalRows()));
    }
    else
    {
        constant_rep_[level] = Vector(coarsener_[level - 1].Restrict(constant_rep_[level - 1]));
    }

    rhs_[level] = BlockVector(mgl_[level].Offsets());
    sol_[level] = BlockVector(mgl_[level].Offsets());

    size_to_level_[rhs_[level].GetBlock(1).size()] = level;
}

void GraphUpscale::CoarsenEssVdofs(int level, int num_dofs)
{
    assert(level > 0);
    ess_vdofs_[level].resize(ess_vdofs_[level - 1].size());
    std::iota(ess_vdofs_[level].begin(), ess_vdofs_[level].end(),
              num_dofs - ess_vdofs_[level - 1].size());
}

void GraphUpscale::Project_PW_One(int level, const VectorView& x, VectorView& y) const
{
    proj_pw1_[level].Mult(x, y);
}

} // namespace smoothg

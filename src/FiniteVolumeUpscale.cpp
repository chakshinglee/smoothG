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

    @brief Implements FiniteVolumeUpscale class
*/

#include "FiniteVolumeUpscale.hpp"

namespace smoothg
{

std::vector<std::vector<double> > BuildCoarseToFineNormalFlip(
    const GraphTopology& topo, const mfem::SparseMatrix& vertex_edge)
{
    const auto& face_Agg = topo.face_Agg_;
    const auto& face_edge = topo.face_edge_;
    const auto vert_Agg = smoothg::Transpose(topo.Agg_vertex_);
    const auto edge_vertex = smoothg::Transpose(vertex_edge);

    std::vector<std::vector<double> > c2f_normal_flip(face_Agg.Height());
    mfem::Array<int> edges;
    for (int face = 0; face < face_Agg.Height(); face++)
    {
        GetTableRow(face_edge, face, edges);
        int first_Agg = face_Agg.GetRowColumns(face)[0];
        c2f_normal_flip[face].resize(edges.Size());
        for (int edge = 0; edge < edges.Size(); edge++)
        {
            int first_vert = edge_vertex.GetRowColumns(edges[edge])[0];
            int first_vert_Agg = vert_Agg.GetRowColumns(first_vert)[0];
            c2f_normal_flip[face][edge] = (first_vert_Agg == first_Agg) ? 1.0 : -1.0;
        }
    }
    return c2f_normal_flip;
}

// TODO(gelever1): Refactor these two constructors into one (or use Init function)
FiniteVolumeUpscale::FiniteVolumeUpscale(MPI_Comm comm,
                                         const mfem::SparseMatrix& vertex_edge,
                                         const mfem::Vector& weight,
                                         const mfem::Array<int>& partitioning,
                                         const mfem::HypreParMatrix& edge_d_td,
                                         const mfem::SparseMatrix& edge_boundary_att,
                                         const mfem::Array<int>& ess_attr,
                                         double spect_tol, int max_evects,
                                         bool dual_target, bool scaled_dual,
                                         bool energy_dual, bool hybridization,
                                         const SAAMGeParam* saamge_param)
    : Upscale(comm, vertex_edge.Height(), hybridization),
      edge_d_td_(edge_d_td),
      edge_boundary_att_(edge_boundary_att),
      ess_attr_(ess_attr)
{
    mfem::StopWatch chrono;
    chrono.Start();

    // Hypre may modify the original vertex_edge, which we seek to avoid
    mfem::SparseMatrix ve_copy(vertex_edge);

    mixed_laplacians_.emplace_back(vertex_edge, weight, edge_d_td_,
                                   MixedMatrix::DistributeWeight::False);

    auto graph_topology = make_unique<GraphTopology>(ve_copy, edge_d_td_, partitioning,
                                                     &edge_boundary_att_);

    c2f_normal_flip_ = BuildCoarseToFineNormalFlip(*graph_topology, vertex_edge);

    bool coarse_coefficient = false;
    coarsener_ = make_unique<SpectralAMG_MGL_Coarsener>(
                     mixed_laplacians_[0], std::move(graph_topology), spect_tol,
                     max_evects, dual_target, scaled_dual, energy_dual,
                     coarse_coefficient);
    coarsener_->construct_coarse_subspace();

    mixed_laplacians_.push_back(coarsener_->GetCoarse());

    mfem::SparseMatrix& Dref = GetCoarseMatrix().GetD();
    mfem::Array<int> marker(Dref.Width());
    marker = 0;

    MarkDofsOnBoundary(coarsener_->get_GraphTopology_ref().face_bdratt_,
                       coarsener_->construct_face_facedof_table(),
                       ess_attr, marker);

    for (int i = 0; i < ess_attr.Size(); i++)
    {
        if (ess_attr[i] == 0)
        {
            remove_one_dof_ = false;
            break;
        }
    }

    if (hybridization) // Hybridization solver
    {
        auto face_bdratt = coarsener_->get_GraphTopology_ref().face_bdratt_;
        coarse_solver_ = make_unique<HybridSolver>(
                             comm, mixed_laplacians_.back(), *coarsener_,
                             &face_bdratt, &marker, 0, saamge_param);
    }
    else // L2-H1 block diagonal preconditioner
    {
        GetCoarseMatrix().BuildM();
        mfem::SparseMatrix& Mref = GetCoarseMatrix().GetM();
        for (int mm = 0; mm < marker.Size(); ++mm)
        {
            if (marker[mm])
                Mref.EliminateRowCol(mm);
        }

        Dref.EliminateCols(marker);

        coarse_solver_ = make_unique<MinresBlockSolverFalse>(
                             comm, GetCoarseMatrix(), remove_one_dof_);
    }

    MakeCoarseVectors();

    chrono.Stop();
    setup_time_ += chrono.RealTime();
}

FiniteVolumeUpscale::FiniteVolumeUpscale(MPI_Comm comm,
                                         const mfem::SparseMatrix& vertex_edge,
                                         const mfem::Vector& weight,
                                         const mfem::SparseMatrix& w_block,
                                         const mfem::Array<int>& partitioning,
                                         const mfem::HypreParMatrix& edge_d_td,
                                         const mfem::SparseMatrix& edge_boundary_att,
                                         const mfem::Array<int>& ess_attr,
                                         double spect_tol, int max_evects,
                                         bool dual_target, bool scaled_dual,
                                         bool energy_dual, bool hybridization,
                                         const SAAMGeParam* saamge_param)
    : Upscale(comm, vertex_edge.Height(), hybridization),
      edge_d_td_(edge_d_td),
      edge_boundary_att_(edge_boundary_att),
      ess_attr_(ess_attr)
{
    mfem::StopWatch chrono;
    chrono.Start();

    // Hypre may modify the original vertex_edge, which we seek to avoid
    mfem::SparseMatrix ve_copy(vertex_edge);

    mixed_laplacians_.emplace_back(vertex_edge, weight, w_block, edge_d_td_,
                                   MixedMatrix::DistributeWeight::False);

    auto graph_topology = make_unique<GraphTopology>(
                              ve_copy, edge_d_td_, partitioning, &edge_boundary_att_);

    bool coarse_coefficient = false;
    coarsener_ = make_unique<SpectralAMG_MGL_Coarsener>(
                     mixed_laplacians_[0], std::move(graph_topology), spect_tol,
                     max_evects, dual_target, scaled_dual, energy_dual, coarse_coefficient);
    coarsener_->construct_coarse_subspace();

    mixed_laplacians_.push_back(coarsener_->GetCoarse());

    mfem::SparseMatrix& Dref = GetCoarseMatrix().GetD();
    mfem::Array<int> marker(Dref.Width());
    marker = 0;

    MarkDofsOnBoundary(coarsener_->get_GraphTopology_ref().face_bdratt_,
                       coarsener_->construct_face_facedof_table(),
                       ess_attr, marker);

    remove_one_dof_ = !(mixed_laplacians_.back().CheckW());

    if (hybridization) // Hybridization solver
    {
        auto face_bdratt = coarsener_->get_GraphTopology_ref().face_bdratt_;
        coarse_solver_ = make_unique<HybridSolver>(
                             comm, mixed_laplacians_.back(), *coarsener_,
                             &face_bdratt, &marker, 0, saamge_param);
    }
    else // L2-H1 block diagonal preconditioner
    {
        GetCoarseMatrix().BuildM();
        mfem::SparseMatrix& Mref = GetCoarseMatrix().GetM();
        for (int mm = 0; mm < marker.Size(); ++mm)
        {
            if (marker[mm])
                Mref.EliminateRowCol(mm);
        }

        Dref.EliminateCols(marker);

        coarse_solver_ = make_unique<MinresBlockSolverFalse>(comm, mixed_laplacians_.back());
    }

    MakeCoarseVectors();

    chrono.Stop();
    setup_time_ += chrono.RealTime();
}

void FiniteVolumeUpscale::MakeFineSolver()
{
    mfem::Array<int> marker;
    BooleanMult(edge_boundary_att_, ess_attr_, marker);

    if (!fine_solver_)
    {
        if (hybridization_) // Hybridization solver
        {
            fine_solver_ = make_unique<HybridSolver>(comm_, GetFineMatrix(),
                                                     &edge_boundary_att_, &marker);
        }
        else // L2-H1 block diagonal preconditioner
        {
            mfem::SparseMatrix& Mref = GetFineMatrix().GetM();
            mfem::SparseMatrix& Dref = GetFineMatrix().GetD();
            const bool w_exists = GetFineMatrix().CheckW();

            for (int mm = 0; mm < marker.Size(); ++mm)
            {
                if (marker[mm])
                {
                    //Mref.EliminateRowCol(mm, ess_data[k][mm], *(rhs[k]));

                    const bool set_diag = true;
                    Mref.EliminateRow(mm, set_diag);
                }
            }

            Dref.EliminateCols(marker);
            if (!w_exists && remove_one_dof_ && myid_ == 0)
            {
                Dref.EliminateRow(0);
            }

            fine_solver_ = make_unique<MinresBlockSolverFalse>(
                               comm_, GetFineMatrix(), remove_one_dof_);
        }
    }
}

} // namespace smoothg

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

    @brief Implements FiniteVolumeMLMC class
*/

#include "FiniteVolumeMLMC.hpp"

namespace smoothg
{

FiniteVolumeMLMC::FiniteVolumeMLMC(MPI_Comm comm,
                                   const mfem::SparseMatrix& vertex_edge,
                                   const mfem::Vector& weight,
                                   const mfem::Array<int>& partitioning,
                                   const mfem::HypreParMatrix& edge_d_td,
                                   const mfem::SparseMatrix& edge_boundary_att,
                                   const mfem::Array<int>& ess_attr,
                                   double spect_tol, int max_evects,
                                   bool dual_target, bool scaled_dual,
                                   bool energy_dual, bool hybridization,
                                   bool coarse_components,
                                   const SAAMGeParam* saamge_param)
    : Upscale(comm, vertex_edge.Height(), hybridization),
      weight_(weight),
      edge_d_td_(edge_d_td),
      edge_boundary_att_(edge_boundary_att),
      ess_attr_(ess_attr),
      coarse_components_(coarse_components),
      saamge_param_(saamge_param)
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

    coarsener_ = make_unique<SpectralAMG_MGL_Coarsener>(
                     mixed_laplacians_[0], std::move(graph_topology),
                     spect_tol, max_evects, dual_target, scaled_dual, energy_dual,
                     coarse_components_);
    coarsener_->construct_coarse_subspace();

    mixed_laplacians_.push_back(coarsener_->GetCoarse());

    MakeCoarseSolver();

    MakeCoarseVectors();

    chrono.Stop();
    setup_time_ += chrono.RealTime();
}

FiniteVolumeMLMC::FiniteVolumeMLMC(MPI_Comm comm,
                                   const mfem::SparseMatrix& vertex_edge,
                                   const std::vector<mfem::Vector>& local_weight,
                                   const mfem::Array<int>& partitioning,
                                   const mfem::HypreParMatrix& edge_d_td,
                                   const mfem::SparseMatrix& edge_boundary_att,
                                   const mfem::Array<int>& ess_attr,
                                   double spect_tol, int max_evects,
                                   bool dual_target, bool scaled_dual,
                                   bool energy_dual, bool hybridization,
                                   bool coarse_components,
                                   const SAAMGeParam* saamge_param)
    :
    Upscale(comm, vertex_edge.Height(), hybridization),
    weight_(local_weight[0]),
    edge_d_td_(edge_d_td),
    edge_boundary_att_(edge_boundary_att),
    ess_attr_(ess_attr),
    coarse_components_(coarse_components),
    saamge_param_(saamge_param)
{
    mfem::StopWatch chrono;
    chrono.Start();

    // Hypre may modify the original vertex_edge, which we seek to avoid
    mfem::SparseMatrix ve_copy(vertex_edge);

    mixed_laplacians_.emplace_back(vertex_edge, local_weight, edge_d_td_);

    auto graph_topology = make_unique<GraphTopology>(ve_copy, edge_d_td_, partitioning,
                                                     &edge_boundary_att_);

    c2f_normal_flip_ = BuildCoarseToFineNormalFlip(*graph_topology, vertex_edge);

    {
        mfem::SparseMatrix agg_average_tmp(graph_topology->Agg_vertex_);
        agg_average_.Swap(agg_average_tmp);
        mfem::Vector agg_inv_sizes(agg_average_.Height());
        for (int i = 0; i < agg_average_.Height(); i++)
        {
            agg_inv_sizes(i) = 1.0 / agg_average_.RowSize(i);
        }
        agg_average_.ScaleRows(agg_inv_sizes);
    }


    coarsener_ = make_unique<SpectralAMG_MGL_Coarsener>(
                     mixed_laplacians_[0], std::move(graph_topology),
                     spect_tol, max_evects, dual_target, scaled_dual, energy_dual,
                     coarse_components_);
    coarsener_->construct_coarse_subspace();

    mixed_laplacians_.push_back(coarsener_->GetCoarse());

    MakeCoarseSolver();

    MakeCoarseVectors();

    chrono.Stop();
    setup_time_ += chrono.RealTime();
}

void FiniteVolumeMLMC::RescaleCoefficient(int level, const mfem::Vector& coeff)
{
    if (level == 0)
    {
        RescaleFineCoefficient(coeff);
    }
    else
    {
        RescaleCoarseCoefficient(coeff);
    }
}

void FiniteVolumeMLMC::RescaleFineCoefficient(const mfem::Vector& coeff)
{
    GetFineMatrix().UpdateM(coeff);
    if (!hybridization_)
    {
        ForceMakeFineSolver();
    }
    else
    {
        auto hybrid_solver = dynamic_cast<HybridSolver*>(fine_solver_.get());
        assert(hybrid_solver);
        hybrid_solver->UpdateAggScaling(coeff);
    }
}

void BuildCoarseLevelLocalMassMatrix(
        const mfem::SparseMatrix& Agg_fdof,
        const mfem::SparseMatrix& Agg_cdof,
        const mfem::SparseMatrix& Pedges,
        const mfem::SparseMatrix& M,
        std::vector<mfem::DenseMatrix>& CM_el)
{
    assert(Agg_fdof.Height() == Agg_cdof.Height());
    assert(Agg_fdof.Width() == Pedges.Height());
    assert(Agg_cdof.Width() == Pedges.Width());

    int nAgg = Agg_fdof.Height();
    CM_el.resize(nAgg);

    mfem::Array<int> colMapper(Pedges.Width());
    colMapper = -1;

    auto edge_fdof_Agg = smoothg::Transpose(Agg_fdof);
    mfem::Array<int> local_edge_fdof, local_edge_cdof;
    int * i_Agg_fdof = Agg_fdof.GetI();
    int * j_Agg_fdof = Agg_fdof.GetJ();
    int * i_Agg_cdof = Agg_cdof.GetI();
    int * j_Agg_cdof = Agg_cdof.GetJ();

    double * M_data = M.GetData();
    mfem::Vector Mloc_v;
    int edge_fdof;
    for (int i = 0; i < nAgg; i++)
    {
        int nlocal_edge_fdof = Agg_fdof.RowSize(i);
        int nlocal_edge_cdof = Agg_cdof.RowSize(i);
        local_edge_fdof.MakeRef(j_Agg_fdof+i_Agg_fdof[i],
                nlocal_edge_fdof);
        local_edge_cdof.MakeRef(j_Agg_cdof+i_Agg_cdof[i],
                nlocal_edge_cdof);
        auto Ploc = ExtractRowAndColumns(Pedges, local_edge_fdof,
                                         local_edge_cdof, colMapper);
        Mloc_v.SetSize(nlocal_edge_fdof);
        for (int j = 0; j < nlocal_edge_fdof; j++)
        {
            edge_fdof = local_edge_fdof[j];
            if (edge_fdof_Agg.RowSize(edge_fdof) == 2)
                Mloc_v(j) = M_data[edge_fdof]/2;
            else
                Mloc_v(j) = M_data[edge_fdof];
        }

        std::unique_ptr<mfem::SparseMatrix> CMloc(mfem::Mult_AtDA(Ploc, Mloc_v));
        CM_el[i].SetSize(CMloc->Width());
        Full(*CMloc, CM_el[i]);
    }
}

void FiniteVolumeMLMC::RescaleCoarseCoefficient(const mfem::Vector& coeff)
{
//    GetCoarseMatrix().UpdateM(coeff);
//    if (!hybridization_)
//    {
//        MakeCoarseSolver();
//    }
//    else
//    {
//        auto hybrid_solver = dynamic_cast<HybridSolver*>(coarse_solver_.get());
//        assert(hybrid_solver);
//        hybrid_solver->UpdateAggScaling(coeff);
//    }

    GetFineMatrix().UpdateM(coeff);
    if (!hybridization_)
    {
        std::unique_ptr<mfem::SparseMatrix> M_c(
                    mfem::RAP(coarsener_->get_Psigma(), GetFineMatrix().GetM(), coarsener_->get_Psigma()));
        GetCoarseMatrix().SetM(*M_c);
    }
    else
    {
        std::vector<mfem::DenseMatrix> CM_el;
        BuildCoarseLevelLocalMassMatrix(
                    coarsener_->get_GraphTopology_ref().Agg_alledge_,
                    coarsener_->construct_Agg_cedgedof_table(),
                    coarsener_->get_Psigma(),
                    GetFineMatrix().GetM(),
                    CM_el);
        dynamic_cast<ElementMBuilder&>(GetCoarseMatrix().GetMBuilder()).SetElementM(CM_el);
    }
    MakeCoarseSolver();
}

void FiniteVolumeMLMC::MakeCoarseSolver()
{
    mfem::SparseMatrix& Dref = GetCoarseMatrix().GetD();
    mfem::Array<int> marker(Dref.Width());
    marker = 0;

    MarkDofsOnBoundary(coarsener_->get_GraphTopology_ref().face_bdratt_,
                       coarsener_->construct_face_facedof_table(),
                       ess_attr_, marker);

    for (int i = 0; i < ess_attr_.Size(); i++)
    {
        if (ess_attr_[i] == 0)
        {
            remove_one_dof_ = false;
            break;
        }
    }

    if (hybridization_) // Hybridization solver
    {
        // coarse_components method does not store element matrices
        assert(!coarse_components_);

        auto& face_bdratt = coarsener_->get_GraphTopology_ref().face_bdratt_;
        coarse_solver_ = make_unique<HybridSolver>(
                             comm_, GetCoarseMatrix(), *coarsener_,
                             &face_bdratt, &marker, 0, saamge_param_);
    }
    else // L2-H1 block diagonal preconditioner
    {
        GetCoarseMatrix().BuildM();
        mfem::SparseMatrix& Mref = GetCoarseMatrix().GetM();
        for (int mm = 0; mm < marker.Size(); ++mm)
        {
            // Assume M diagonal, no ess data
            if (marker[mm])
                Mref.EliminateRow(mm, true);
        }
        Dref.EliminateCols(marker);

        coarse_solver_ = make_unique<MinresBlockSolverFalse>(
                    comm_, GetCoarseMatrix(), remove_one_dof_);
    }
}

void FiniteVolumeMLMC::ForceMakeFineSolver()
{
    mfem::Array<int> marker;
    BooleanMult(edge_boundary_att_, ess_attr_, marker);

    if (hybridization_) // Hybridization solver
    {
        fine_solver_ = make_unique<HybridSolver>(comm_, GetFineMatrix(),
                                                 &edge_boundary_att_, &marker);
    }
    else // L2-H1 block diagonal preconditioner
    {
        mfem::SparseMatrix& Mref = GetFineMatrix().GetM();
        mfem::SparseMatrix& Dref = GetFineMatrix().GetD();

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

        fine_solver_ = make_unique<MinresBlockSolverFalse>(
                    comm_, GetFineMatrix(), remove_one_dof_);
    }
}

void FiniteVolumeMLMC::MakeFineSolver()
{
    if (!fine_solver_)
    {
        ForceMakeFineSolver();
    }
}

} // namespace smoothg

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

#include "GraphCoarsenBuilder.hpp"
#include "GraphTopology.hpp"
#include "MatrixUtilities.hpp"

namespace smoothg
{

ElementMBuilder::ElementMBuilder(
    std::vector<mfem::DenseMatrix>& edge_traces,
    std::vector<mfem::DenseMatrix>& vertex_target,
    std::vector<mfem::DenseMatrix>& CM_el,
    const mfem::SparseMatrix& Agg_face,
    int total_num_traces, int ncoarse_vertexdofs)
    :
    CM_el_(CM_el)
{
    total_num_traces_ = total_num_traces;
    const unsigned int nAggs = vertex_target.size();

    CM_el.resize(nAggs);
    mfem::Array<int> faces;
    for (unsigned int i = 0; i < nAggs; i++)
    {
        int nlocal_coarse_dofs = vertex_target[i].Width() - 1;
        GetTableRow(Agg_face, i, faces);
        for (int j = 0; j < faces.Size(); ++j)
            nlocal_coarse_dofs += edge_traces[faces[j]].Width();
        CM_el[i].SetSize(nlocal_coarse_dofs);
    }
    edge_cdof_marker_.SetSize(total_num_traces + ncoarse_vertexdofs - nAggs);
    edge_cdof_marker_ = -1;
}

AssembleMBuilder::AssembleMBuilder(
    std::vector<mfem::DenseMatrix>& vertex_target,
    int total_num_traces, int ncoarse_vertexdofs)
{
    total_num_traces_ = total_num_traces;

    const unsigned int nAggs = vertex_target.size();
    CoarseM_ = make_unique<mfem::SparseMatrix>(
                   total_num_traces + ncoarse_vertexdofs - nAggs,
                   total_num_traces + ncoarse_vertexdofs - nAggs);
}

CoefficientMBuilder::CoefficientMBuilder(
    const GraphTopology& topology,
    mfem::SparseMatrix& Pedges,
    const mfem::SparseMatrix& face_cdof,
    std::vector<mfem::DenseMatrix>& edge_traces,
    std::vector<mfem::DenseMatrix>& vertex_target,
    std::vector<mfem::DenseMatrix>& CM_el,
    int total_num_traces, int ncoarse_vertexdofs)
    :
    topology_(topology),
    Pedges_(Pedges),
    face_cdof_(face_cdof),
    total_num_traces_(total_num_traces),
    ncoarse_vertexdofs_(ncoarse_vertexdofs)
{
}

void ElementMBuilder::RegisterRow(int agg_index, int row, int cdof_loc, int bubble_counter)
{
    agg_index_ = agg_index;
    cdof_loc_ = cdof_loc;
    edge_cdof_marker_[row] = cdof_loc;
}

void AssembleMBuilder::RegisterRow(int agg_index, int row, int cdof_loc, int bubble_counter)
{
    row_ = row;
    bubble_counter_ = bubble_counter;
}

void CoefficientMBuilder::RegisterRow(int agg_index, int row, int cdof_loc, int bubble_counter)
{
}

void ElementMBuilder::SetTraceBubbleBlock(int l, double value)
{
    mfem::DenseMatrix& CM_el_loc(CM_el_[agg_index_]);
    CM_el_loc(l, cdof_loc_) = value;
    CM_el_loc(cdof_loc_, l) = value;
}

void AssembleMBuilder::SetTraceBubbleBlock(int l, double value)
{
    const int global_col = total_num_traces_ + bubble_counter_ + l;
    CoarseM_->Set(row_, global_col, value);
    CoarseM_->Set(global_col, row_, value);
}

void CoefficientMBuilder::SetTraceBubbleBlock(int l, double value)
{
}

void ElementMBuilder::AddTraceTraceBlockDiag(double value)
{
    CM_el_[agg_index_](cdof_loc_, cdof_loc_) = value;
}

void AssembleMBuilder::AddTraceTraceBlockDiag(double value)
{
    CoarseM_->Add(row_, row_, value);
}

void CoefficientMBuilder::AddTraceTraceBlockDiag(double value)
{
}

void ElementMBuilder::AddTraceTraceBlock(int l, double value)
{
    mfem::DenseMatrix& CM_el_loc(CM_el_[agg_index_]);
    CM_el_loc(edge_cdof_marker_[l], cdof_loc_) = value;
    CM_el_loc(cdof_loc_, edge_cdof_marker_[l]) = value;
}

void AssembleMBuilder::AddTraceTraceBlock(int l, double value)
{
    CoarseM_->Add(row_, l, value);
    CoarseM_->Add(l, row_, value);
}

void CoefficientMBuilder::AddTraceTraceBlock(int l, double value)
{
}

void ElementMBuilder::SetBubbleBubbleBlock(int l, int j, double value)
{
    mfem::DenseMatrix& CM_el_loc(CM_el_[agg_index_]);
    CM_el_loc(l, j) = value;
    CM_el_loc(j, l) = value;
}

void AssembleMBuilder::SetBubbleBubbleBlock(int l, int j, double value)
{
    const int global_row = total_num_traces_ + bubble_counter_ + l;
    const int global_col = total_num_traces_ + bubble_counter_ + j;
    CoarseM_->Set(global_row, global_col, value);
    CoarseM_->Set(global_col, global_row, value);
}

void CoefficientMBuilder::SetBubbleBubbleBlock(int l, int j, double value)
{
}

void ElementMBuilder::ResetEdgeCdofMarkers(int size)
{
    edge_cdof_marker_.SetSize(size);
    edge_cdof_marker_ = -1;
    edge_cdof_marker2_.SetSize(size);
    edge_cdof_marker2_ = -1;
}

void AssembleMBuilder::ResetEdgeCdofMarkers(int size)
{
}

void CoefficientMBuilder::ResetEdgeCdofMarkers(int size)
{
}

void ElementMBuilder::FillEdgeCdofMarkers(int face_num, const mfem::SparseMatrix& face_Agg,
                                          const mfem::SparseMatrix& Agg_cdof_edge)
{
    mfem::Array<int> Aggs;
    mfem::Array<int> local_Agg_edge_cdof;
    GetTableRow(face_Agg, face_num, Aggs);
    Agg0_ = Aggs[0];
    GetTableRow(Agg_cdof_edge, Agg0_, local_Agg_edge_cdof);
    for (int k = 0; k < local_Agg_edge_cdof.Size(); k++)
    {
        edge_cdof_marker_[local_Agg_edge_cdof[k]] = k;
    }
    if (Aggs.Size() == 2)
    {
        Agg1_ = Aggs[1];
        GetTableRow(Agg_cdof_edge, Agg1_, local_Agg_edge_cdof);
        for (int k = 0; k < local_Agg_edge_cdof.Size(); k++)
        {
            edge_cdof_marker2_[local_Agg_edge_cdof[k]] = k;
        }
    }
    else
    {
        Agg1_ = -1;
    }
}

void AssembleMBuilder::FillEdgeCdofMarkers(int face_num, const mfem::SparseMatrix& face_Agg,
                                           const mfem::SparseMatrix& Agg_cdof_edge)
{
}

void CoefficientMBuilder::FillEdgeCdofMarkers(int face_num, const mfem::SparseMatrix& face_Agg,
                                              const mfem::SparseMatrix& Agg_cdof_edge)
{
}

void ElementMBuilder::AddTraceAcross(int row, int col, double value)
{
    mfem::DenseMatrix& CM_el_loc1(CM_el_[Agg0_]);

    int id0_in_Agg0 = edge_cdof_marker_[row];
    int id1_in_Agg0 = edge_cdof_marker_[col];
    if (Agg1_ == -1)
    {
        CM_el_loc1(id0_in_Agg0, id1_in_Agg0) += value;
    }
    else
    {
        mfem::DenseMatrix& CM_el_loc2(CM_el_[Agg1_]);
        CM_el_loc1(id0_in_Agg0, id1_in_Agg0) += value / 2.;
        int id0_in_Agg1 = edge_cdof_marker2_[row];
        int id1_in_Agg1 = edge_cdof_marker2_[col];
        CM_el_loc2(id0_in_Agg1, id1_in_Agg1) += value / 2.;
    }
}

void AssembleMBuilder::AddTraceAcross(int row, int col, double value)
{
    CoarseM_->Add(row, col, value);
}

void CoefficientMBuilder::AddTraceAcross(int row, int col, double value)
{
}

std::unique_ptr<mfem::SparseMatrix> ElementMBuilder::GetCoarseM()
{
    return std::unique_ptr<mfem::SparseMatrix>(nullptr);
}

std::unique_ptr<mfem::SparseMatrix> AssembleMBuilder::GetCoarseM()
{
    CoarseM_->Finalize(0);
    return std::move(CoarseM_);
}

void CoefficientMBuilder::SetCoefficient(const mfem::Vector& agg_weights)
{
    agg_weights_ = agg_weights;
}

/// this method may be unnecessary?
void CoefficientMBuilder::GetCoarseFaceDofs(int face, mfem::Array<int>& local_coarse_dofs)
{
    GetTableRow(face_cdof_, face, local_coarse_dofs);
}

mfem::DenseMatrix CoefficientMBuilder::RTP(const mfem::DenseMatrix& R,
                                           const mfem::DenseMatrix& P)
{
    mfem::DenseMatrix out(R.Width(), P.Width());
    mfem::DenseMatrix Rt;
    Rt.Transpose(R);
    Mult(Rt, P, out);
    return out;
}

void CoefficientMBuilder::BuildComponents()
{
    // indexing!

    // F_F block
    const int num_faces = topology_.Agg_face_.Width();
    const int num_aggs = topology_.Agg_face_.Height();
    mfem::Array<int> local_fine_dofs;
    mfem::Array<int> local_coarse_dofs;
    comp_F_F_.resize(num_faces);
    for (int face=0; face<num_faces; ++face)
    {
        mfem::DenseMatrix P_F;
        GetCoarseFaceDofs(face, local_coarse_dofs);
        GetTableRow(topology_.face_edge_, face, local_fine_dofs);
        Pedges_.GetSubMatrix(local_fine_dofs, local_coarse_dofs, P_F);
        comp_F_F_[face] = RTP(P_F, P_F);
    }

    // the EF_EF block
    // for (pairs of *faces* that share an *aggregate*)
    mfem::Array<int> local_faces;
    mfem::Array<int> local_fine_dofs_prime;
    mfem::Array<int> local_coarse_dofs_prime;
    // not sure what to do here: map pairs of faces to indices?
    std::map< std::pair<int,int>, int> face_pair_to_index;
    int counter = 0;
    for (int agg=0; agg<num_aggs; ++agg)
    {
        GetTableRow(topology_.Agg_face_, agg, local_faces);
        for (int f=0; f<local_faces.Size(); ++f)
        {
            int face = local_faces[f];
            GetTableRow(topology_.face_edge_, face, local_fine_dofs);
            GetCoarseFaceDofs(face, local_coarse_dofs);
            mfem::DenseMatrix P_EF;
            Pedges_.GetSubMatrix(local_fine_dofs, local_coarse_dofs, P_EF);
            for (int fprime=f+1; fprime<local_faces.Size(); ++fprime)
            {
                int faceprime = local_faces[fprime];
                GetTableRow(topology_.face_edge_, faceprime, local_fine_dofs_prime);
                GetCoarseFaceDofs(faceprime, local_coarse_dofs_prime);
                mfem::DenseMatrix P_EFprime;
                Pedges_.GetSubMatrix(local_fine_dofs, local_coarse_dofs, P_EFprime);
                comp_EF_EF_.push_back(RTP(P_EF, P_EFprime));
                auto pair = std::make_pair(face, faceprime);
                face_pair_to_index[pair] = counter++;
            }
        }
    }

    // EF_E block and E_E block
    comp_E_E_.resize(num_aggs);
    for (int agg=0; agg<num_aggs; ++agg)
    {
        GetTableRow(topology_.Agg_edge_, agg, local_fine_dofs);
        mfem::DenseMatrix P_E;
        Pedges_.GetSubMatrix(local_fine_dofs, local_fine_dofs, P_E);
        comp_E_E_[agg] = RTP(P_E, P_E);
        GetTableRow(topology_.Agg_face_, agg, local_faces);
        for (int af=0; af<local_faces.Size(); ++af)
        {
            int face = local_faces[af];
            GetCoarseFaceDofs(face, local_coarse_dofs);
            mfem::DenseMatrix P_EF;
            Pedges_.GetSubMatrix(local_fine_dofs, local_coarse_dofs, P_EF);
            // int index = -1; // ??? todo, how to index these guys
            // comp_EF_E[index] = RTP(P_EF, P_E);
            comp_EF_E_.push_back(RTP(P_EF, P_E));
            // also store transpose, or just have it implicitly?
        }
    }
}

std::unique_ptr<mfem::SparseMatrix> CoefficientMBuilder::GetCoarseM()
{
    // what we're doing:
    //   1. assemble from components, identity weights, compare to original
    //   2. change interface so we can actually change weights
    //   3. what to do in parallel....

    // build the components...
    BuildComponents();

    // temporarily just use hard-coded weights = 1
    const int num_aggs = topology_.Agg_face_.Height();
    const int num_faces = topology_.Agg_face_.Width();
    mfem::Vector agg_weights(num_aggs);
    agg_weights = 1.0;
    SetCoefficient(agg_weights);

    // assemble from components...
    auto CoarseM = make_unique<mfem::SparseMatrix>(
        total_num_traces_ + ncoarse_vertexdofs_ - num_aggs,
        total_num_traces_ + ncoarse_vertexdofs_ - num_aggs);

    // these data structures for face_Agg etc do not exist, there may be a better way to do this
    auto face_Agg = smoothg::Transpose(topology_.Agg_face_);
    mfem::Array<int> neighbor_aggs;
    mfem::Array<int> local_coarse_dofs;
    for (int face=0; face<num_faces; ++face)
    {
        double face_weight;
        GetTableRow(face_Agg, face, neighbor_aggs);
        MFEM_ASSERT(neighbor_aggs.Size() <= 2, "Face has three or more aggregates!");
        if (neighbor_aggs.Size() == 1)
        {
            face_weight = agg_weights_[neighbor_aggs[0]];
        }
        else
        {
            const double recip = 2.0 / (1.0 / agg_weights_[neighbor_aggs[0]] +
                                        1.0 / agg_weights_[neighbor_aggs[1]]);
            face_weight = 1.0 / recip;
        }
        comp_F_F_[face] *= face_weight;
        GetCoarseFaceDofs(face, local_coarse_dofs);
        CoarseM->AddSubMatrix(local_coarse_dofs, local_coarse_dofs, comp_F_F_[face]);
        comp_F_F_[face] *= 1.0 / face_weight;
    }

    for (int agg=0; agg<num_aggs; ++agg)
    {

    }

    CoarseM->Finalize(0);
    return std::move(CoarseM);
}

Agg_cdof_edge_Builder::Agg_cdof_edge_Builder(std::vector<mfem::DenseMatrix>& edge_traces,
                                             std::vector<mfem::DenseMatrix>& vertex_target,
                                             const mfem::SparseMatrix& Agg_face,
                                             bool build_coarse_relation)
    :
    Agg_dof_nnz_(0),
    build_coarse_relation_(build_coarse_relation)
{
    const unsigned int nAggs = vertex_target.size();

    if (build_coarse_relation_)
    {
        Agg_dof_i_ = new int[nAggs + 1];
        Agg_dof_i_[0] = 0;

        mfem::Array<int> faces; // this is repetitive of InitializePEdgesNNZ
        for (unsigned int i = 0; i < nAggs; i++)
        {
            int nlocal_coarse_dofs = vertex_target[i].Width() - 1;
            GetTableRow(Agg_face, i, faces);
            for (int j = 0; j < faces.Size(); ++j)
                nlocal_coarse_dofs += edge_traces[faces[j]].Width();
            Agg_dof_i_[i + 1] = Agg_dof_i_[i] + nlocal_coarse_dofs;
        }
        Agg_dof_j_ = new int[Agg_dof_i_[nAggs]];
        Agg_dof_d_ = new double[Agg_dof_i_[nAggs]];
        std::fill(Agg_dof_d_, Agg_dof_d_ + Agg_dof_i_[nAggs], 1.);
    }
}

void Agg_cdof_edge_Builder::Register(int k)
{
    if (build_coarse_relation_)
        Agg_dof_j_[Agg_dof_nnz_++] = k;
}

std::unique_ptr<mfem::SparseMatrix> Agg_cdof_edge_Builder::GetAgg_cdof_edge(int rows, int cols)
{
    if (build_coarse_relation_)
    {
        return make_unique<mfem::SparseMatrix>(
                   Agg_dof_i_, Agg_dof_j_, Agg_dof_d_, rows, cols);
    }
    return std::unique_ptr<mfem::SparseMatrix>(nullptr);
}

}

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

    @brief GraphCoarsen class
*/

#include <unordered_map>
#include "GraphCoarsen.hpp"

namespace smoothg
{
GraphCoarsen::GraphCoarsen(const Graph& graph, const MixedMatrix& mgl,
                           int max_evects, double spect_tol)
    : GraphCoarsen(GraphTopology(graph), mgl, max_evects, spect_tol)
{

}

GraphCoarsen::GraphCoarsen(GraphTopology gt, const MixedMatrix& mgl,
                           int max_evects, double spect_tol)
    : gt_(std::move(gt)),
      max_evects_(max_evects), spect_tol_(spect_tol),
      vertex_targets_(gt_.NumAggs()),
      edge_targets_(gt_.NumFaces()),
      agg_ext_sigma_(gt_.NumAggs()),
      D_trace_sum_(gt_.NumAggs())
{
    MPI_Comm comm = mgl.GlobalD().GetComm();

    ParMatrix v_vdof(comm, mgl.vertex_vdof);
    ParMatrix v_edof(comm, mgl.vertex_edof);
    ParMatrix v_bdof(comm, mgl.vertex_bdof);
    ParMatrix e_edof(comm, mgl.edge_edof);

    ParMatrix true_edge_edge = gt_.edge_true_edge_.Transpose();
    ParMatrix no_bub = gt_.agg_ext_edge_.Mult(true_edge_edge).Mult(e_edof).Mult(mgl.EdgeTrueEdge());
    ParMatrix with_bub = gt_.agg_ext_vertex_.Mult(v_bdof).Mult(mgl.EdgeTrueEdge());

    agg_ext_edof_ = parlinalgcpp::ParAdd(no_bub, with_bub);
    agg_ext_vdof_ = gt_.agg_ext_vertex_.Mult(v_vdof);

    ParMatrix permute_v = MakeExtPermutation(gt_.agg_ext_vertex_.Mult(v_vdof));
    ParMatrix permute_e = MakeExtPermutation(agg_ext_edof_);
    ParMatrix permute_e_T = permute_e.Transpose();

    ParMatrix M_ext_global = permute_e.Mult(mgl.GlobalM().Mult(permute_e_T));
    ParMatrix D_ext_global = permute_v.Mult(mgl.GlobalD().Mult(permute_e_T));

    ParMatrix face_perm_edge = gt_.face_edge_.Mult(e_edof).Mult(mgl.EdgeTrueEdge()).Mult(permute_e_T);

    int marker_size = std::max(permute_v.Rows(), permute_e.Rows());
    col_marker_.resize(marker_size, -1);

    SparseMatrix agg_edgedof = gt_.agg_edge_local_.Mult(mgl.edge_edof);
    SparseMatrix agg_bubbledof = gt_.agg_vertex_local_.Mult(mgl.vertex_bdof);

    agg_edgedof_ = Add(1.0, agg_edgedof, 1.0, agg_bubbledof);
    agg_vertexdof_ = gt_.agg_vertex_local_.Mult(mgl.vertex_vdof);
    face_edgedof_ = gt_.face_edge_local_.Mult(mgl.edge_edof);

    // added code

    const SparseMatrix& face_edge = face_perm_edge.GetDiag();
    std::vector<SparseMatrix> tdof_sorts(gt_.face_true_face_.Rows());
    for (unsigned int face = 0; face < tdof_sorts.size(); ++face)
    {
        if (gt_.face_face_.GetOffd().RowSize(face)) // face is shared
        {
            const std::vector<int> edges = face_edge.GetIndices(face);
            bool is_owned = gt_.face_true_face_.GetDiag().RowSize(face);
            auto& permute_map = is_owned ? permute_e.GetDiag() : permute_e.GetOffd();

            std::map<int, int> tdof_map;
            for (unsigned int i = 0; i < edges.size(); ++i)
            {
                int dof = edges[i];
                tdof_map[permute_map.GetIndices(dof)[0]] = i;
            }

            CooMatrix tdof_sort(edges.size());
            int count = 0;
            for (auto it = tdof_map.begin(); it != tdof_map.end(); ++it)
            {
                tdof_sort.Add(count++, it->second, 1.0);
            }
            tdof_sorts[face] = tdof_sort.ToSparse();
        }
    }

    // end added code

    ComputeVertexTargets(M_ext_global, D_ext_global);
    ComputeEdgeTargets(mgl, face_perm_edge, tdof_sorts);

    BuildFaceCoarseDof();
    BuildAggBubbleDof();

    BuildPvertex();

    BuildQedge(mgl);
    BuildPedge(mgl);

    DebugChecks(mgl);

    std::vector<DenseMatrix>().swap(agg_ext_sigma_);
    std::vector<DenseMatrix>().swap(edge_targets_);
    agg_ext_vdof_ = ParMatrix();
    agg_ext_edof_ = ParMatrix();
}

GraphCoarsen::GraphCoarsen(const GraphCoarsen& other) noexcept
    : gt_(other.gt_),
      max_evects_(other.max_evects_),
      spect_tol_(other.spect_tol_),
      Q_edge_(other.Q_edge_),
      P_edge_(other.P_edge_),
      P_vertex_(other.P_vertex_),
      face_cdof_(other.face_cdof_),
      agg_bubble_dof_(other.agg_bubble_dof_),
      vertex_targets_(other.vertex_targets_),
      edge_targets_(other.edge_targets_),
      agg_ext_sigma_(other.agg_ext_sigma_),
      // Tmp stuff
      agg_vertexdof_(other.agg_vertexdof_),
      agg_edgedof_(other.agg_edgedof_),
      face_edgedof_(other.face_edgedof_),
      agg_ext_vdof_(other.agg_ext_vdof_),
      agg_ext_edof_(other.agg_ext_edof_),
      // End tmp stuff
      col_marker_(other.col_marker_),
      D_trace_sum_(other.D_trace_sum_)
{

}

GraphCoarsen::GraphCoarsen(GraphCoarsen&& other) noexcept
{
    swap(*this, other);
}

GraphCoarsen& GraphCoarsen::operator=(GraphCoarsen other) noexcept
{
    swap(*this, other);

    return *this;
}

void swap(GraphCoarsen& lhs, GraphCoarsen& rhs) noexcept
{
    swap(lhs.gt_, rhs.gt_);

    std::swap(lhs.max_evects_, rhs.max_evects_);
    std::swap(lhs.spect_tol_, rhs.spect_tol_);

    swap(lhs.Q_edge_, rhs.Q_edge_);
    swap(lhs.P_edge_, rhs.P_edge_);
    swap(lhs.P_vertex_, rhs.P_vertex_);
    swap(lhs.face_cdof_, rhs.face_cdof_);
    swap(lhs.agg_bubble_dof_, rhs.agg_bubble_dof_);

    swap(lhs.vertex_targets_, rhs.vertex_targets_);
    swap(lhs.edge_targets_, rhs.edge_targets_);
    swap(lhs.agg_ext_sigma_, rhs.agg_ext_sigma_);

    swap(lhs.col_marker_, rhs.col_marker_);

    swap(lhs.D_trace_sum_, rhs.D_trace_sum_);

    // Temp Stuff
    swap(lhs.agg_vertexdof_, rhs.agg_vertexdof_);
    swap(lhs.agg_edgedof_, rhs.agg_edgedof_);
    swap(lhs.face_edgedof_, rhs.face_edgedof_);

    swap(lhs.agg_ext_vdof_, rhs.agg_ext_vdof_);
    swap(lhs.agg_ext_edof_, rhs.agg_ext_edof_);
    // End Temp Stuff
}

void MakeTrueDofSortingMap()
{

}

void GraphCoarsen::ComputeVertexTargets(const ParMatrix& M_ext_global,
                                        const ParMatrix& D_ext_global)
{
    const SparseMatrix& M_ext = M_ext_global.GetDiag();
    const SparseMatrix& D_ext = D_ext_global.GetDiag();

    int num_aggs = gt_.NumAggs();

    LocalEigenSolver eigs(max_evects_, spect_tol_);

    DenseMatrix evects;
    DenseMatrix DT_evect;

    for (int agg = 0; agg < num_aggs; ++agg)
    {
        std::vector<int> edge_dofs_ext = GetExtDofs(agg_ext_edof_, agg);
        std::vector<int> vertex_dofs_ext = GetExtDofs(agg_ext_vdof_, agg);

        std::vector<int> vertex_dofs_local = agg_vertexdof_.GetIndices(agg);

        if (edge_dofs_ext.size() == 0)
        {
            vertex_targets_[agg] = DenseMatrix(1, 1, {1.0});
            continue;
        }

        SparseMatrix M_sub = M_ext.GetSubMatrix(edge_dofs_ext, edge_dofs_ext, col_marker_);
        SparseMatrix D_sub = D_ext.GetSubMatrix(vertex_dofs_ext, edge_dofs_ext, col_marker_);

        eigs.BlockCompute(M_sub, D_sub, evects);

//        if (evects.Cols() > 1)
//        {
//            SparseSolver M_inv(std::move(M_sub));

//            OffsetMultAT(D_sub, evects, DT_evect, 1);
//            OffsetMult(M_inv, DT_evect, agg_ext_sigma_[agg], 0);
//        }

        DenseMatrix evects_restricted = RestrictLocal(evects, col_marker_,
                vertex_dofs_ext, vertex_dofs_local);

        VectorView first_vect = evects_restricted.GetColView(0);
        vertex_targets_[agg] = Orthogonalize(evects_restricted, first_vect, 1, max_evects_);
    }
}

std::vector<std::vector<DenseMatrix>> GraphCoarsen::CollectSigma(
        const SparseMatrix& face_edgedof, const std::vector<SparseMatrix>& tdof_sorts)
{
    SharedEntityComm<DenseMatrix> sec_sigma(gt_.face_true_face_);

    int num_faces = gt_.NumFaces();

    for (int face = 0; face < num_faces; ++face)
    {
        std::vector<int> face_dofs = face_edgedof.GetIndices(face);
        std::vector<int> neighbors = gt_.face_agg_local_.GetIndices(face);

        int total_vects = 0;
        int col_count = 0;

        for (auto agg : neighbors)
        {
            total_vects += agg_ext_sigma_[agg].Cols();
        }

        DenseMatrix face_sigma(face_dofs.size(), total_vects);

        for (auto agg : neighbors)
        {
            if (agg_ext_sigma_[agg].Cols() > 0)
            {
                std::vector<int> edge_dofs_ext = GetExtDofs(agg_ext_edof_, agg);

                DenseMatrix face_restrict = RestrictLocal(agg_ext_sigma_[agg], col_marker_,
                                                          edge_dofs_ext, face_dofs);

                if (neighbors.size() == 1)
                {
                    face_restrict = tdof_sorts[face].Mult(face_restrict);
                }

                face_sigma.SetCol(col_count, face_restrict);
                col_count += face_restrict.Cols();
            }
        }

        assert(col_count == total_vects);

        sec_sigma.ReduceSend(face, std::move(face_sigma));
    }

    return sec_sigma.Collect();
}

std::vector<std::vector<Vector>> GraphCoarsen::CollectConstant(const MixedMatrix& mgl)
{
    const auto& fine_constant = mgl.constant_vect_;

    SharedEntityComm<Vector> sec_constant(gt_.face_true_face_);

    int num_faces = gt_.NumFaces();

    for (int face = 0; face < num_faces; ++face)
    {
        std::vector<int> neighbors = gt_.face_agg_local_.GetIndices(face);
        std::vector<double> constant_data;

        for (auto agg : neighbors)
        {
            std::vector<int> agg_vertices = agg_vertexdof_.GetIndices(agg);
            auto sub_vect = fine_constant.GetSubVector(agg_vertices);

            constant_data.insert(std::end(constant_data), std::begin(sub_vect),
                                   std::end(sub_vect));
        }

        sec_constant.ReduceSend(face, Vector(std::move(constant_data)));
    }

    return sec_constant.Collect();
}

std::vector<std::vector<SparseMatrix>> GraphCoarsen::CollectD(const MixedMatrix& mgl)
{
    const auto& D_local = mgl.LocalD();

    SharedEntityComm<SparseMatrix> sec_D(gt_.face_true_face_);

    int num_faces = gt_.NumFaces();

    for (int face = 0; face < num_faces; ++face)
    {
        std::vector<int> vertex_ext_dofs;
        std::vector<int> edge_ext_dofs = face_edgedof_.GetIndices(face);
        std::vector<int> neighbors = gt_.face_agg_local_.GetIndices(face);

        for (auto agg : neighbors)
        {
            std::vector<int> agg_edges = agg_edgedof_.GetIndices(agg);
            edge_ext_dofs.insert(std::end(edge_ext_dofs), std::begin(agg_edges),
                                 std::end(agg_edges));

            std::vector<int> agg_vertices = agg_vertexdof_.GetIndices(agg);
            vertex_ext_dofs.insert(std::end(vertex_ext_dofs), std::begin(agg_vertices),
                                   std::end(agg_vertices));
        }

        SparseMatrix D_face = D_local.GetSubMatrix(vertex_ext_dofs, edge_ext_dofs, col_marker_);

        sec_D.ReduceSend(face, std::move(D_face));
    }

    return sec_D.Collect();
}

std::vector<std::vector<SparseMatrix>> GraphCoarsen::CollectM(const SparseMatrix& M_local)
{
    SharedEntityComm<SparseMatrix> sec_M(gt_.face_true_face_);

    int num_faces = gt_.NumFaces();

    for (int face = 0; face < num_faces; ++face)
    {
        std::vector<int> edge_ext_dofs = face_edgedof_.GetIndices(face);
        std::vector<int> neighbors = gt_.face_agg_local_.GetIndices(face);

        for (auto agg : neighbors)
        {
            std::vector<int> agg_edges = agg_edgedof_.GetIndices(agg);
            edge_ext_dofs.insert(std::end(edge_ext_dofs), std::begin(agg_edges),
                                 std::end(agg_edges));
        }

        SparseMatrix M_face = M_local.GetSubMatrix(edge_ext_dofs, edge_ext_dofs, col_marker_);

        sec_M.ReduceSend(face, std::move(M_face));
    }

    return sec_M.Collect();
}

void GraphCoarsen::ComputeEdgeTargets(const MixedMatrix& mgl,
                                      const ParMatrix& face_perm_edge,
                                      const std::vector<SparseMatrix>& tdof_sorts)
{
    const SparseMatrix& face_edge = face_perm_edge.GetDiag();

    auto shared_sigma = CollectSigma(face_edge, tdof_sorts);
    auto shared_constant = CollectConstant(mgl);
    auto shared_M = CollectM(mgl.LocalM());
    auto shared_D = CollectD(mgl);

    const SparseMatrix& face_shared = gt_.face_face_.GetOffd();

    SharedEntityComm<DenseMatrix> sec_face(gt_.face_true_face_);
    DenseMatrix collected_sigma;

    int num_faces = gt_.NumFaces();

    for (int face = 0; face < num_faces; ++face)
    {
        int num_face_edges = face_edge.RowSize(face);

        if (!sec_face.IsOwnedByMe(face))
        {
            continue;
        }

        if (num_face_edges == 1)
        {
            edge_targets_[face] = DenseMatrix(1, 1, {1.0});
            continue;
        }

        auto& face_M = shared_M[face];
        auto& face_D = shared_D[face];
        auto& face_sigma = shared_sigma[face];
        auto& face_constant = shared_constant[face];

        linalgcpp::HStack(face_sigma, collected_sigma);

        bool shared = face_shared.RowSize(face) > 0;

        int split = shared ? face_D[0].Rows() : GetSplit(face);
        SparseMatrix M_local = shared ? CombineM(face_M, num_face_edges) : std::move(face_M[0]);
        SparseMatrix D_local = shared ? CombineD(face_D, num_face_edges) : std::move(face_D[0]);
        Vector constant_local = shared ? CombineConstant(face_constant) : std::move(face_constant[0]);


        GraphEdgeSolver solver(std::move(M_local), std::move(D_local));
        Vector one_neg_one = MakeOneNegOne(constant_local, split);

        if (std::fabs(constant_local.Mult(one_neg_one)) > 1e-8)
        {
            printf("%d constnat * one neg one : %.8e\n", MyId(), constant_local.Mult(one_neg_one) );
        }

        Vector pv_sol = solver.Mult(one_neg_one);
        VectorView pv_sigma(pv_sol.begin(), num_face_edges);

        edge_targets_[face] = Orthogonalize(collected_sigma, pv_sigma, 0, max_evects_);
    }

    sec_face.Broadcast(edge_targets_);

    for (int face = 0; face < num_faces; ++face)
    {
        if (face_shared.RowSize(face) > 0)
        {
            edge_targets_[face] = tdof_sorts[face].MultAT(edge_targets_[face]);
        }
    }

    ScaleEdgeTargets(mgl);
}

void GraphCoarsen::ScaleEdgeTargets(const MixedMatrix& mgl)
{
    const auto& D_local = mgl.LocalD();
    int num_faces = gt_.NumFaces();

    Vector one;
    Vector oneD;

    DenseMatrix sigma_face;

    for (int face = 0; face < num_faces; ++face)
    {
        DenseMatrix& edge_traces(edge_targets_[face]);

        assert(edge_traces.Cols() >= 1);

        int agg = gt_.face_agg_local_.GetIndices(face)[0];

        std::vector<int> vertices = agg_vertexdof_.GetIndices(agg);
        std::vector<int> face_dofs = face_edgedof_.GetIndices(face);

        SparseMatrix D_transfer = D_local.GetSubMatrix(vertices, face_dofs, col_marker_);

        mgl.constant_vect_.GetSubVector(vertices, one);

        oneD.SetSize(D_transfer.Cols());
        D_transfer.MultAT(one, oneD);

        VectorView pv_trace = edge_traces.GetColView(0);

        double oneDpv = oneD.Mult(pv_trace);
        double beta = (oneDpv < 0) ? -1.0 : 1.0;
        oneDpv *= beta;

        pv_trace /= oneDpv;

        int num_traces = edge_traces.Cols();

        for (int k = 1; k < num_traces; ++k)
        {
            VectorView trace = edge_traces.GetColView(k);
            double alpha = oneD.Mult(trace);

            trace.Sub(alpha * beta, pv_trace);
        }

//        if (num_traces > 1)
//        {
//            edge_traces.GetCol(1, num_traces, sigma_face);
//            sigma_face.SVD();
//            edge_traces.SetCol(1, sigma_face);
//        }
    }
}

SparseMatrix GraphCoarsen::CombineM(const std::vector<SparseMatrix>& face_M, int num_face_edges) const
{
    assert(face_M.size() == 2);

    int size = face_M[0].Rows() + face_M[1].Rows() - num_face_edges;
    int offset = face_M[0].Rows() - num_face_edges;

    CooMatrix M_coo(size, size);
    M_coo.Reserve(face_M[0].nnz() + face_M[1].nnz());

    const auto& M_0_indptr = face_M[0].GetIndptr();
    const auto& M_0_indices = face_M[0].GetIndices();
    const auto& M_0_data = face_M[0].GetData();
    const auto& M_0_rows = face_M[0].Rows();

    for (int i = 0; i < M_0_rows; ++i)
    {
        for (int j = M_0_indptr[i]; j < M_0_indptr[i + 1]; ++j)
        {
            M_coo.Add(i, M_0_indices[j], M_0_data[j]);
        }
    }

    const auto& M_1_indptr = face_M[1].GetIndptr();
    const auto& M_1_indices = face_M[1].GetIndices();
    const auto& M_1_data = face_M[1].GetData();
    const auto& M_1_rows = face_M[1].Rows();

    for (int i = 0; i < M_1_rows; ++i)
    {
        for (int j = M_1_indptr[i]; j < M_1_indptr[i + 1]; ++j)
        {
            int col = M_1_indices[j];

            if (i < num_face_edges && col < num_face_edges)
            {
                M_coo.Add(i, M_1_indices[j], M_1_data[j]);
            }
            else
            {
                M_coo.Add(i + offset, M_1_indices[j] + offset, M_1_data[j]);
            }
        }
    }

    return M_coo.ToSparse();
}

SparseMatrix GraphCoarsen::CombineD(const std::vector<SparseMatrix>& face_D,
                                   int num_face_edges) const
{
    assert(face_D.size() == 2);

    int rows = face_D[0].Rows() + face_D[1].Rows();
    int cols = face_D[0].Cols() + face_D[1].Cols() - num_face_edges;

    std::vector<int> indptr = face_D[0].GetIndptr();
    indptr.insert(std::end(indptr), std::begin(face_D[1].GetIndptr()) + 1,
                  std::end(face_D[1].GetIndptr()));

    int row_start = face_D[0].Rows() + 1;
    int row_end = indptr.size();
    int nnz_offset = face_D[0].nnz();

    for (int i = row_start; i < row_end; ++i)
    {
        indptr[i] += nnz_offset;
    }

    std::vector<int> indices = face_D[0].GetIndices();
    indices.insert(std::end(indices), std::begin(face_D[1].GetIndices()),
                   std::end(face_D[1].GetIndices()));

    int col_offset = face_D[0].Cols() - num_face_edges;
    int nnz_end = indices.size();

    for (int i = nnz_offset; i < nnz_end; ++i)
    {
        if (indices[i] >= num_face_edges)
        {
            indices[i] += col_offset;
        }
    }

    std::vector<double> data = face_D[0].GetData();
    data.insert(std::end(data), std::begin(face_D[1].GetData()),
                std::end(face_D[1].GetData()));

    return SparseMatrix(std::move(indptr), std::move(indices), std::move(data),
                        rows, cols);
}

Vector GraphCoarsen::CombineConstant(const std::vector<Vector>& face_constant) const
{
    assert(face_constant.size() == 2);

    int split = face_constant[0].size();
    int size = face_constant[0].size() + face_constant[1].size();

    Vector vect(size);

    std::copy(std::begin(face_constant[0]), std::end(face_constant[0]), std::begin(vect));
    std::copy(std::begin(face_constant[1]), std::end(face_constant[1]), std::begin(vect) + split);

    return vect;
}

Vector GraphCoarsen::MakeOneNegOne(int size, int split) const
{
    assert(size >= 0);
    assert(split >= 0);

    Vector vect(size);

    for (int i = 0; i < split; ++i)
    {
        vect[i] = 1.0 / split;
    }

    for (int i = split; i < size; ++i)
    {
        vect[i] = -1.0 / (size - split);
    }

    return vect;
}

Vector GraphCoarsen::MakeOneNegOne(const Vector& constant, int split) const
{
    assert(split >= 0);

    int size = constant.size();

    Vector vect(size);

    double v1_sum = 0.0;
    double v2_sum = 0.0;

    for (int i = 0; i < split; ++i)
    {
        v1_sum += constant[i] * constant[i];
    }

    for (int i = split; i < size; ++i)
    {
        v2_sum += constant[i] * constant[i];
    }

    double c2 = -1.0 * (v1_sum / v2_sum);

    for (int i = 0; i < split; ++i)
    {
        vect[i] = constant[i];
    }

    for (int i = split; i < size; ++i)
    {
        vect[i] = c2 * constant[i];
    }

    return vect;
}

int GraphCoarsen::GetSplit(int face) const
{
    std::vector<int> neighbors = gt_.face_agg_local_.GetIndices(face);
    assert(neighbors.size() >= 1);
    int agg = neighbors[0];

    return gt_.agg_vertex_local_.RowSize(agg);
}

void GraphCoarsen::BuildFaceCoarseDof()
{
    int num_faces = gt_.NumFaces();

    std::vector<int> indptr(num_faces + 1);
    indptr[0] = 0;

    for (int i = 0; i < num_faces; ++i)
    {
        indptr[i + 1] = indptr[i] + edge_targets_[i].Cols();
    }

    int num_traces = indptr.back();

    std::vector<int> indices(num_traces);
    std::iota(std::begin(indices), std::end(indices), 0);

    std::vector<double> data(num_traces, 1.0);

    face_cdof_ = SparseMatrix(std::move(indptr), std::move(indices), std::move(data),
                              num_faces, num_traces);
}

void GraphCoarsen::BuildAggBubbleDof()
{
    int num_aggs = vertex_targets_.size();

    std::vector<int> indptr(num_aggs + 1);
    indptr[0] = 0;

    for (int i = 0; i < num_aggs; ++i)
    {
        assert(vertex_targets_[i].Cols() >= 1);

        indptr[i + 1] = indptr[i] + vertex_targets_[i].Cols() - 1;
    }

    int num_traces = SumCols(edge_targets_);
    int num_bubbles = indptr.back();

    std::vector<int> indices(num_bubbles);
    std::iota(std::begin(indices), std::end(indices), num_traces);

    std::vector<double> data(num_bubbles, 1.0);

    agg_bubble_dof_ = SparseMatrix(std::move(indptr), std::move(indices), std::move(data),
                                   num_aggs, num_traces + num_bubbles);
}

void GraphCoarsen::BuildPvertex()
{
    const SparseMatrix& agg_vertex = agg_vertexdof_;
    int num_aggs = gt_.NumAggs();
    int num_vertexdofs = agg_vertex.Cols();

    std::vector<int> indptr(num_vertexdofs + 1);
    indptr[0] = 0;

    for (int i = 0; i < num_aggs; ++i)
    {
        std::vector<int> vertices = agg_vertex.GetIndices(i);
        int num_coarse_dofs = vertex_targets_[i].Cols();

        for (auto vertex : vertices)
        {
            indptr[vertex + 1] = num_coarse_dofs;
        }
    }

    for (int i = 0; i < num_vertexdofs; ++i)
    {
        indptr[i + 1] += indptr[i];
    }

    int nnz = indptr.back();
    std::vector<int> indices(nnz);
    std::vector<double> data(nnz);

    int coarse_dof_counter = 0;

    for (int i = 0; i < num_aggs; ++i)
    {
        std::vector<int> fine_dofs = agg_vertex.GetIndices(i);
        int num_fine_dofs = fine_dofs.size();
        int num_coarse_dofs = vertex_targets_[i].Cols();

        const DenseMatrix& target_i = vertex_targets_[i];

        for (int j = 0; j < num_fine_dofs; ++j)
        {
            int counter = indptr[fine_dofs[j]];

            for (int k = 0; k < num_coarse_dofs; ++k)
            {
                indices[counter] = coarse_dof_counter + k;
                data[counter] = target_i(j, k);

                counter++;
            }
        }

        coarse_dof_counter += num_coarse_dofs;
    }

    P_vertex_ = SparseMatrix(std::move(indptr), std::move(indices), std::move(data),
                             num_vertexdofs , coarse_dof_counter);
}

int GraphCoarsen::ComputeEdgeNNZ() const
{
    const SparseMatrix& agg_face = gt_.agg_face_local_;
    const SparseMatrix& agg_edge = agg_edgedof_;
    const SparseMatrix& face_edge = face_edgedof_;

    int num_aggs = gt_.NumAggs();
    int num_faces = gt_.NumFaces();

    int nnz = 0;
    for (int agg = 0; agg < num_aggs; ++agg)
    {
        int edge_dofs = agg_edge.RowSize(agg);
        int bubble_dofs = agg_bubble_dof_.RowSize(agg);

        std::vector<int> faces = agg_face.GetIndices(agg);

        for (auto face : faces)
        {
            int face_coarse_dofs = face_cdof_.RowSize(face);
            nnz += edge_dofs * face_coarse_dofs;
        }

        nnz += edge_dofs * bubble_dofs;
    }

    for (int face = 0; face < num_faces; ++face)
    {
        int face_fine_dofs = face_edge.RowSize(face);
        int face_coarse_dofs = face_cdof_.RowSize(face);

        nnz += face_fine_dofs * face_coarse_dofs;
    }

    return nnz;
}

void GraphCoarsen::BuildPedge(const MixedMatrix& mgl)
{
    const SparseMatrix& agg_face = gt_.agg_face_local_;
    const SparseMatrix& agg_edge = agg_edgedof_;
    const SparseMatrix& face_edge = face_edgedof_;
    const SparseMatrix& agg_vertex = agg_vertexdof_;

    int num_aggs = gt_.NumAggs();
    int num_faces = gt_.NumFaces();
    int num_edgedofs = face_edge.Cols();
    int num_coarse_dofs = agg_bubble_dof_.Cols();

    CooMatrix P_edge(num_edgedofs, num_coarse_dofs);
    P_edge.Reserve(ComputeEdgeNNZ());

    DenseMatrix bubbles;
    DenseMatrix trace_ext;
    DenseMatrix D_trace;
    DenseMatrix M_trace;
    Vector one;

    for (int agg = 0; agg < num_aggs; ++agg)
    {
        std::vector<int> faces = agg_face.GetIndices(agg);
        std::vector<int> edge_dofs = agg_edge.GetIndices(agg);
        std::vector<int> vertex_dofs = agg_vertex.GetIndices(agg);
        std::vector<int> bubble_dofs = agg_bubble_dof_.GetIndices(agg);

        SparseMatrix M = mgl.LocalM().GetSubMatrix(edge_dofs, edge_dofs, col_marker_);
        SparseMatrix D = mgl.LocalD().GetSubMatrix(vertex_dofs, edge_dofs, col_marker_);

        mgl.constant_vect_.GetSubVector(vertex_dofs, one);

        // TODO(gelever1): We may still be able to continue w/ jsut one vertex dof
//        assert (edge_dofs.size() > 0);
        if (edge_dofs.size() == 0)
        {
            for (auto face : faces)
            {
                std::vector<int> face_fine_dofs = face_edge.GetIndices(face);

                SparseMatrix D_transfer = mgl.LocalD().GetSubMatrix(vertex_dofs, face_fine_dofs, col_marker_);
                D_trace.SetSize(D_transfer.Rows(), edge_targets_[face].Cols());
                D_transfer.Mult(edge_targets_[face], D_trace);
                D_trace_sum_[agg].push_back(D_trace.GetColView(0).Mult(vertex_targets_[agg].GetColView(0)));
            }
            continue;
        }

        GraphEdgeSolver solver(std::move(M), std::move(D));

        for (auto face : faces)
        {
            std::vector<int> face_coarse_dofs = face_cdof_.GetIndices(face);
            std::vector<int> face_fine_dofs = face_edge.GetIndices(face);

            SparseMatrix M_transfer = mgl.LocalM().GetSubMatrix(edge_dofs, face_fine_dofs, col_marker_);
            SparseMatrix D_transfer = mgl.LocalD().GetSubMatrix(vertex_dofs, face_fine_dofs, col_marker_);

            M_trace.SetSize(M_transfer.Rows(), edge_targets_[face].Cols());
            D_trace.SetSize(D_transfer.Rows(), edge_targets_[face].Cols());

            M_transfer.Mult(edge_targets_[face], M_trace);
            D_transfer.Mult(edge_targets_[face], D_trace);

            D_trace_sum_[agg].push_back(D_trace.GetColView(0).Mult(vertex_targets_[agg].GetColView(0)));

            OrthoConstant(D_trace, one);

            solver.BlockMult(M_trace, D_trace, trace_ext);
            P_edge.Add(edge_dofs, face_coarse_dofs, trace_ext);
        }

        solver.OffsetMult(1, vertex_targets_[agg], bubbles);
        P_edge.Add(edge_dofs, bubble_dofs, bubbles);
    }

    for (int face = 0; face < num_faces; ++face)
    {
        std::vector<int> face_fine_dofs = face_edge.GetIndices(face);
        std::vector<int> face_coarse_dofs = face_cdof_.GetIndices(face);

        P_edge.Add(face_fine_dofs, face_coarse_dofs, -1.0, edge_targets_[face]);
    }

    //P_edge.EliminateZeros(1e-10);
    P_edge_ = P_edge.ToSparse();
}

void GraphCoarsen::BuildQedge(const MixedMatrix& mgl)
{
    const SparseMatrix& agg_vertex = agg_vertexdof_;
    const SparseMatrix& agg_edge = agg_edgedof_;
    const SparseMatrix& agg_face = gt_.agg_face_local_;
    const SparseMatrix& face_edge = face_edgedof_;
    const SparseMatrix& face_agg = gt_.face_agg_local_;

    int num_aggs = gt_.NumAggs();
    int num_faces = gt_.NumFaces();
    int num_edgedofs = face_edge.Cols();
    int num_coarse_dofs = agg_bubble_dof_.Cols();

    CooMatrix Q_edge(num_edgedofs, num_coarse_dofs);
    Q_edge.Reserve(ComputeEdgeNNZ());

    DenseMatrix Q_i;

    DenseMatrix sigma_f;
    DenseMatrix DT_one_sigma_f_PV, DT_one_pi_f_PV;

    Vector sigma_f_PV;
    Vector one_rep;
    Vector one_D;

    for (int agg = 0; agg < num_aggs; ++agg)
    {
        if (vertex_targets_[agg].Cols() <= 1)
        {
            continue;
        }

        std::vector<int> edge_dofs = agg_edge.GetIndices(agg);
        std::vector<int> vertex_dofs = agg_vertex.GetIndices(agg);
        std::vector<int> faces = agg_face.GetIndices(agg);

        for (auto&& face : faces)
        {
            auto face_dofs = face_edge.GetIndices(face);
            edge_dofs.insert(std::end(edge_dofs), std::begin(face_dofs), std::end(face_dofs));
        }

        SparseMatrix D_transfer = mgl.LocalD().GetSubMatrix(vertex_dofs, edge_dofs, col_marker_);
        OffsetMultAT(D_transfer, vertex_targets_[agg], Q_i, 1);

        std::vector<int> cdofs = agg_bubble_dof_.GetIndices(agg);
        Q_edge.Add(edge_dofs, cdofs, Q_i);
    }

    for (int face = 0; face < num_faces; ++face)
    {
        int agg = face_agg.GetIndices(face)[0];

        std::vector<int> face_fine_dofs = face_edge.GetIndices(face);
        std::vector<int> face_coarse_dofs = face_cdof_.GetIndices(face);
        std::vector<int> vertex_dofs = agg_vertex.GetIndices(agg);

        const auto& trace = edge_targets_[face];
        Vector PV_trace = trace.GetCol(0);

        mgl.constant_vect_.GetSubVector(vertex_dofs, one_rep);

        SparseMatrix D_transfer = mgl.LocalD().GetSubMatrix(vertex_dofs, face_fine_dofs, col_marker_);

        one_D.SetSize(D_transfer.Cols());
        D_transfer.MultAT(one_rep, one_D);

        double one_D_PV = one_D.Mult(PV_trace);
        if (std::fabs(one_D_PV) - 1.0 > 1e-8)
        {
            PV_trace /= one_D_PV;
        }

        Q_i.SetSize(trace.Rows(), trace.Cols());

        if (trace.Cols() > 1)
        {
            trace.GetCol(1, trace.Cols(), sigma_f);

            DenseMatrix sigma_f_T = sigma_f.Transpose();
            DenseMatrix sigma_f_prod = sigma_f_T.Mult(sigma_f);
            sigma_f_prod.Invert();

            DenseMatrix pi_f = sigma_f.Mult(sigma_f_prod);

            Vector pi_f_PV = pi_f.MultAT(PV_trace);
            OuterProduct(one_D, pi_f_PV, DT_one_pi_f_PV);
            DT_one_pi_f_PV /= one_D_PV;

            pi_f -= DT_one_pi_f_PV;

//            sigma_f_PV.SetSize(sigma_f.Cols());
//            sigma_f.MultAT(PV_trace, sigma_f_PV);
//            OuterProduct(one_D, sigma_f_PV, DT_one_sigma_f_PV);
//            DT_one_sigma_f_PV /= one_D_PV;

//            sigma_f -= DT_one_sigma_f_PV;

            Q_i.SetCol(1, pi_f);
        }

        if (one_D_PV < 0)
        {
            one_D *= -1.0;
        }

        Q_i.SetCol(0, one_D);

        Q_edge.Add(face_fine_dofs, face_coarse_dofs, -1.0, Q_i);
    }

    //Q_edge.EliminateZeros(1e-10);
    Q_edge_ = Q_edge.ToSparse();
}

SparseMatrix GraphCoarsen::BuildAggCDofVertex() const
{
    //TODO(gelever1): do this proper
    SparseMatrix agg_cdof_vertex = agg_vertexdof_.Mult(P_vertex_);
    agg_cdof_vertex = 1.0;
    agg_cdof_vertex.SortIndices();

    return agg_cdof_vertex;
}

SparseMatrix GraphCoarsen::BuildAggCDofEdge() const
{
    int num_aggs = gt_.NumAggs();
    int num_traces = face_cdof_.Cols();
    int num_cdofs = P_edge_.Cols();

    std::vector<int> indptr(num_aggs + 1);
    indptr[0] = 0;

    std::vector<int> indices;

    int bubble_counter = 0;

    for (int agg = 0; agg < num_aggs; ++agg)
    {
        int num_bubbles_i = vertex_targets_[agg].Cols() - 1;
        for (int i = 0; i < num_bubbles_i; ++i)
        {
            indices.push_back(num_traces + bubble_counter + i);
        }

        std::vector<int> faces = gt_.agg_face_local_.GetIndices(agg);

        for (auto face : faces)
        {
            std::vector<int> face_cdofs = face_cdof_.GetIndices(face);

            for (auto dof : face_cdofs)
            {
                indices.push_back(dof);
            }
        }

        indptr[agg + 1] = indices.size();

        bubble_counter += num_bubbles_i;
    }

    std::vector<double> data(indices.size(), 1.0);

    return SparseMatrix(std::move(indptr), std::move(indices), std::move(data),
                        num_aggs, num_cdofs);
}

ParMatrix GraphCoarsen::BuildDofTrueDof() const
{
    int num_faces = gt_.NumFaces();
    int num_coarse_dofs = P_edge_.Cols();

    MPI_Comm comm = gt_.face_true_face_.GetComm();
    auto cface_starts = parlinalgcpp::GenerateOffsets(comm, num_coarse_dofs);
    const auto& face_starts = gt_.face_true_face_.GetRowStarts();

    SparseMatrix face_cdof_expand(face_cdof_.GetIndptr(), face_cdof_.GetIndices(),
                                  face_cdof_.GetData(), num_faces, num_coarse_dofs);
    ParMatrix face_cdof_d(comm, face_starts, cface_starts, std::move(face_cdof_expand));

    ParMatrix cface_cface = parlinalgcpp::RAP(gt_.face_face_, face_cdof_d);

    const SparseMatrix& cface_cface_offd = cface_cface.GetOffd();
    const std::vector<int>& cface_cface_colmap = cface_cface.GetColMap();

    std::vector<int> offd_indptr(num_coarse_dofs + 1);
    offd_indptr[0] = 0;

    int offd_nnz = 0;

    for (int i = 0; i < num_coarse_dofs; ++i)
    {
        if (cface_cface_offd.RowSize(i) > 0)
        {
            offd_nnz++;
        }

        offd_indptr[i + 1] = offd_nnz;
    }

    std::vector<int> offd_indices(offd_nnz);
    std::vector<double> offd_data(offd_nnz, 1.0);

    const auto& face_cdof_indptr = face_cdof_.GetIndptr();
    const auto& face_cdof_indices = face_cdof_.GetIndices();

    int col_count = 0;

    for (int i = 0; i < num_faces; ++i)
    {
        if (gt_.face_face_.GetOffd().RowSize(i) > 0)
        {
            int first_dof = face_cdof_indices[face_cdof_indptr[i]];

            std::vector<int> face_cdofs = cface_cface_offd.GetIndices(first_dof);
            assert(static_cast<int>(face_cdofs.size()) == face_cdof_.RowSize(i));

            for (auto cdof : face_cdofs)
            {
                offd_indices[col_count++] = cdof;
            }
        }
    }

    assert(col_count == offd_nnz);
    assert(col_count == static_cast<int>(cface_cface_colmap.size()));

    SparseMatrix d_td_d_diag = SparseIdentity(num_coarse_dofs);
    SparseMatrix d_td_d_offd(std::move(offd_indptr), std::move(offd_indices),
                             std::move(offd_data), num_coarse_dofs, col_count);

    ParMatrix d_td_d(comm, cface_starts, cface_starts,
                     std::move(d_td_d_diag), std::move(d_td_d_offd),
                     cface_cface_colmap);

    return MakeEntityTrueEntity(d_td_d);
}

SparseMatrix GraphCoarsen::BuildCoarseD() const
{
    int num_aggs = gt_.NumAggs();
    int total_traces = face_cdof_.Cols();

    int counter = 0;

    CooMatrix D_coarse(P_vertex_.Cols(), P_edge_.Cols());

    for (int agg = 0; agg < num_aggs; ++agg)
    {
        std::vector<int> faces = gt_.agg_face_local_.GetIndices(agg);
        int num_faces = faces.size();

        for (int j = 0; j < num_faces; ++j)
        {
            double val = -1.0 * D_trace_sum_[agg][j];
            std::vector<int> face_coarse_dofs = face_cdof_.GetIndices(faces[j]);

            D_coarse.Add(counter + agg, face_coarse_dofs[0], val);
        }

        int num_bubbles_i = vertex_targets_[agg].Cols() - 1;

        for (int j = 0; j < num_bubbles_i; ++j)
        {
            D_coarse.Add(counter + agg + 1 + j, total_traces + counter + j, 1.0);
        }

        counter += num_bubbles_i;
    }

    return D_coarse.ToSparse();
}

std::vector<DenseMatrix> GraphCoarsen::BuildElemM(const MixedMatrix& mgl,
                                                  const SparseMatrix& agg_cdof_edge) const
{
    SparseMatrix agg_elem = agg_edgedof_.Mult(mgl.GetElemDof().Transpose());
    int num_aggs = gt_.NumAggs();

    std::vector<DenseMatrix> M_elem(num_aggs);

    DenseMatrix M_loc;
    DenseMatrix M_sub;
    DenseMatrix P_sub;
    DenseMatrix M_tmp;

    for (int i = 0; i < num_aggs; ++i)
    {
        auto agg_dofs = agg_edgedof_.GetIndices(i);
        auto elems = agg_elem.GetIndices(i);
        auto faces = gt_.agg_face_local_.GetIndices(i);

        for (auto&& face : faces)
        {
            auto face_dofs = face_edgedof_.GetIndices(face);
            agg_dofs.insert(std::end(agg_dofs), std::begin(face_dofs), std::end(face_dofs));
        }

        int num_elem = elems.size();
        int num_dofs = agg_dofs.size();

        M_loc.SetSize(num_dofs, num_dofs);
        M_loc = 0.0;

        for (int j = 0; j < num_elem; ++j)
        {
            const auto& M_fine_elem = mgl.GetElemM();
            auto elem = elems[j];
            auto fine_dofs = mgl.GetElemDof().GetIndices(elem);

            SetMarker(col_marker_, fine_dofs);

            std::vector<int> loc_dofs;
            std::vector<int> M_sub_dofs;

            for (int k = 0; k < num_dofs; ++k)
            {
                auto dof = agg_dofs[k];
                auto index = col_marker_[dof];
                if (index >= 0)
                {
                    loc_dofs.push_back(k);
                    M_sub_dofs.push_back(index);
                }
            }

            M_fine_elem[elem].GetSubMatrix(M_sub_dofs, M_sub_dofs, M_sub);
            M_loc.AddSubMatrix(loc_dofs, loc_dofs, M_sub);

            ClearMarker(col_marker_, fine_dofs);
        }

        auto cdofs = agg_cdof_edge.GetIndices(i);

        GetSubMatrix(P_edge_, agg_dofs, cdofs, col_marker_, P_sub);

        M_tmp.SetSize(P_sub.Cols(), M_loc.Cols());
        M_elem[i].SetSize(P_sub.Cols(), P_sub.Cols());

        M_loc.Mult(P_sub, M_tmp);
        P_sub.MultAT(M_tmp, M_elem[i]);
    }

    return M_elem;
}

DenseMatrix GraphCoarsen::RestrictLocal(const DenseMatrix& ext_mat,
                                        std::vector<int>& global_marker,
                                        const std::vector<int>& ext_indices,
                                        const std::vector<int>& local_indices) const
{
    SetMarker(global_marker, ext_indices);

    int local_size = local_indices.size();

    std::vector<int> row_map(local_size);

    for (int i = 0; i < local_size; ++i)
    {
        assert(global_marker[local_indices[i]] >= 0);
        row_map[i] = global_marker[local_indices[i]];
    }

    ClearMarker(global_marker, ext_indices);

    return ext_mat.GetRow(row_map);
}

std::vector<int> GraphCoarsen::GetExtDofs(const ParMatrix& mat_ext, int row) const
{
    const auto& diag = mat_ext.GetDiag();
    const auto& offd = mat_ext.GetOffd();

    auto diag_dofs = diag.GetIndices(row);
    auto offd_dofs = offd.GetIndices(row);

    int diag_size = diag.Cols();

    for (auto i : offd_dofs)
    {
        diag_dofs.push_back(i + diag_size);
    }

    return diag_dofs;
}

ParMatrix GraphCoarsen::MakeExtPermutation(const ParMatrix& parmat) const
{
    MPI_Comm comm = parmat.GetComm();

    const auto& diag = parmat.GetDiag();
    const auto& offd = parmat.GetOffd();
    const auto& colmap = parmat.GetColMap();

    int num_diag = diag.Cols();
    int num_offd = offd.Cols();
    int num_ext = num_diag + num_offd;

    const auto& mat_starts = parmat.GetColStarts();
    auto ext_starts = parlinalgcpp::GenerateOffsets(comm, num_ext);

    SparseMatrix perm_diag = SparseIdentity(num_ext, num_diag);
    SparseMatrix perm_offd = SparseIdentity(num_ext, num_offd, num_diag);

    return ParMatrix(comm, ext_starts, mat_starts, std::move(perm_diag), std::move(perm_offd), colmap);
}

MixedMatrix GraphCoarsen::Coarsen(const MixedMatrix& mgl) const
{
    SparseMatrix agg_cdof_edge = BuildAggCDofEdge();
    auto M_elem = BuildElemM(mgl, agg_cdof_edge);

    SparseMatrix D_c = BuildCoarseD();
    std::vector<std::vector<double>>().swap(D_trace_sum_);
    std::vector<DenseMatrix>().swap(vertex_targets_);

    SparseMatrix W_c;

    if (mgl.LocalW().Rows() == P_vertex_.Rows())
    {
        SparseMatrix P_vertex_T = P_vertex_.Transpose();
        W_c = P_vertex_T.Mult(mgl.LocalW().Mult(P_vertex_));
    }

    ParMatrix edge_true_edge = BuildDofTrueDof();

    MixedMatrix mm(std::move(M_elem), std::move(agg_cdof_edge),
                   std::move(D_c), std::move(W_c),
                   std::move(edge_true_edge),
                   BuildAggCDofVertex(),
                   face_cdof_);

    mm.vertex_vdof = agg_vertexdof_.Mult(P_vertex_);
    mm.vertex_edof = agg_edgedof_.Mult(P_edge_);
    mm.vertex_bdof = agg_bubble_dof_;
    mm.edge_edof = face_edgedof_.Mult(P_edge_);

    mm.vertex_vdof = 1.0;
    mm.vertex_edof = 1.0;
    mm.edge_edof = 1.0;
    mm.constant_vect_ = P_vertex_.MultAT(mgl.constant_vect_);

    mm.num_ess_vdof_ = mgl.num_ess_vdof_;

    gt_ = GraphTopology();
    agg_vertexdof_ = SparseMatrix();
    agg_edgedof_ = SparseMatrix();
    agg_bubble_dof_ = SparseMatrix();
    face_edgedof_ = SparseMatrix();
    face_cdof_ = SparseMatrix();

    return mm;
}

void GraphCoarsen::DebugChecks(const MixedMatrix& mgl) const
{
//    /*
    double test_tol = 1e-10;

    // PTP should be identity
    {
        Vector vertex_test_vect(P_vertex_.Cols());
        vertex_test_vect.Randomize();

        Vector identity_diff = P_vertex_.MultAT(P_vertex_.Mult(vertex_test_vect));
        identity_diff -= vertex_test_vect;

        double norm = identity_diff.L2Norm() / vertex_test_vect.L2Norm();

        if (std::fabs(norm) > test_tol)
        {
            printf("%d Warning: PTP difference %.8e\n", MyId(), norm);
        }
    }

    // PU PU^T D sigma = D P sigma
    {
        Vector edge_test_vect(P_edge_.Cols());
        edge_test_vect.Randomize();

        Vector D_P_sigma = mgl.LocalD().Mult(P_edge_.Mult(edge_test_vect));
        Vector PU_D_P_sigma = P_vertex_.Mult(P_vertex_.MultAT(D_P_sigma));

        Vector edge_diff = PU_D_P_sigma - D_P_sigma;

        double norm = edge_diff.L2Norm() / D_P_sigma.L2Norm();

        if (std::fabs(norm) > test_tol)
        {
            printf("%d Warning: Pu Pu^T D P sigma = D P sigma difference %.8e\n", MyId(), norm);
        }
    }

    // Q^T P = I
    {
        Vector test(P_edge_.Cols());
        test.Randomize();

        Vector diff = Q_edge_.MultAT(P_edge_.Mult(test));
        diff -= test;

        double norm = diff.L2Norm();

        if (std::fabs(norm) > test_tol)
        {
            printf("%d Warning: Q^T Psigma = I difference %.8e\n", MyId(), norm);
        }
    }

    // D (P Q^T)_edge  = (P PT)_vertex D
    {
        Vector test(P_edge_.Rows());
        test.Randomize();

        Vector sigma_proj = mgl.LocalD().Mult(P_edge_.Mult(Q_edge_.MultAT(test)));
        Vector u_proj = P_vertex_.Mult(P_vertex_.MultAT(mgl.LocalD().Mult(test)));

        Vector diff = sigma_proj - u_proj;

        double norm = diff.L2Norm();

        if (std::fabs(norm) > test_tol)
        {
            printf("%d Warning: D PQT = PPT D difference %.8e\n", MyId(), norm);
        }
    }
//    */
}

Vector GraphCoarsen::Interpolate(const VectorView& coarse_vect) const
{
    return P_vertex_.Mult(coarse_vect);
}

void GraphCoarsen::Interpolate(const VectorView& coarse_vect, VectorView fine_vect) const
{
    P_vertex_.Mult(coarse_vect, fine_vect);
}

Vector GraphCoarsen::Restrict(const VectorView& fine_vect) const
{
    return P_vertex_.MultAT(fine_vect);
}

void GraphCoarsen::Restrict(const VectorView& fine_vect, VectorView coarse_vect) const
{
    P_vertex_.MultAT(fine_vect, coarse_vect);
}

BlockVector GraphCoarsen::Interpolate(const BlockVector& coarse_vect) const
{
    std::vector<int> fine_offsets = {0, P_edge_.Rows(), P_edge_.Rows() + P_vertex_.Rows()};
    BlockVector fine_vect(fine_offsets);

    Interpolate(coarse_vect, fine_vect);

    return fine_vect;
}

void GraphCoarsen::Interpolate(const BlockVector& coarse_vect, BlockVector& fine_vect) const
{
    P_edge_.Mult(coarse_vect.GetBlock(0), fine_vect.GetBlock(0));
    P_vertex_.Mult(coarse_vect.GetBlock(1), fine_vect.GetBlock(1));
}

BlockVector GraphCoarsen::Restrict(const BlockVector& fine_vect) const
{
    std::vector<int> coarse_offsets = {0, P_edge_.Cols(), P_edge_.Cols() + P_vertex_.Cols()};
    BlockVector coarse_vect(coarse_offsets);

    Restrict(fine_vect, coarse_vect);

    return coarse_vect;
}

void GraphCoarsen::Restrict(const BlockVector& fine_vect, BlockVector& coarse_vect) const
{
    P_edge_.MultAT(fine_vect.GetBlock(0), coarse_vect.GetBlock(0));
    P_vertex_.MultAT(fine_vect.GetBlock(1), coarse_vect.GetBlock(1));
}

BlockVector GraphCoarsen::Project(const BlockVector& fine_vect) const
{
    std::vector<int> coarse_offsets = {0, Q_edge_.Cols(), Q_edge_.Cols() + P_vertex_.Cols()};
    BlockVector coarse_vect(coarse_offsets);

    Project(fine_vect, coarse_vect);

    return coarse_vect;
}

void GraphCoarsen::Project(const BlockVector& fine_vect, BlockVector& coarse_vect) const
{
    Q_edge_.MultAT(fine_vect.GetBlock(0), coarse_vect.GetBlock(0));
    P_vertex_.MultAT(fine_vect.GetBlock(1), coarse_vect.GetBlock(1));
}


} // namespace smoothg

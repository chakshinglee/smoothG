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

/** @file GraphCoarsen.hpp

    @brief The main graph coarsening routines.
*/

#ifndef __GRAPHCOARSEN_HPP__
#define __GRAPHCOARSEN_HPP__

#include "Utilities.hpp"
#include "LocalEigenSolver.hpp"
#include "MixedMatrix.hpp"
#include "SharedEntityComm.hpp"
#include "GraphTopology.hpp"
#include "GraphEdgeSolver.hpp"
#include "MinresBlockSolver.hpp"

namespace smoothg
{

/** @brief Coarsens a given graph

    This project is intended to take a graph and build a smaller (upscaled)
    graph that is representative of the original in some way. We represent
    the graph Laplacian in a mixed form, solve some local eigenvalue problems
    to uncover near-nullspace modes, and use those modes as coarse degrees
    of freedom.
*/

class GraphCoarsen
{
public:
    /** @brief Default Construtor */
    GraphCoarsen() = default;

    /** @brief Construtor from a fine level graph

        @param graph Fine level graph
        @param mgl Fine level mixed matrix
        @param max_evects maximum number of eigenvectors per aggregate
        @param spect_tol spectral tolerance used to determine how many eigenvectors
                         to keep per aggregate
    */
    GraphCoarsen(const Graph& graph, const MixedMatrix& mgl,
                 int max_evects, double spect_tol);

    /** @brief Construtor from graph topology

        @param gt precomputed graph topology
        @param mgl Fine level mixed matrix
        @param max_evects maximum number of eigenvectors per aggregate
        @param spect_tol spectral tolerance used to determine how many eigenvectors
                         to keep per aggregate
    */
    GraphCoarsen(GraphTopology gt, const MixedMatrix& mgl,
                 int max_evects, double spect_tol);


    /** @brief Default Destructor */
    ~GraphCoarsen() noexcept = default;

    /** @brief Copy Constructor */
    GraphCoarsen(const GraphCoarsen& other) noexcept;

    /** @brief Move Constructor */
    GraphCoarsen(GraphCoarsen&& other) noexcept;

    /** @brief Assignment Operator */
    GraphCoarsen& operator=(GraphCoarsen other) noexcept;

    /** @brief Swap to coarseners */
    friend void swap(GraphCoarsen& lhs, GraphCoarsen& rhs) noexcept;

    /** @brief Create the coarse mixed matrix
        @param mgl Fine level mixed matrix
    */
    MixedMatrix Coarsen(const MixedMatrix& mgl) const;

    /** @brief Interpolate a coarse vertex vector to the fine level
        @param coarse_vect vertex vector to interpolate
        @returns fine_vect interpolated fine level vertex vector
    */
    Vector Interpolate(const VectorView& coarse_vect) const;

    /** @brief Interpolate a coarse vertex vector up to the fine level
        @param coarse_vect vertex vector to interpolate
        @param fine_vect interpolated fine level vertex vector
    */
    void Interpolate(const VectorView& coarse_vect, VectorView fine_vect) const;

    /** @brief Restrict a fine level vertex vector to the coarse level
        @param fine_vect fine level vertex vector
        @returns coarse_vect restricted vertex vector
    */
    Vector Restrict(const VectorView& fine_vect) const;

    /** @brief Restrict a fine level vertex vector to the coarse level
        @param fine_vect fine level vertex vector
        @param coarse_vect restricted vertex vector
    */
    void Restrict(const VectorView& fine_vect, VectorView coarse_vect) const;

    /** @brief Interpolate a coarse mixed form vector to the fine level
        @param coarse_vect mixed form vector to interpolate
        @returns fine_vect interpolated fine level mixed form vector
    */
    BlockVector Interpolate(const BlockVector& coarse_vect) const;

    /** @brief Interpolate a coarse mixed form vector up to the fine level
        @param coarse_vect mixed form vector to interpolate
        @param fine_vect interpolated fine level mixed form vector
    */
    void Interpolate(const BlockVector& coarse_vect, BlockVector& fine_vect) const;

    /** @brief Restrict a fine level mixed form vector to the coarse level
        @param fine_vect fine level mixed form vector
        @returns coarse_vect restricted mixed form vector
    */
    BlockVector Restrict(const BlockVector& fine_vect) const;

    /** @brief Restrict a fine level mixed form vector to the coarse level
        @param fine_vect fine level mixed form vector
        @param coarse_vect restricted mixed form vector
    */
    void Restrict(const BlockVector& fine_vect, BlockVector& coarse_vect) const;

    /** @brief Project a fine level mixed form vector to the coarse level
        @param fine_vect fine level mixed form vector
        @returns coarse_vect projected mixed form vector
    */
    BlockVector Project(const BlockVector& fine_vect) const;

    /** @brief Project a fine level mixed form vector to the coarse level
        @param fine_vect fine level mixed form vector
        @param coarse_vect projected mixed form vector
    */
    void Project(const BlockVector& fine_vect, BlockVector& coarse_vect) const;

    /** @brief Get Graph Topology */
    const GraphTopology& GetGraphTopology() const { return gt_; }

    const SparseMatrix& Qedge() const { return Q_edge_; }
    const SparseMatrix& Pedge() const { return P_edge_; }
    const SparseMatrix& Pvertex() const { return P_vertex_; }
    const GraphTopology& Topology() const { return gt_; }

private:
    template <class T>
    using Vect2D = std::vector<std::vector<T>>;

    void ComputeVertexTargets(const ParMatrix& M_ext, const ParMatrix& D_ext);
    void ComputeEdgeTargets(const MixedMatrix& mgl,
                            const ParMatrix& face_edge_perm,
                            const std::vector<SparseMatrix>& tdof_sorts);
    void ScaleEdgeTargets(const MixedMatrix& mgl);


    Vect2D<DenseMatrix> CollectSigma(const SparseMatrix& face_edge,
                                     const std::vector<SparseMatrix>& tdof_sorts);
    Vect2D<Vector> CollectConstant(const MixedMatrix& mgl);
    Vect2D<SparseMatrix> CollectD(const MixedMatrix& mgl);
    Vect2D<SparseMatrix> CollectM(const SparseMatrix& M_local);

    SparseMatrix CombineM(const std::vector<SparseMatrix>& face_M, int num_face_edges) const;
    SparseMatrix CombineD(const std::vector<SparseMatrix>& face_D, int num_face_edges) const;
    Vector CombineConstant(const std::vector<Vector>& face_rhs) const;

    Vector MakeOneNegOne(const Vector& constant, int split) const;
    Vector MakeOneNegOne(int size, int split) const;

    int GetSplit(int face) const;

    void BuildAggFaceM(const MixedMatrix& mgl, int face, int agg,
                      const SparseMatrix& vertex_agg,
                      const SparseMatrix& edge_vertex,
                      std::vector<int>& col_marker,
                      DenseMatrix& M_local) const;

    void BuildAggBubbleDof();
    void BuildFaceCoarseDof();
    void BuildPvertex();
    void BuildPedge(const MixedMatrix& mgl);
    void BuildQedge(const MixedMatrix& mgl);

    // These only depend on GraphTopology and are sent directly to
    // the coarse mixed matrix
    // {
    SparseMatrix BuildAggCDofVertex() const;
    SparseMatrix BuildAggCDofEdge() const;
    ParMatrix BuildDofTrueDof() const;
    // }

    SparseMatrix BuildCoarseD() const;
    std::vector<DenseMatrix> BuildElemM(const MixedMatrix& mgl,
                                        const SparseMatrix& agg_cdof_edge) const;

    DenseMatrix RestrictLocal(const DenseMatrix& ext_mat,
                              std::vector<int>& global_marker,
                              const std::vector<int>& ext_indices,
                              const std::vector<int>& local_indices) const;

    std::vector<int> GetExtDofs(const ParMatrix& mat_ext, int row) const;

    ParMatrix MakeExtPermutation(const ParMatrix& parmat) const;
    int ComputeEdgeNNZ() const;

    void DebugChecks(const MixedMatrix& mgl) const;

    mutable GraphTopology gt_;

    int max_evects_;
    double spect_tol_;

    SparseMatrix Q_edge_;
    SparseMatrix P_edge_;
    SparseMatrix P_vertex_;
    mutable SparseMatrix face_cdof_;
    mutable SparseMatrix agg_bubble_dof_;

    mutable std::vector<DenseMatrix> vertex_targets_;
    std::vector<DenseMatrix> edge_targets_;
    std::vector<DenseMatrix> agg_ext_sigma_;

    ////////////////////
    // ML Stuff
    mutable SparseMatrix agg_vertexdof_;
    mutable SparseMatrix agg_edgedof_;
    mutable SparseMatrix face_edgedof_;

    ParMatrix agg_ext_vdof_;
    ParMatrix agg_ext_edof_;
    // End ML Stuff
    //////////////////

    mutable std::vector<int> col_marker_;

    mutable std::vector<std::vector<double>> D_trace_sum_;
};

} // namespace smoothg

#endif /* __GRAPHCOARSEN_HPP__ */

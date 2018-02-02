/*bheader**********************************************************************
 *
 * copyright (c) 2017,  lawrence livermore national security, llc.
 * produced at the lawrence livermore national laboratory.
 * llnl-code-xxxxxx. all rights reserved.
 *
 * this file is part of smoothg.  see file copyright for details.
 * for more information and source code availability see xxxxx.
 *
 * smoothg is free software; you can redistribute it and/or modify it under the
 * terms of the gnu lesser general public license (as published by the free
 * software foundation) version 2.1 dated february 1999.
 *
 ***********************************************************************eheader*/

/** @file

    @brief Contains GraphUpscale class
*/

#ifndef __GRAPHUPSCALE_HPP__
#define __GRAPHUPSCALE_HPP__

#include "MinresBlockSolver.hpp"
#include "HybridSolver.hpp"
#include "SpectralAMG_MGL_Coarsener.hpp"
#include "MetisGraphPartitioner.hpp"
#include "MixedMatrix.hpp"
#include "Upscale.hpp"
#include "mfem.hpp"

namespace smoothg
{

/**
   @brief Use upscaling as operator.
*/
class GraphUpscale : public Upscale
{
public:
    /**
       @brief Constructor

       @param comm MPI communicator
       @param vertex_edge relationship between vertices and edge
       @param global_partitioning partition of global vertices
       @param weight edge weights. if not provided, set to all ones
       @param spect_tol spectral tolerance determines how many eigenvectors to
                        keep per aggregate
       @param max_evects maximum number of eigenvectors to keep per aggregate
       @param hybridization use hybridization as solver
    */
    GraphUpscale(MPI_Comm comm,
                 const mfem::SparseMatrix& vertex_edge,
                 const mfem::Array<int>& global_partitioning,
                 double spect_tol = 0.001, int max_evects = 4, bool hybridization = false,
                 const mfem::Vector& weight = mfem::Vector());
    /**
       @brief Constructor

       @param comm MPI communicator
       @param vertex_edge relationship between vertices and edge
       @param coarse_factor how coarse to partition the graph
       @param weight edge weights. if not provided, set to all ones
       @param spect_tol spectral tolerance determines how many eigenvectors to
                        keep per aggregate
       @param max_evects maximum number of eigenvectors to keep per aggregate
       @param hybridization use hybridization as solver
    */
    GraphUpscale(MPI_Comm comm,
                 const mfem::SparseMatrix& vertex_edge,
                 int coarse_factor,
                 double spect_tol = 0.001, int max_evects = 4, bool hybridization = false,
                 const mfem::Vector& weight = mfem::Vector());

    /// Read permuted vertex vector
    mfem::Vector ReadVertexVector(const std::string& filename) const;

    /// Read permuted edge vector
    mfem::Vector ReadEdgeVector(const std::string& filename) const;

    /// Read permuted vertex vector, in mixed form
    mfem::BlockVector ReadVertexBlockVector(const std::string& filename) const;

    /// Read permuted edge vector, in mixed form
    mfem::BlockVector ReadEdgeBlockVector(const std::string& filename) const;

    /// Write permuted vertex vector
    void WriteVertexVector(const mfem::Vector& vect, const std::string& filename) const;

    /// Write permuted edge vector
    void WriteEdgeVector(const mfem::Vector& vect, const std::string& filename) const;

    // Create Fine Level Solver
    void MakeFineSolver() const;

private:
    void Init(const mfem::SparseMatrix& vertex_edge,
              const mfem::Array<int>& global_partitioning,
              const mfem::Vector& weight,
              double spect_tol, int max_evects);

    mfem::Vector ReadVector(const std::string& filename, int global_size,
                            const mfem::Array<int>& local_to_global) const;

    void WriteVector(const mfem::Vector& vect, const std::string& filename, int global_size,
                     const mfem::Array<int>& local_to_global) const;

    std::unique_ptr<smoothg::ParGraph> pgraph_;

    const int global_edges_;
    const int global_vertices_;
};

} // namespace smoothg

#endif /* __GRAPHUPSCALE_HPP__ */

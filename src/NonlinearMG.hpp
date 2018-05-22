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

    @brief Contains class NonlinearMG
*/

#ifndef __NONLINEARMG_HPP__
#define __NONLINEARMG_HPP__

#include "mfem.hpp"

namespace smoothg
{

/**
   @brief Abstract class for operator hierarchies.

   This class provides interface to do actions related to the operator at each
   level as well as restricting/interpolating vectors between consecutive levels

   @note The convention that smaller level means finer level is adopted
*/
class Hierarchy
{
public:
    Hierarchy(MPI_Comm comm, int num_levels) : comm_(comm), num_levels_(num_levels) {}

    /// Evaluates the action of the operator out = A[level](in)
    virtual void Mult(int level, const mfem::Vector& in, mfem::Vector& out) const = 0;

    /// Solves the (possibly nonlinear) problem A[level](sol) = rhs
    virtual void Solve(int level, const mfem::Vector& rhs, mfem::Vector& sol) = 0;

    /// Restrict a vector from level to level+1 (coarser level)
    virtual void Restrict(int level, const mfem::Vector& fine, mfem::Vector& coarse) const = 0;

    /// Interpolate a vector from level to level-1 (finer level)
    virtual void Interpolate(int level, const mfem::Vector& coarse, mfem::Vector& fine) const = 0;

    /// Project a vector from level to level+1 (coarser level)
    virtual void Project(int level, const mfem::Vector& fine, mfem::Vector& coarse) const = 0;

    /// Relaxation on each level
    virtual void Smoothing(int level, const mfem::Vector& in, mfem::Vector& out) const = 0;

    /// @name Get the dimension of domain and range of the operator at each level
    /// @{
    virtual const int DomainSize(int level) const = 0;
    virtual const int RangeSize(int level) const = 0;
    /// @}

    /// Get number of levels of the hierarchy
    const int GetNumLevels() const { return num_levels_; }

    MPI_Comm GetComm() const { return comm_; }
protected:
    MPI_Comm comm_;
    int num_levels_;
};

enum Cycle { V_CYCLE, FMG };

/**
   @brief Nonlinear multigrid using full approximation scheme and nonlinear relaxation.

   @todo remove virtual keyword (I guess they are not necessary)
*/
class NonlinearMG
{
public:
    // the time dependent operators gets updated during solving
    NonlinearMG(Hierarchy& hierarchy, Cycle cycle = Cycle::FMG);
    NonlinearMG() = delete;

    ~NonlinearMG() = default;

    /**
       Solve a nonlinear problem using FAS

       The BlockVectors here are in "dof" numbering, rather than "truedof" numbering.
       That is, dofs on processor boundaries are *repeated* in the vectors that
       come into and go out of this method.
    */
    void Solve(const mfem::Vector& rhs, mfem::Vector& sol);
    void Mult(const mfem::Vector& rhs, mfem::Vector& sol);

    ///@name Set solver parameters
    ///@{
    void SetPrintLevel(int print_level) { print_level_ = print_level; }
    void SetMaxIter(int max_num_iter) { max_num_iter_ = max_num_iter; }
    void SetRelTol(double rtol) { rtol_ = rtol; }
    void SetAbsTol(double atol) { atol_ = atol; }
    ///@}

    ///@name Get results of iterative solve
    ///@{
    int GetNumIterations() const { return num_iterations_; }
    double GetTiming() const { return timing_; }
    bool IsConverged() const { return converged_; }
    ///@}

protected:
    void FAS_FMG();
    void FAS_VCycle(int level);
    double ResidualNorm(const mfem::Vector& sol, const mfem::Vector& rhs) const;

    MPI_Comm comm_;

    // default iterative solver options
    int print_level_ = 0;
    int max_num_iter_ = 10;
    double rtol_ = 1e-3;
    double atol_ = 1e-6;

    int myid_;
    int num_iterations_;
    double timing_;
    int iter_;
    bool converged_;

    Hierarchy& hierarchy_;
    Cycle cycle_;
    int num_levels_;
    std::vector<mfem::Vector> rhs_;
    std::vector<mfem::Vector> sol_;
    mutable std::vector<mfem::Vector> help_;
};

} // namespace smoothg

#endif /* __NONLINEARMG_HPP__ */

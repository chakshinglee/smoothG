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

#include "Utilities.hpp"

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
    virtual void Mult(int level, const VectorView& in, VectorView& out) = 0;

    /// Solves the (possibly nonlinear) problem A[level](sol) = rhs
    virtual void Solve(int level, const VectorView& rhs, VectorView& sol) = 0;

    /// Restrict a vector from level to level+1 (coarser level)
    virtual void Restrict(int level, const VectorView& fine, VectorView& coarse) const = 0;

    /// Interpolate a vector from level to level-1 (finer level)
    virtual void Interpolate(int level, const VectorView& coarse, VectorView& fine) const = 0;

    /// Project a vector from level to level+1 (coarser level)
    virtual void Project(int level, const VectorView& fine, VectorView& coarse) const = 0;

    /// Relaxation on each level
    virtual void Smoothing(int level, const VectorView& in, VectorView& out) const = 0;

    virtual Vector AssembleTrueVector(int level, const VectorView& vec_dof) const = 0;

    virtual Vector SelectTrueVector(int level, const VectorView& vec_dof) const = 0;

    virtual Vector RestrictTrueVector(int level, const VectorView& vec_tdof) const = 0;

    virtual Vector DistributeTrueVector(int level, const VectorView& vec_tdof) const = 0;

    /// @name Get the dimension of domain and range of the operator at each level
    /// @{
    virtual int DomainSize(int level) const = 0;
    virtual int RangeSize(int level) const = 0;
    /// @}

    /// Get number of levels of the hierarchy
    int GetNumLevels() const { return num_levels_; }

    MPI_Comm GetComm() const { return comm_; }
protected:
    MPI_Comm comm_;
    int num_levels_;
};

enum SolveType { Newton, Picard };

class NonlinearSolver
{
public:
    NonlinearSolver(MPI_Comm comm, int size);

    // Compute the residual Rx = R(x).
    virtual void Mult(const VectorView& x, VectorView& Rx) = 0;

    // Solve R(sol) = rhs
    virtual void Solve(const VectorView& rhs, VectorView& sol) = 0;

    virtual Vector AssembleTrueVector(const VectorView& vec_dof) const = 0;

    ///@name Set solver parameters
    ///@{
    void SetPrintLevel(int print_level) { print_level_ = print_level; }
    void SetMaxIter(int max_num_iter) { max_num_iter_ = max_num_iter; }
    void SetRelTol(double rtol) { rtol_ = rtol; }
    void SetAbsTol(double atol) { atol_ = atol; }
    ///@}

    ///@name Get results of iterative solve
    ///@{
    int GetNumIterations() const { return iter_; }
    double GetTiming() const { return timing_; }
    bool IsConverged() const { return converged_; }
    ///@}

    int Size() const { return size_; }
protected:
    double ResidualNorm(const VectorView &sol, const VectorView &rhs);

    // default nonlinear solver options
    int print_level_ = 0;
    int max_num_iter_ = 50;
    double rtol_ = 1e-6;
    double atol_ = 1e-8;

    int iter_;
    double timing_;
    bool converged_;

    MPI_Comm comm_;
    int myid_;
    int size_;

    Vector residual_;
};

enum Cycle { V_CYCLE, FMG };

/**
   @brief Nonlinear multigrid using full approximation scheme and nonlinear relaxation.
*/
class NonlinearMG :public NonlinearSolver
{
public:
    // the time dependent operators gets updated during solving
    NonlinearMG(Hierarchy& hierarchy, Cycle cycle = Cycle::FMG);
    NonlinearMG() = delete;

    ~NonlinearMG() = default;

    /**
       Solve a nonlinear problem using FAS
       The BlockVectors here are in "dof" numbering, NOT "truedof" numbering.
    */
    virtual void Solve(const VectorView& rhs, VectorView& sol);
    virtual void Mult(const VectorView& x, VectorView& Rx);
    virtual Vector AssembleTrueVector(const VectorView& vec_dof) const;

protected:
    void FAS_FMG();
    void FAS_Cycle(int level);

    Hierarchy& hierarchy_;
    Cycle cycle_;
    int num_levels_;
    std::vector<Vector> rhs_;
    std::vector<Vector> sol_;
    mutable std::vector<Vector> help_;
};

} // namespace smoothg

#endif /* __NONLINEARMG_HPP__ */

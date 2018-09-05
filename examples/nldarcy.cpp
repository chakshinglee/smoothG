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
   @file nldarcy.cpp
   @brief nonlinear Darcy's problem.
*/

#include <fstream>
#include <sstream>
#include <mpi.h>

#include "smoothG.hpp"
#include "fv.hpp"

using namespace smoothg;

using linalgcpp::ReadText;
using linalgcpp::ReadCSR;
using std::unique_ptr;

/**
   @brief Nonlinear elliptic problem

   Given \f$f \in L^2(\Omega)\f$, \f$k(p)\f$ a differentiable function of p,
   find \f$p\f$ such that \f$-div(k_0k(p)\nabla p) = f\f$.
*/

class SingleLevelSolver : public NonlinearSolver
{
public:
    /**
       @todo take Kappa(p) as input
    */
    SingleLevelSolver(GraphUpscale& up, int level, SolveType solve_type);

    // Solve A(sol) = rhs
    virtual void Solve(const VectorView& rhs, VectorView& sol);

    // Compute Ax = A(x).
    virtual void Mult(const VectorView& x, VectorView& Ax);

    virtual Vector AssembleTrueVector(const VectorView& vec_dof) const;

private:
    void PicardSolve(const BlockVector& rhs, BlockVector& x);
    void PicardStep(const BlockVector& rhs, BlockVector& x);
    void NewtonSolve(const BlockVector& rhs, BlockVector& x);

    int level_;
    GraphUpscale& up_;
    SolveType solve_type_;

    std::vector<int> offsets_;
    Vector p_;         // coefficient vector in piecewise 1 basis
    std::vector<double> kp_;        // kp_ = Kappa(p)
};

// nonlinear elliptic hierarchy
class NonlinearEllipticHierarchy : public Hierarchy
{
public:
    NonlinearEllipticHierarchy(GraphUpscale& up, SolveType solve_type);
    NonlinearEllipticHierarchy() = delete;

    virtual void Mult(int level, const VectorView& x, VectorView& Ax);
    virtual void Solve(int level, const VectorView& rhs, VectorView& sol);
    virtual void Restrict(int level, const VectorView& fine, VectorView& coarse) const;
    virtual void Interpolate(int level, const VectorView& coarse, VectorView& fine) const;
    virtual void Project(int level, const VectorView& fine, VectorView& coarse) const;
    virtual void Smoothing(int level, const VectorView& in, VectorView& out) const;
    virtual Vector AssembleTrueVector(int level, const VectorView& vec_dof) const;
    virtual Vector SelectTrueVector(int level, const VectorView& vec_dof) const;
    virtual Vector RestrictTrueVector(int level, const VectorView& vec_tdof) const;
    virtual Vector DistributeTrueVector(int level, const VectorView& vec_tdof) const;
    virtual int DomainSize(int level) const { return solvers_[level]->Size(); }
    virtual int RangeSize(int level) const { return solvers_[level]->Size(); }
private:
    GraphUpscale& up_;
    std::vector<unique_ptr<SingleLevelSolver> > solvers_;
    std::vector<std::vector<int> > offsets_;
    Vector PuTPu_diag_;
};

void Kappa(const VectorView &p, std::vector<double>& kp);

int main(int argc, char* argv[])
{
    // Initialize MPI
    MpiSession mpi_info(argc, argv);
    MPI_Comm comm = mpi_info.comm_;
    int myid = mpi_info.myid_;

    // program options from command line
    int dim = 2;
    int max_evects = 4;
    double spect_tol = 1.0;
    bool hybridization = false;
    int num_levels = 2;
    double coarsening_factor = 10.0;

    int slice = 0;
    int num_sr = 3;
    int num_pr = 0;
    double correlation_length = 0.1;
    bool do_visualization = false;

    linalgcpp::ArgParser arg_parser(argc, argv);
    arg_parser.Parse(max_evects, "-m", "Maximum eigenvectors per aggregate.");
    arg_parser.Parse(spect_tol, "-t", "Spectral tolerance for eigenvalue problem.");
    arg_parser.Parse(hybridization, "-hb", "Enable hybridization.");
    arg_parser.Parse(num_levels, "-nl", "Number of levels.");
    arg_parser.Parse(coarsening_factor, "-cf", "Coarsening factor");
    arg_parser.Parse(dim, "-d", "Dimension of the physical space");
    arg_parser.Parse(slice, "-s", "z-direction slice number of SPE10 data set");
    arg_parser.Parse(num_sr, "-nsr", "Number of serial refinement");
    arg_parser.Parse(num_pr, "-npr", "Number of parallel refinement");
    arg_parser.Parse(correlation_length, "-cl", "Correlation length");
    arg_parser.Parse(do_visualization, "-vis", "Visualize the solution or not");
//num_levels = num_pr + 2;
    if (!arg_parser.IsGood())
    {
        ParPrint(myid, arg_parser.ShowHelp());
        ParPrint(myid, arg_parser.ShowErrors());
        return EXIT_FAILURE;
    }
    ParPrint(myid, arg_parser.ShowOptions());

    std::vector<int> ess_v_attr(dim == 2 ? 4: 3, 1);
    LognormalProblem fv_problem(dim, num_sr, num_pr, correlation_length, ess_v_attr);
//    SPE10Problem fv_problem("spe_perm.dat", dim, 5, slice, ess_v_attr, 15, 0);
    Graph graph = fv_problem.GetFVGraph(coarsening_factor*coarsening_factor, false);

    // Construct graph hierarchy
    GraphUpscale upscale(graph, fv_problem.GetLocalWeight(),
                         {spect_tol, max_evects, hybridization, num_levels,
                         coarsening_factor, fv_problem.GetEssentialVertDofs()});
    graph = Graph();

    upscale.PrintInfo();
    upscale.ShowSetupTime();

    BlockVector rhs(upscale.GetBlockVector(0));
    rhs.GetBlock(0) = 0.0;
    rhs.GetBlock(1) = fv_problem.GetVertexRHS();

    Timer timer(Timer::Start::True);

    BlockVector sol_picard(upscale.GetBlockVector(0));
    sol_picard = 0.0;

    SingleLevelSolver sls(upscale, 0, Picard);
    sls.SetPrintLevel(1);
//    sls.Solve(rhs, sol_picard);

    upscale.Solve(rhs, sol_picard);

return 0;
    timer.Click();

    if (myid == 0)
    {
        std::cout << "Picard iteration took " << sls.GetNumIterations()
                  << " iterations in " << timer.TotalTime() << " seconds.\n";
    }

    timer.Click();

    BlockVector sol_nlmg(upscale.GetBlockVector(0));
    sol_nlmg = 0.0;
    NonlinearEllipticHierarchy hierarchy(upscale, Picard);
    NonlinearMG nlmg(hierarchy, V_CYCLE);
    nlmg.SetPrintLevel(1);
    nlmg.Solve(rhs, sol_nlmg);

    timer.Click();

    if (myid == 0)
    {
        std::cout << "Nonlinear MG took " << nlmg.GetNumIterations()
                  << " iterations in " << timer[2] << " seconds.\n";
    }

    double p_err = CompareError(comm, sol_picard.GetBlock(1), sol_nlmg.GetBlock(1));
    if (myid == 0)
    {
        std::cout << "Relative errors: " << p_err << "\n";
    }

    if (do_visualization)
    {
        mfem::socketstream sout;
        mfem::Vector vis_help(sol_picard.GetBlock(1).begin(), rhs.GetBlock(1).size());
        fv_problem.VisSetup(sout, vis_help, 0.0, 0.0, "");
    }

    return EXIT_SUCCESS;
}

SingleLevelSolver::SingleLevelSolver(GraphUpscale& up, int level, SolveType solve_type)
    : NonlinearSolver(up.GetComm(), up.GetMatrix(level).Rows()),
      level_(level), up_(up), solve_type_(solve_type)
{
    offsets_ = up_.BlockOffsets(level_);
    p_.SetSize(up_.GetMatrix(level).GetElemDof().Rows());
    kp_.resize(p_.size());
}

void SingleLevelSolver::Mult(const VectorView& x, VectorView& Ax)
{
    assert(size_ == Ax.size());
    assert(size_ == x.size());
    BlockVector block_x(x, offsets_);

    // Update operator
    up_.Project_PW_One(level_, block_x.GetBlock(1), p_);
    Kappa(p_, kp_);

//    up_.MakeSolver(level_, kp_);
//    up_.Mult(level_, x, Ax);

    up_.Mult(level_, kp_, x, Ax);
}

Vector SingleLevelSolver::AssembleTrueVector(const VectorView& vec_dof) const
{
    return up_.GetMatrix(level_).AssembleTrueVector(vec_dof);
}

void SingleLevelSolver::Solve(const VectorView &rhs, VectorView &sol)
{
    assert(solve_type_ == Newton || solve_type_ == Picard);

    BlockVector block_sol(sol, offsets_);
    BlockVector block_rhs(rhs, offsets_);
    if (solve_type_ == Picard)
    {
        PicardSolve(block_rhs, block_sol);
    }
    else
    {
        NewtonSolve(block_rhs, block_sol);
    }
    sol = block_sol;
}

void SingleLevelSolver::PicardSolve(const BlockVector& rhs, BlockVector& x)
{
    if (max_num_iter_ == 1)
    {
        PicardStep(rhs, x);
    }
    else
    {
        double norm = parlinalgcpp::ParL2Norm(comm_, rhs);
        converged_ = false;
        for (iter_ = 0; iter_ < max_num_iter_; iter_++)
        {
            double resid = ResidualNorm(x, rhs);
            double rel_resid = resid / norm;

            if (myid_ == 0 && print_level_ > 0)
            {
                std::cout << "Picard iter " << iter_ << ":  rel resid = "
                          << rel_resid << "  abs resid = " << resid << "\n";
            }

            if (resid < atol_ || rel_resid < rtol_)
            {
                converged_ = true;
                break;
            }

            PicardStep(rhs, x);
        }

        if (myid_ == 0 && !converged_ && print_level_ >= 0)
        {
            std::cout << "Warning: Picard iteration reached maximum number of iterations!\n";
        }
    }
}

void SingleLevelSolver::PicardStep(const BlockVector& rhs, BlockVector& x)
{
    up_.Project_PW_One(level_, x.GetBlock(1), p_);
//if (level_>0 && myid_ ==0) {Vector p2(p_.begin(), 5); p2.Print();}
    Kappa(p_, kp_);
    up_.MakeSolver(level_, kp_);

    if (level_ ==0)//< up_.NumLevels() - 1)
        up_.SetMaxIter(max_num_iter_ * 10);
    else
        up_.SetMaxIter(max_num_iter_ * 20);

    up_.SolveLevel(level_, rhs, x);
    up_.ShowSolveInfo(level_);
}

void SingleLevelSolver::NewtonSolve(const BlockVector& rhs, BlockVector& x)
{
    // TBD...
}

NonlinearEllipticHierarchy::NonlinearEllipticHierarchy(
        GraphUpscale& up, SolveType solve_type)
    : Hierarchy(up.GetComm(), up.NumLevels()), up_(up),
      solvers_(num_levels_), offsets_(num_levels_)
{
    for (int i = 0; i < up_.NumLevels(); ++i)
    {
        solvers_[i] = make_unique<SingleLevelSolver>(up_, i, solve_type);
        solvers_[i]->SetPrintLevel(-1);
        solvers_[i]->SetMaxIter(1);
        offsets_[i] = up_.BlockOffsets(i);
    }
}

void NonlinearEllipticHierarchy::Mult(
        int level, const VectorView& x, VectorView& Ax)
{
    solvers_[level]->Mult(x, Ax);
}

void NonlinearEllipticHierarchy::Solve(
        int level, const VectorView& rhs, VectorView& sol)
{
    solvers_[level]->Solve(rhs, sol);
}

void NonlinearEllipticHierarchy::Restrict(
        int level, const VectorView& fine, VectorView& coarse) const
{
    BlockVector block_fine(fine, offsets_[level]);
    coarse = up_.GetCoarsener(level).Restrict(block_fine);
}

void NonlinearEllipticHierarchy::Interpolate(
        int level, const VectorView& coarse, VectorView& fine) const
{
    BlockVector block_coarse(coarse, offsets_[level]);
    fine = up_.GetCoarsener(level - 1).Interpolate(block_coarse);
}

void NonlinearEllipticHierarchy::Project(
        int level, const VectorView& fine, VectorView& coarse) const
{
    BlockVector block_fine(fine, offsets_[level]);
    coarse = up_.GetCoarsener(level).Project(block_fine);
}

void NonlinearEllipticHierarchy::Smoothing(
        int level, const VectorView& in, VectorView& out) const
{
    solvers_[level]->Solve(in, out);
}

Vector NonlinearEllipticHierarchy::AssembleTrueVector(
        int level, const VectorView& vec_dof) const
{
    return solvers_[level]->AssembleTrueVector(vec_dof);
}

Vector NonlinearEllipticHierarchy::SelectTrueVector(
        int level, const VectorView& vec_dof) const
{
    return up_.GetMatrix(level).SelectTrueVector(vec_dof);
}

Vector NonlinearEllipticHierarchy::RestrictTrueVector(
        int level, const VectorView& vec_tdof) const
{
    return up_.GetMatrix(level).RestrictTrueVector(vec_tdof);
}

Vector NonlinearEllipticHierarchy::DistributeTrueVector(
        int level, const VectorView& vec_tdof) const
{
    return up_.GetMatrix(level).DistributeTrueVector(vec_tdof);
}

// Kappa(p) = exp(- \alpha p)
void Kappa(const VectorView& p, std::vector<double>& kp)
{
    assert(kp.size() == p.size());
    for (int i = 0; i < p.size(); i++)
    {
        kp[i] = std::exp(3. * (p[i]));
        assert(kp[i] > 0.0);
    }
}

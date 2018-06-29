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
   @file singlephase.cpp
   @brief This is an example for upscaling a graph Laplacian coming from a finite
   volume discretization of a single phase flow and transport model in parallel.

   A simple way to run the example:

   mpirun -n 4 ./singlephase
*/

#include <fstream>
#include <sstream>
#include <mpi.h>

#include "mfem.hpp"
//#include "spe10.hpp"
#include "well.hpp"

#include "../src/picojson.h"
#include "../src/smoothG.hpp"

using namespace smoothg;

enum Level { Fine = 0, Coarse };
enum CoarseAdv { Upwind, RAP, FastRAP };
enum SolveType { Newton, Picard };

/**
   @brief Two phase flow and transport time dependent operator

   Given S^n from previous time step, the residual operator is
   R(x) = R(v, p, S) = (M(S)v - D^Tp - h, Dv - f, M_s(S - S^n)/dt - Adv(v)F(S) - g)
*/

class TwoPhaseTDO : public mfem::TimeDependentOperator
{
public:
    TwoPhaseTDO(int size, SolveType solve_type);

    // Solve for k such that R(x^n + dt*k) = R(x^{n+1}) = 0.
    virtual void ImplicitSolve(const double dt, const mfem::Vector& x, mfem::Vector& k);

    // update time step dt, previous state S^n and source term (h, f, g + M_s S^n / dt)
    virtual void Update(double dt, const mfem::Vector& prev_state) = 0;

    // Solve for k such that R(x^n + dt*k) = rhs, need to call Update(dt, x^n) first
    virtual void Solve(const mfem::Vector& rhs, mfem::Vector& sol) = 0;

    virtual const mfem::Array<int>& GetOffsets() const = 0;

    ///@name Set solver parameters
    ///@{
    virtual void SetPrintLevel(int print_level) { print_level_ = print_level; }
    virtual void SetMaxIter(int max_num_iter) { max_num_iter_ = max_num_iter; }
    virtual void SetRelTol(double rtol) { rtol_ = rtol; }
    virtual void SetAbsTol(double atol) { atol_ = atol; }
    ///@}

    ///@name Get results of iterative solve
    ///@{
    virtual int GetNumIterations() const { return iter_; }
    virtual double GetTiming() const { return timing_; }
    virtual bool IsConverged() const { return converged_; }
    ///@}
protected:
    // default solver options
    int print_level_ = 0;
    int max_num_iter_ = 500;
    double rtol_ = 1e-6;
    double atol_ = 1e-8;

    int iter_;
    double timing_;
    bool converged_;

    SolveType solve_type_;
    mfem::Vector rhs_;     // always = 0, needed for ImplicitSolve
};

class TwoPhaseOp : public TwoPhaseTDO
{
public:
    TwoPhaseOp(const SPE10Problem& spe10problem, FiniteVolumeMLMC& up,
               Level level, SolveType solve_type);

    virtual ~TwoPhaseOp() = default;

    virtual void Update(double dt, const mfem::Vector& prev_state);
    virtual void Solve(const mfem::Vector& rhs, mfem::Vector& sol);
    virtual const mfem::Array<int>& GetOffsets() const { return offsets_; }

    // Compute the residual of the two-phase flow system Rx = R(x) = R(v, p, S).
    void Mult(const mfem::Vector& x, mfem::Vector& Rx);


protected:
    // Setup needed for (advection) transport operator Adv
    void SetupTransportOp(const mfem::SparseMatrix& elem_facet,
                          const mfem::HypreParMatrix& facet_truefacet);
    void MakeTransportOp(const mfem::Vector& normal_flux);
    void MakeTransportSolver(const mfem::Vector& normal_flux);

    // Compute d(Rv(v, S)) / dv,
    void dRvdv(const mfem::Vector& normal_flux, const mfem::Vector& FS);

    // Solve R(x) = R(v, p, S) = rhs, input x is initial guess, output x is solution
    void PicardSolve(const mfem::BlockVector &rhs, mfem::BlockVector& x);
    void PicardStep(const mfem::BlockVector &rhs, mfem::BlockVector& x);
    void NewtonSolve(const mfem::BlockVector &rhs, mfem::BlockVector& x);

    double ResidualNorm(const mfem::Vector& sol, const mfem::Vector& rhs);

    MPI_Comm comm_;
    int myid_;

    Level level_;
    FiniteVolumeMLMC& up_;
    mfem::SparseMatrix f_tf_diag_; // diag part of facet_truefacet
    unique_ptr<mfem::HypreParMatrix> facet_truefacet_elem_;
    mfem::SparseMatrix f_tf_e_diag_;
    mfem::SparseMatrix f_tf_e_offd_;
    mfem::Vector M_s_;   // mass matrix for saturation

    ///@name components for transport_op_
    ///@{
    HYPRE_Int* tran_op_colmap_;
    mfem::Array<int> tran_op_starts_;
    unique_ptr<mfem::HypreParMatrix> transport_op_;
    ///@}

    ///@name components for transport_op_
    ///@{
    HYPRE_Int num_tf_;
    HYPRE_Int* tf_starts_;
    unique_ptr<mfem::HypreParMatrix> dRvdv_;
    ///@}

    mfem::Array<int> offsets_;
    unique_ptr<mfem::Operator> flow_solver_;
    unique_ptr<mfem::HypreBoomerAMG> transport_prec_;
    mfem::CGSolver transport_solver_;
    unique_ptr<mfem::BlockOperator> solver_;

    mutable mfem::Vector tot_mob_;            // total mobility
    mutable mfem::Vector frac_flow_;          // factional flow
    unique_ptr<mfem::BlockVector> source_0_;  // source term at time 0 (h, f, g)
    unique_ptr<mfem::BlockVector> source_t_;  // current source (h, f, g + M_s S^n/dt)
    mfem::BlockVector prev_state_;            // reference to states at previous step
    mfem::Vector residual_;

    double dt_;                               // time step size
    friend class TPH;
};

// two phase hierarchy
class TPH : public Hierarchy
{
public:
    TPH(const SPE10Problem& spe10problem, FiniteVolumeMLMC& up, SolveType solve_type);
    TPH() = delete;

    // Compute resid = R(in)
    virtual void Mult(int level, const mfem::Vector& x, mfem::Vector& Rx) const;
    virtual void Solve(int level, const mfem::Vector& rhs, mfem::Vector& sol);
    virtual void Restrict(int level, const mfem::Vector& fine, mfem::Vector& coarse) const;
    virtual void Interpolate(int level, const mfem::Vector& coarse, mfem::Vector& fine) const;
    virtual void Project(int level, const mfem::Vector& fine, mfem::Vector& coarse) const;
    virtual void Smoothing(int level, const mfem::Vector& in, mfem::Vector& out) const;
    virtual const int DomainSize(int level) const { return ops_[level]->Width(); }
    virtual const int RangeSize(int level) const { return ops_[level]->Height(); }

    void Update(double dt, const mfem::Vector& prev_state);

    const mfem::Array<int>& GetOffsets() const { return ops_[0]->GetOffsets(); }
private:
    FiniteVolumeMLMC& up_;
    std::vector<unique_ptr<TwoPhaseOp>> ops_;
    mutable std::vector<mfem::BlockVector> helper_;
    std::vector<mfem::Array<int>> flow_offsets_;
    mfem::Vector prev_state_coarse_;
    mfem::Vector PuTPu_diag;
};

// two-phase flow and transport solver using nonlinear multigrid
class NMG_TP : public TwoPhaseTDO
{
public:
    NMG_TP(const SPE10Problem& spe10problem, FiniteVolumeMLMC& up,
           Cycle cycle, SolveType solve_type);
    NMG_TP() = delete;

    virtual void Update(double dt, const mfem::Vector& prev_state)
    {
        tph_.Update(dt, prev_state);
    }

    virtual void Solve(const mfem::Vector& rhs, mfem::Vector& sol)
    {
        nl_mg_.Solve(rhs, sol);
    }

    virtual const mfem::Array<int>& GetOffsets() const { return tph_.GetOffsets(); }

    virtual void SetPrintLevel(int print_level) { nl_mg_.SetPrintLevel(print_level); }
    virtual void SetMaxIter(int max_num_iter) { nl_mg_.SetMaxIter(max_num_iter); }
    virtual void SetRelTol(double rtol) { nl_mg_.SetRelTol(rtol); }
    virtual void SetAbsTol(double atol) { nl_mg_.SetAbsTol(atol); }

    virtual int GetNumIterations() const { return nl_mg_.GetNumIterations(); }
    virtual bool IsConverged() const { return nl_mg_.IsConverged(); }
private:
    TPH tph_;
    NonlinearMG nl_mg_;
};

void TotalMobility(const mfem::Vector& S, mfem::Vector& LamS);
void FractionalFlow(const mfem::Vector& S, mfem::Vector& FS);
void TotalMobilityDerivative(const mfem::Vector& S, mfem::Vector& dLamS);
void FractionalFlowDerivative(const mfem::Vector& S, mfem::Vector& dFS);

mfem::Vector TwoPhaseFlow(const SPE10Problem& spe10problem, FiniteVolumeMLMC &up, double delta_t,
                          double total_time, int vis_step, Level level,
                          const std::string& caption, bool use_mg);

mfem::Array<int> well_vertices;
int main(int argc, char* argv[])
{
    int num_procs, myid;

    // 1. Initialize MPI
    mpi_session session(argc, argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    // program options from command line
    mfem::OptionsParser args(argc, argv);
    const char* permFile = "spe_perm.dat";
    args.AddOption(&permFile, "-p", "--perm",
                   "SPE10 permeability file data.");
    int nDimensions = 2;
    args.AddOption(&nDimensions, "-d", "--dim",
                   "Dimension of the physical space.");
    int slice = 0;
    args.AddOption(&slice, "-s", "--slice",
                   "Slice of SPE10 data to take for 2D run.");
    int max_evects = 4;
    args.AddOption(&max_evects, "-m", "--max-evects",
                   "Maximum eigenvectors per aggregate.");
    double spect_tol = 1.0;
    args.AddOption(&spect_tol, "-t", "--spect-tol",
                   "Spectral tolerance for eigenvalue problems.");
    bool metis_agglomeration = false;
    args.AddOption(&metis_agglomeration, "-ma", "--metis-agglomeration",
                   "-nm", "--no-metis-agglomeration",
                   "Use Metis as the partitioner (instead of geometric).");
    int spe10_scale = 5;
    args.AddOption(&spe10_scale, "-sc", "--spe10-scale",
                   "Scale of problem, 1=small, 5=full SPE10.");
    bool hybridization = false;
    args.AddOption(&hybridization, "-hb", "--hybridization", "-no-hb",
                   "--no-hybridization", "Enable hybridization.");
    bool dual_target = true;
    args.AddOption(&dual_target, "-du", "--dual-target", "-no-du",
                   "--no-dual-target", "Use dual graph Laplacian in trace generation.");
    bool scaled_dual = true;
    args.AddOption(&scaled_dual, "-sd", "--scaled-dual", "-no-sd",
                   "--no-scaled-dual", "Scale dual graph Laplacian by (inverse) edge weight.");
    bool energy_dual = true;
    args.AddOption(&energy_dual, "-ed", "--energy-dual", "-no-ed",
                   "--no-energy-dual", "Use energy matrix in trace generation.");
    double delta_t = 1.0;
    args.AddOption(&delta_t, "-dt", "--delta-t", "Time step.");
    double total_time = 1000.0;
    args.AddOption(&total_time, "-time", "--total-time", "Total time to step.");
    int vis_step = 0;
    args.AddOption(&vis_step, "-vs", "--vis-step", "Step size for visualization.");
    int write_step = 0;
    args.AddOption(&write_step, "-ws", "--write-step", "Step size for writing data to file.");
    int well_height = 1;
    args.AddOption(&well_height, "-wh", "--well-height", "Well Height.");
    double inject_rate = 0.3;
    args.AddOption(&inject_rate, "-ir", "--inject-rate", "Injector rate.");
    double bottom_hole_pressure = 175.0;
    args.AddOption(&bottom_hole_pressure, "-bhp", "--bottom-hole-pressure",
                   "Bottom Hole Pressure.");
    double well_shift = 1.0;
    args.AddOption(&well_shift, "-wsh", "--well-shift", "Shift well from corners");
    int nz = 15;
    args.AddOption(&nz, "-nz", "--num-z", "Num of slices in z direction for 3d run.");
    int coarsening_factor = 10;
    args.AddOption(&coarsening_factor, "-cf", "--coarsen-factor", "Coarsening factor");
    args.Parse();
    if (!args.Good())
    {
        if (myid == 0)
        {
            args.PrintUsage(std::cout);
        }
        MPI_Finalize();
        return 1;
    }
    if (myid == 0)
    {
        args.PrintOptions(std::cout);
    }

    const int nbdr = 6;
    mfem::Array<int> ess_attr(nbdr);
    ess_attr = 1;

    // Setting up finite volume discretization problem
    double scaled_inject = SPE10Problem::CellVolume() * inject_rate;
    printf("Inject: %.8f Scaled Inject: %.8f\n", inject_rate, scaled_inject);
    SPE10Problem spe10problem(permFile, nDimensions, spe10_scale, slice, ess_attr,
                              nz, well_height, scaled_inject, bottom_hole_pressure, well_shift);

    auto& vertex_edge = spe10problem.GetVertexEdge();
    auto& edge_d_td = spe10problem.GetEdgeToTrueEdge();
    auto& weight = spe10problem.GetWeight();
    auto& local_weight = spe10problem.GetLocalWeight();
    auto& edge_bdr_att = spe10problem.GetEdgeBoundaryAttributeTable();
    auto& well_list = spe10problem.GetWells();

    int num_producer = 0;
    for (auto& well : well_list)
    {
        if (well.GetType() == WellType::Producer)
            num_producer++;
    }

    // add one boundary attribute for edges connecting production wells to reservoir
    int total_num_producer;
    MPI_Allreduce(&num_producer, &total_num_producer, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (total_num_producer > 0)
    {
        ess_attr.Append(0);
    }

    int num_well_cells = 0;
    for (auto& well : well_list)
    {
        num_well_cells += well.GetNumberOfCells();
    }

    {
        auto edge_vertex = smoothg::Transpose(vertex_edge);

        mfem::Array<int> vertices;
        for (int i = edge_vertex.Height() - num_well_cells; i < edge_vertex.Height(); ++i)
        {
            GetTableRow(edge_vertex, i, vertices);

            well_vertices.Append(vertices);
        }
        well_vertices.Print();
    }

    std::vector<int> wells(num_procs, 0);
    int my_wells = well_list.size();
    MPI_Gather(&my_wells, 1, MPI::INT, wells.data(), 1, MPI::INT, 0, comm);

    if (myid == 0)
    {
        for (int i = 0; i < num_procs; ++i)
        {
            std::cout << "Proc: " << i << " Wells: " << wells[i] << "\n";
        }
    }

    mfem::Array<int> partition;
    int nparts = std::max(vertex_edge.Height() / coarsening_factor, 1);
    bool adaptive_part = false;
    bool use_edge_weight = (nDimensions == 3) && (nz > 1);
    PartitionVerticesByMetis(vertex_edge, weight, well_vertices, nparts,
                             partition, adaptive_part, use_edge_weight);

    mfem::Array<int> geo_coarsening_factor(3);
    geo_coarsening_factor[0] = 5;
    geo_coarsening_factor[1] = 5;
    geo_coarsening_factor[2] = nDimensions == 3 ? 1 : nz;
//    spe10problem.CartPart(partition, nz, geo_coarsening_factor, well_vertices);

    // Create Upscaler and Solve
    FiniteVolumeMLMC fvupscale(comm, vertex_edge, local_weight, partition, edge_d_td,
                               edge_bdr_att, ess_attr, spect_tol, max_evects,
                               dual_target, scaled_dual, energy_dual, hybridization, false);
    fvupscale.PrintInfo();
    fvupscale.ShowSetupTime();
    fvupscale.MakeFineSolver();

    const bool use_mg = true;

    auto S_mg = TwoPhaseFlow(
                spe10problem, fvupscale, delta_t, total_time, vis_step,
                Fine, "saturation solved by nonlinear MG", use_mg);

    // Fine scale transport based on fine flux
    auto S_fine = TwoPhaseFlow(
                spe10problem, fvupscale, delta_t, total_time, vis_step,
                Fine, "saturation solved by Picard iteration", !use_mg);

//    auto S_coarse = TwoPhaseFlow(
//                spe10problem, fvupscale, delta_t, total_time, vis_step,
//                Coarse, "saturation based on fastRAP", !use_mg);

//    double sat_err = CompareError(comm, S_coarse, S_fine);
    double sat_err2 = CompareError(comm, S_mg, S_fine);
    if (myid == 0)
    {
//        std::cout << "Saturation errors: " << sat_err << "\n";
        std::cout << "Saturation errors: " << sat_err2 << "\n";
    }

    return EXIT_SUCCESS;
}

mfem::socketstream sout;
int option = 0;
bool setup = true;
mfem::Vector TwoPhaseFlow(const SPE10Problem& spe10problem, FiniteVolumeMLMC& up,
                          double delta_t, double total_time, int vis_step, Level level,
                          const std::string& caption, bool use_mg)
{
    option++;

    int myid;
    MPI_Comm_rank(up.GetComm(), &myid);

    mfem::StopWatch chrono;

    chrono.Clear();
    MPI_Barrier(up.GetComm());
    chrono.Start();

    double time = 0.0;

    unique_ptr<TwoPhaseTDO> time_dependent_op;
    if (use_mg)
    {
        assert(level == Fine);
        time_dependent_op = make_unique<NMG_TP>(spe10problem, up, FMG, Picard);
    }
    else
    {
        time_dependent_op = make_unique<TwoPhaseOp>(spe10problem, up, level, Picard);
    }
//    time_dependent_op->SetPrintLevel(1);
    time_dependent_op->SetMaxIter(50);
    time_dependent_op->SetRelTol(1e-6);
    time_dependent_op->SetAbsTol(1e-8);
    time_dependent_op->SetTime(time);

    mfem::BackwardEulerSolver ode_solver;
    ode_solver.Init(*time_dependent_op);

    // state = (flux, pressure, saturation)
    mfem::BlockVector state(time_dependent_op->GetOffsets());
    state = 0.0;
    mfem::BlockVector previous_state(state);

    // visualization for S
    mfem::Vector S_vis;
    if (level == Fine)
    {
        S_vis.SetDataAndSize(state.GetBlock(1).GetData(), state.BlockSize(1));
    }
    else
    {
        S_vis.SetSize(spe10problem.GetVertexRHS().Size());
    }

    if (vis_step && setup)
    {
        if (level == Coarse)
        {
            up.Interpolate(state.GetBlock(1), S_vis);
        }
        spe10problem.VisSetup(sout, S_vis, 0.0, 1.0, caption);
//        setup = false;
    }

    bool done = false;
    double dt_real = std::min(delta_t, total_time - time);
    double nonlinear_iter = 0.0;
    int ti = 0;
    for ( ; !done; )
    {
        dt_real = std::min(std::min(delta_t, total_time - time), dt_real * 2.0);
        previous_state = state;
        ode_solver.Step(state, time, dt_real);
        nonlinear_iter += time_dependent_op->GetNumIterations();
        while (!time_dependent_op->IsConverged()) // TODO: add smallest step size condition
        {
            time -= dt_real;
            dt_real /= 2.0;
            if (myid == 0)
            {
                std::cout << "Restart nonlinear solve with time step size " << dt_real << "\n";
            }
            state = previous_state;
            ode_solver.Step(state, time, dt_real);
            nonlinear_iter += time_dependent_op->GetNumIterations();
        }
        ti++;

        done = (time >= total_time - 1e-8 * delta_t);

        if (myid == 0)
        {
            std::cout << "time step: " << ti << ", time: " << time << "\r";//std::endl;
        }
        if (vis_step && (done || ti % vis_step == 0))
        {
            if (level == Coarse)
            {
                up.Interpolate(state.GetBlock(1), S_vis);
            }
            spe10problem.VisUpdate(sout, S_vis);
        }
    }
    MPI_Barrier(up.GetComm());
    if (myid == 0)
    {
        std::cout << "Time stepping done in " << chrono.RealTime() << "s.\n"
                  << "Total # nonlinear iter = " << nonlinear_iter << ".\n";
        std::cout << "Average # nonlinear iter = " << nonlinear_iter/ti << ".\n";
    }

    mfem::Vector out;
    if (level == Coarse)
    {
        up.Interpolate(state.GetBlock(1), S_vis);
        out = S_vis;
    }
    else
    {
        out = state.GetBlock(1);
    }

    return out;
}

TwoPhaseTDO::TwoPhaseTDO(int size, SolveType solve_type)
    : mfem::TimeDependentOperator(size, 0.0, IMPLICIT), solve_type_(solve_type), rhs_(size)
{
    rhs_ = 0.0;
}

void TwoPhaseTDO::ImplicitSolve(const double dt, const mfem::Vector& x, mfem::Vector& k)
{
    Update(dt, x);
    k = x;
    Solve(rhs_, k);

    if (solve_type_ == Picard)
    {
        k -= x;
        k /= dt;
    }
}

TwoPhaseOp::TwoPhaseOp(const SPE10Problem &spe10problem, FiniteVolumeMLMC &up,
                       Level level, SolveType solve_type)
    : TwoPhaseTDO(up.GetNumTotalDofs(level)+up.GetNumVertexDofs(level), solve_type),
      comm_(up.GetComm()), level_(level), up_(up), transport_solver_(comm_)
{
    MPI_Comm_rank(comm_, &myid_);
    offsets_.SetSize(3, 0);
    offsets_[1] = up.GetMatrix(level_).GetNumTotalDofs();
    offsets_[2] = offsets_[1] + up.GetMatrix(level_).GetNumVertexDofs();

    source_0_ = make_unique<mfem::BlockVector>(offsets_);

    mfem::SparseMatrix M_s = SparseIdentity(spe10problem.GetVertexRHS().Size());
    M_s *= spe10problem.CellVolume();
    if (level_ == Fine)
    {
        SetupTransportOp(spe10problem.GetVertexEdge(), spe10problem.GetEdgeToTrueEdge());

        source_0_->GetBlock(1) = spe10problem.GetVertexRHS();
        source_0_->GetBlock(0).SetVector(spe10problem.GetEdgeRHS(), 0);
        source_0_->GetBlock(0).SetVector(source_0_->GetBlock(1), up.GetNumEdgeDofs(0));

        M_s.GetDiag(M_s_);
        flow_solver_ = make_unique<UpscaleFineBlockSolve>(up_);
    }
    else
    {
//        SetupTransportOp(up_.GetAggFace(), up_.GetFaceTrueFace());
        SetupTransportOp(spe10problem.GetVertexEdge(), spe10problem.GetEdgeToTrueEdge());

        mfem::BlockVector tmp(up_.GetFineBlockVector());
        tmp.GetBlock(0) = spe10problem.GetEdgeRHS();
        tmp.GetBlock(1) = spe10problem.GetVertexRHS();
        mfem::BlockVector tmp_c = up_.Restrict(tmp);
        source_0_->GetBlock(0).SetVector(tmp_c, 0);
        source_0_->GetBlock(1) = tmp_c.GetBlock(1);

        unique_ptr<mfem::SparseMatrix> M_s_c(mfem::RAP(up.GetPu(), M_s, up.GetPu()));
        M_s_c->GetDiag(M_s_);
        flow_solver_ = make_unique<UpscaleCoarseBlockSolve>(up_);
    }
    source_0_->GetBlock(1) *= -1.0;  // g = -f
    source_t_ = make_unique<mfem::BlockVector>(*source_0_);

    residual_.SetSize(offsets_.Last());
    rhs_.SetSize(offsets_.Last());
    rhs_ = 0.0;

    up_.SetPrintLevel(-1);

//    MakeTransportOp(rhs_->GetBlock(0));
    solver_ = make_unique<mfem::BlockOperator>(offsets_);
    solver_->SetBlock(0, 0, flow_solver_.get());

    tot_mob_.SetSize(source_0_->BlockSize(1));
    frac_flow_.SetSize(source_0_->BlockSize(1));

    transport_solver_.SetPrintLevel(print_level_);
    transport_solver_.SetMaxIter(5000);
    transport_solver_.SetRelTol(rtol_);
    transport_solver_.SetAbsTol(atol_);
}

void TwoPhaseOp::Mult(const mfem::Vector& x, mfem::Vector &Rx)
{
    assert(offsets_.Last() == Rx.Size());
    assert(offsets_.Last() == x.Size());
    mfem::BlockVector block_x(x.GetData(), offsets_);
    mfem::BlockVector block_Rx(Rx.GetData(), offsets_);

    block_Rx.GetBlock(0) = 0.0;
//    if (level_ == Fine)
//    {
//        up_.MultFine(block_x.GetBlock(0), block_Rx.GetBlock(0));
//    }
//    else
//    {
//        up_.MultCoarse(block_x.GetBlock(0), block_Rx.GetBlock(0));
//    }
//    block_Rx.GetBlock(0) -= source_t_->GetBlock(0);

    MakeTransportOp(block_x.GetBlock(0));

    block_Rx.GetBlock(1) = block_x.GetBlock(1);
    RescaleVector(M_s_, block_Rx.GetBlock(1));
    block_Rx.GetBlock(1) /= dt_;
    block_Rx.GetBlock(1) -= source_t_->GetBlock(1);
    FractionalFlow(block_x.GetBlock(1), frac_flow_);
    transport_op_->Mult(1.0, frac_flow_, 1.0, block_Rx.GetBlock(1));
}

double TwoPhaseOp::ResidualNorm(const mfem::Vector& sol, const mfem::Vector& rhs)
{
    residual_ = 0.0;
    Mult(sol, residual_);
    residual_ -= rhs;
    return mfem::ParNormlp(residual_, 2, comm_);
}

void TwoPhaseOp::Solve(const mfem::Vector& rhs, mfem::Vector& sol)
{
    assert(solve_type_ == Newton || solve_type_ == Picard);

    mfem::BlockVector block_sol(sol.GetData(), offsets_);
    mfem::BlockVector block_rhs(rhs.GetData(), offsets_);
    if (solve_type_ == Picard)
    {
        PicardSolve(block_rhs, block_sol);
    }
    else
    {
        NewtonSolve(block_rhs, block_sol);
    }
}

void TwoPhaseOp::PicardSolve(const mfem::BlockVector& rhs, mfem::BlockVector& x)
{
    mfem::BlockVector adjusted_source(*source_t_);

    adjusted_source += rhs;
    if (max_num_iter_ == 1)
    {
        PicardStep(adjusted_source, x);
    }
    else
    {
        double norm = mfem::ParNormlp(adjusted_source, 2, comm_);
        converged_ = false;
        for (iter_ = 0; iter_ < max_num_iter_; iter_++)
        {
            PicardStep(adjusted_source, x);

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
                iter_++;
                break;
            }
        }

        if (myid_ == 0 && !converged_ && print_level_ >= 0)
        {
            std::cout << "Warning: Picard iteration reached maximum number of iterations!\n";
        }
    }
}

void TwoPhaseOp::PicardStep(const mfem::BlockVector& rhs, mfem::BlockVector& x)
{
    // solve pressure equation
    if (level_ == Coarse)
    {
        auto S_fine = up_.Interpolate(x.GetBlock(1));
        tot_mob_.SetSize(S_fine.Size());
        TotalMobility(S_fine, tot_mob_);
    }
    else
    {
        TotalMobility(x.GetBlock(1), tot_mob_);
    }
    up_.RescaleCoefficient(level_, tot_mob_);
    flow_solver_->Mult(rhs.GetBlock(0), x.GetBlock(0));

    // solve saturation equation
    MakeTransportOp(x.GetBlock(0));
    FractionalFlow(x.GetBlock(1), frac_flow_);
    x.GetBlock(1) = rhs.GetBlock(1);
    transport_op_->Mult(-1.0, frac_flow_, 1.0, x.GetBlock(1));
    InvRescaleVector(M_s_, x.GetBlock(1));
    x.GetBlock(1) *= dt_;
}

void TwoPhaseOp::NewtonSolve(const mfem::BlockVector& rhs, mfem::BlockVector& x)
{
    // TBD...
}

void TwoPhaseOp::SetupTransportOp(const mfem::SparseMatrix& elem_facet,
                                  const mfem::HypreParMatrix& facet_truefacet)
{
    GenerateOffsets(comm_, elem_facet.Height(), tran_op_starts_);

    using ParMatPtr = std::unique_ptr<mfem::HypreParMatrix>;
    ParMatPtr elem_truefacet(facet_truefacet.LeftDiagMult(elem_facet, tran_op_starts_));
    ParMatPtr truefacet_elem(elem_truefacet->Transpose());
    facet_truefacet_elem_.reset(mfem::ParMult(&facet_truefacet, truefacet_elem.get()));
    facet_truefacet_elem_->GetDiag(f_tf_e_diag_);
    facet_truefacet_elem_->GetOffd(f_tf_e_offd_, tran_op_colmap_);

    facet_truefacet.GetDiag(f_tf_diag_);

    num_tf_ = facet_truefacet.N();
    tf_starts_ = facet_truefacet.GetColStarts();

    if (level_ == Coarse)
    {
        GenerateOffsets(comm_, up_.GetPu().Width(), tran_op_starts_);
    }
}

void TwoPhaseOp::MakeTransportOp(const mfem::Vector& normal_flux)
{
    mfem::Array<int> facedofs;
    mfem::Vector normal_flux_loc;
    mfem::Vector normal_flux_fine;

    if (level_ == Coarse)
    {
        mfem::Vector normal_flux_ref(normal_flux.GetData(), up_.GetPsigma().Width());
        normal_flux_fine.SetSize(up_.GetPsigma().Height());
        up_.GetPsigma().Mult(normal_flux_ref, normal_flux_fine);
    }

    mfem::SparseMatrix diag(f_tf_e_diag_.Width(), f_tf_e_diag_.Width());
    mfem::SparseMatrix offd(f_tf_e_diag_.Width(), f_tf_e_offd_.Width());
    for (int face = 0; face < f_tf_diag_.Height(); face++)
    {
        double normal_flux_i_0 = 0.0;
        double normal_flux_i_1 = 0.0;

        if (level_ == Fine)
        {
            normal_flux_i_0 = (fabs(normal_flux(face)) - normal_flux(face)) / 2;
            normal_flux_i_1 = (fabs(normal_flux(face)) + normal_flux(face)) / 2;
        }
        else
        {
            normal_flux_i_0 = (fabs(normal_flux_fine(face)) - normal_flux_fine(face)) / 2;
            normal_flux_i_1 = (fabs(normal_flux_fine(face)) + normal_flux_fine(face)) / 2;

//            const auto& normal_flip = up_.GetCoarseToFineNormalFlip()[face];
//            GetTableRow(up_.GetFaceToFaceDof(), face, facedofs);
//            normal_flux.GetSubVector(facedofs, normal_flux_loc);
//            normal_flux_loc *= -1.0; // P matrix stores negative traces
//            normal_flux_fine.SetSize(normal_flip.size());
//            up_.GetTraces()[face].Mult(normal_flux_loc, normal_flux_fine);
//            for (int i = 0; i < normal_flux_fine.Size(); i++)
//            {
//                double fine_flux_i = normal_flux_fine(i) * normal_flip[i];
//                normal_flux_i_0 += (fabs(fine_flux_i) - fine_flux_i) / 2;
//                normal_flux_i_1 += (fabs(fine_flux_i) + fine_flux_i) / 2;
//            }
        }

        if (f_tf_e_diag_.RowSize(face) == 2) // facet is interior
        {
            const int* elem_pair = f_tf_e_diag_.GetRowColumns(face);
            diag.Set(elem_pair[0], elem_pair[1], -normal_flux_i_1);
            diag.Add(elem_pair[1], elem_pair[1], normal_flux_i_1);
            diag.Set(elem_pair[1], elem_pair[0], -normal_flux_i_0);
            diag.Add(elem_pair[0], elem_pair[0], normal_flux_i_0);
        }
        else
        {
            const int diag_elem = f_tf_e_diag_.GetRowColumns(face)[0];

            if (f_tf_e_offd_.RowSize(face) > 0) // facet is shared
            {
                assert(f_tf_e_offd_.RowSize(face) == 1);
                const int offd_elem = f_tf_e_offd_.GetRowColumns(face)[0];
                if (f_tf_diag_.RowSize(face) > 0) // facet is owned by local proc
                {
                    offd.Set(diag_elem, offd_elem, -normal_flux_i_1);
                    diag.Add(diag_elem, diag_elem, normal_flux_i_0);
                }
                else // facet is owned by the neighbor proc
                {
                    diag.Add(diag_elem, diag_elem, normal_flux_i_1);
                    offd.Set(diag_elem, offd_elem, -normal_flux_i_0);
                }
            }
            else // global boundary
            {
                assert(f_tf_e_diag_.RowSize(face) == 1);
                diag.Add(diag_elem, diag_elem, normal_flux_i_0);
            }
        }
    }
    diag.Finalize();
    offd.Finalize();

    if (level_ == Fine)
    {
        transport_op_ = make_unique<mfem::HypreParMatrix>(
                    comm_, tran_op_starts_.Last(), tran_op_starts_.Last(),
                    tran_op_starts_, tran_op_starts_, &diag, &offd, tran_op_colmap_);

        // Adjust ownership
        transport_op_->SetOwnerFlags(3, 3, 0);
        diag.LoseData();
        offd.LoseData();
    }
    else
    {
        mfem::Array<int> tmp_starts;
        GenerateOffsets(comm_, up_.GetPu().Height(), tmp_starts);
        mfem::HypreParMatrix transport_op_fine(comm_, tmp_starts.Last(), tmp_starts.Last(),
                                               tmp_starts, tmp_starts, &diag, &offd, tran_op_colmap_);

        auto PuT = smoothg::Transpose(up_.GetPu());
        unique_ptr<mfem::HypreParMatrix> tmp(transport_op_fine.LeftDiagMult(PuT, tran_op_starts_));
        unique_ptr<mfem::HypreParMatrix> tmpT(tmp->Transpose());
        unique_ptr<mfem::HypreParMatrix> tmpcT(tmpT->LeftDiagMult(PuT, tran_op_starts_));

        transport_op_.reset(tmpcT->Transpose());
    }
}

void TwoPhaseOp::dRvdv(const mfem::Vector& normal_flux, const mfem::Vector& FS)
{
    mfem::Array<int> facedofs;
    mfem::Vector normal_flux_loc;
    mfem::Vector normal_flux_fine;

    mfem::SparseMatrix diag(f_tf_diag_.Width(), f_tf_e_diag_.Width());
    mfem::SparseMatrix offd(f_tf_diag_.Width(), f_tf_e_offd_.Width());
    for (int face = 0; face < f_tf_diag_.Height(); face++)
    {
        double normal_flux_i_0 = 0.0;
        double normal_flux_i_1 = 0.0;

        if (level_ == Fine)
        {
            // confusing direction, just be consistent with MakeTransportOp
            normal_flux_i_0 = normal_flux(face) < 0 ? -1.0 : 0.0;
            normal_flux_i_1 = -1.0 - normal_flux_i_0;
        }

        assert(f_tf_diag_.RowSize(face) == 1);
        int trueface = f_tf_diag_.GetRowColumns(face)[0];

        const int* elem_pair = f_tf_e_diag_.GetRowColumns(face);
        diag.Set(trueface, elem_pair[0], normal_flux_i_0);

        if (f_tf_e_diag_.RowSize(face) == 2) // facet is interior
        {
            diag.Set(trueface, elem_pair[1], normal_flux_i_1);
        }
        else
        {
            if (f_tf_e_offd_.RowSize(face) > 0) // facet is shared
            {
                assert(f_tf_e_offd_.RowSize(face) == 1);
                const int offd_elem = f_tf_e_offd_.GetRowColumns(face)[0];
                if (f_tf_diag_.RowSize(face) > 0) // facet is owned by local proc
                {
                    offd.Set(trueface, offd_elem, normal_flux_i_1);
                }
            }
        }
    }
    diag.Finalize();
    offd.Finalize();

    dRvdv_ = make_unique<mfem::HypreParMatrix>(
                comm_, num_tf_, tran_op_starts_.Last(),
                tf_starts_, tran_op_starts_, &diag, &offd, tran_op_colmap_);

    // Adjust ownership
    dRvdv_->SetOwnerFlags(3, 3, 0);
    diag.LoseData();
    offd.LoseData();
}

void TwoPhaseOp::MakeTransportSolver(const mfem::Vector& normal_flux)
{
    MakeTransportOp(normal_flux);

    transport_prec_ = make_unique<mfem::HypreBoomerAMG>(*transport_op_);
    transport_prec_->SetPrintLevel(0);
    transport_solver_.SetOperator(*transport_op_);
    transport_solver_.SetPreconditioner(*transport_prec_);
}

void TwoPhaseOp::Update(double dt, const mfem::Vector& prev_state)
{
    dt_ = dt;
    prev_state_.Update(prev_state.GetData(), offsets_);

    mfem::Vector prev_sat(prev_state_.GetBlock(1));
    RescaleVector(M_s_, prev_sat);
    prev_sat /= dt_;
    *source_t_ = *source_0_;
    source_t_->GetBlock(1) += prev_sat;
}

TPH::TPH(const SPE10Problem& spe10problem, FiniteVolumeMLMC& up, SolveType solve_type)
    : Hierarchy(up.GetComm(), 2), up_(up), ops_(2), helper_(2), flow_offsets_(2)
{
    ops_[0] = make_unique<TwoPhaseOp>(spe10problem, up_, Fine, solve_type);
    ops_[1] = make_unique<TwoPhaseOp>(spe10problem, up_, Coarse, solve_type);
    ops_[0]->SetPrintLevel(-1);
    ops_[1]->SetPrintLevel(-1);
    ops_[0]->SetMaxIter(1);
    ops_[1]->SetMaxIter(1);
//    ops_[1]->SetRelTol(1e-7);
//    ops_[1]->SetRelTol(1e-9);

//    up_.SetMaxIter(1);
//    up_.SetRelTol(1e-12);
//    up_.SetAbsTol(1e-15);

    up_.FineBlockOffsets(flow_offsets_[0]);
    up_.CoarseBlockOffsets(flow_offsets_[1]);
    prev_state_coarse_.SetSize(ops_[1]->GetOffsets().Last());

    auto PuT = smoothg::Transpose(up_.GetPu());
    auto PuTPu = smoothg::Mult(PuT, up_.GetPu());
    PuTPu.GetDiag(PuTPu_diag);
}

void TPH::Mult(int level, const mfem::Vector& x, mfem::Vector& Rx) const
{
    ops_[level]->Mult(x, Rx);
}

void TPH::Solve(int level, const mfem::Vector& rhs, mfem::Vector& sol)
{
    ops_[level]->Solve(rhs, sol);
}

void TPH::Restrict(int level, const mfem::Vector& fine, mfem::Vector& coarse) const
{
    assert(level == 0);
    helper_[0].Update(fine.GetData(), ops_[0]->GetOffsets());
    helper_[1].Update(coarse.GetData(), ops_[1]->GetOffsets());
    mfem::BlockVector helper_00(helper_[0].GetData(), flow_offsets_[0]);
    mfem::BlockVector helper_10(helper_[1].GetData(), flow_offsets_[1]);

    up_.Restrict(helper_00, helper_10);
    up_.Restrict(helper_[0].GetBlock(1), helper_[1].GetBlock(1));
}

void TPH::Interpolate(int level, const mfem::Vector& coarse, mfem::Vector& fine) const
{
    assert(level == 1);
    helper_[0].Update(fine.GetData(), ops_[0]->GetOffsets());
    helper_[1].Update(coarse.GetData(), ops_[1]->GetOffsets());
    mfem::BlockVector helper_00(helper_[0].GetData(), flow_offsets_[0]);
    mfem::BlockVector helper_10(helper_[1].GetData(), flow_offsets_[1]);

    up_.Interpolate(helper_10, helper_00);
    up_.Interpolate(helper_[1].GetBlock(1), helper_[0].GetBlock(1));
}

void TPH::Project(int level, const mfem::Vector& fine, mfem::Vector& coarse) const
{
    assert(level == 0);
    helper_[0].Update(fine.GetData(), ops_[0]->GetOffsets());
    helper_[1].Update(coarse.GetData(), ops_[1]->GetOffsets());
    mfem::BlockVector helper_00(helper_[0].GetData(), flow_offsets_[0]);
    mfem::BlockVector helper_10(helper_[1].GetData(), flow_offsets_[1]);

    up_.Project(helper_00, helper_10);
    up_.Project(helper_[0].GetBlock(1), helper_[1].GetBlock(1));
    InvRescaleVector(PuTPu_diag, helper_[1].GetBlock(1));
}

void TPH::Smoothing(int level, const mfem::Vector& in, mfem::Vector& out) const
{
//    up_.SetMaxIter(1);
    ops_[level]->Solve(in, out);
//    up_.SetMaxIter(5000);
}

void TPH::Update(double dt, const mfem::Vector& prev_state)
{
    ops_[0]->Update(dt, prev_state);
    Project(0, prev_state, prev_state_coarse_);
    ops_[1]->Update(dt, prev_state_coarse_);
}

NMG_TP::NMG_TP(const SPE10Problem& spe10problem, FiniteVolumeMLMC& up,
               Cycle cycle, SolveType solve_type)
    : TwoPhaseTDO(up.GetNumTotalDofs(0)+up.GetNumVertexDofs(0), solve_type),
      tph_(spe10problem, up, solve_type), nl_mg_(tph_, cycle)
{
    nl_mg_.SetPrintLevel(0);
}

// LamS = S^2 + ((1-S)^2) / 5
void TotalMobility(const mfem::Vector& S, mfem::Vector& LamS)
{
    for (int i = 0; i < S.Size(); i++)
    {
        double S_w = S(i);
        double S_o = 1.0 - S_w;
        LamS(i)  = S_w * S_w + S_o * S_o / 5.0;
    }
}

// FS = S^2 / LamS
void FractionalFlow(const mfem::Vector& S, mfem::Vector& FS)
{
    for (int i = 0; i < S.Size(); i++)
    {
        double S_w = S(i);
        double S_o = 1.0 - S_w;
        double LamS  = S_w * S_w + S_o * S_o / 5.0;
        FS(i) = S_w * S_w / LamS;
    }
}

// dLamS = 2 S + 2 (1-S) / 5
void TotalMobilityDerivative(const mfem::Vector& S, mfem::Vector& dLamS)
{
    const double constant1 = 12.0 / 5.0;
    const double constant2 = constant1 - 2.0;
    for (int i = 0; i < S.Size(); i++)
    {
        dLamS(i)  = constant1 * S(i) - constant2;
    }
}

// dFS = 10 S (1-S) / (6 S^2 - 2S + 1)^2 = 0.4 S (1-S) / LamS^2
void FractionalFlowDerivative(const mfem::Vector& S, mfem::Vector& dFS)
{
    for (int i = 0; i < S.Size(); i++)
    {
        double S_w = S(i);
        double S_o = 1.0 - S_w;
        double LamS  = S_w * S_w + S_o * S_o / 5.0;
        dFS(i) = 0.4 * S_w * S_o / (LamS * LamS);
    }
}

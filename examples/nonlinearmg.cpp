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
   @brief Two phase flow transport
*/
class TwoPhaseOp : public mfem::TimeDependentOperator
{
public:
    TwoPhaseOp(const SPE10Problem& spe10problem, FiniteVolumeMLMC& up, Level level);

    virtual ~TwoPhaseOp() = default;

    // Compute the right-hand side of the ODE system.
    void Mult(const mfem::Vector& x, mfem::Vector& fx) const;
    // Solve Backward-Euler equation: k = f(x + dt*k, t) for k using Picard method.
    void ImplicitSolve(const double dt, const mfem::Vector& x, mfem::Vector& k);

    /// @note sol is used as initial guess for the nonlinear solve

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

    const mfem::Array<int>& GetOffsets() const { return offsets_; }

protected:
    // Setup needed for transport operator = M_s^{-1} Adv
    void SetupTransportOp(const mfem::SparseMatrix& elem_facet,
                          const mfem::HypreParMatrix& facet_truefacet);
    void MakeTransportOp(const mfem::Vector& normal_flux);
    void MakeTransportSolver(const mfem::Vector& normal_flux);

    void PicardSolve(const double dt, const mfem::Vector& x, mfem::Vector& k);
    void NewtonSolve(const double dt, const mfem::Vector& x, mfem::Vector& k);

    // default solver options
    int print_level_ = 0;
    int max_num_iter_ = 10;
    double rtol_ = 1e-9;
    double atol_ = 1e-12;

    int nnz_;
    mutable int iter_;
    mutable double timing_;
    bool converged_;

    MPI_Comm comm_;
    int myid_;

    Level level_;
    FiniteVolumeMLMC& up_;
    mfem::SparseMatrix f_tf_diag_; // diag part of facet_truefacet
    unique_ptr<mfem::HypreParMatrix> facet_truefacet_elem_;
    mfem::SparseMatrix f_tf_e_diag_;
    mfem::SparseMatrix f_tf_e_offd_;
    mfem::Vector M_s_inv_;

    SolveType solve_type_;

    ///@name components for transport_op_
    ///@{
    HYPRE_Int* tran_op_colmap_;
    mfem::Array<int> tran_op_starts_;
    unique_ptr<mfem::HypreParMatrix> transport_op_;
    ///@}

    mfem::Array<int> offsets_;
    unique_ptr<mfem::Operator> flow_solver_;
    unique_ptr<mfem::HypreBoomerAMG> transport_prec_;
    mfem::CGSolver transport_solver_;
    unique_ptr<mfem::BlockOperator> solver_;

    mutable mfem::BlockVector in_help_;
    mutable mfem::BlockVector out_help_;
    mutable mfem::Vector tot_mob_;   // total mobility
    mutable mfem::Vector frac_flow_;  // factional flow
    unique_ptr<mfem::BlockVector> rhs_;
};

void TotalMobility(const mfem::Vector& S, mfem::Vector& LamS);
void FractionalFlow(const mfem::Vector& S, mfem::Vector& FS);

// two phase hierarchy
class TPH : public Hierarchy
{
public:
    TPH(const SPE10Problem& spe10problem, FiniteVolumeMLMC& up);
    TPH() = delete;

    /// Evaluates the action of the operator out = A[level](in)
    virtual void Mult(int level, const mfem::Vector& in, mfem::Vector& out) const;

    /// Solves the (possibly nonlinear) problem A[level](sol) = rhs
    virtual void Solve(int level, const mfem::Vector& rhs, mfem::Vector& sol);

    /// Restrict a vector from level to level+1 (coarser level)
    virtual void Restrict(int level, const mfem::Vector& fine, mfem::Vector& coarse) const;

    /// Interpolate a vector from level to level-1 (finer level)
    virtual void Interpolate(int level, const mfem::Vector& coarse, mfem::Vector& fine) const;

    /// Relaxation on each level
    virtual void Smoothing(int level, const mfem::Vector& in, mfem::Vector& out) const;

    virtual const int DomainSize(int level) const { return ops_[level]->Width(); }
    virtual const int RangeSize(int level) const { return ops_[level]->Height(); }

    void SetTimeStep(double dt) { dt_ = dt; }
private:
    FiniteVolumeMLMC& up_;
    std::vector<unique_ptr<TwoPhaseOp>> ops_;
    mutable std::vector<mfem::BlockVector> helper_;

    double dt_;
};

mfem::Vector TwoPhaseFlow(const SPE10Problem& spe10problem, FiniteVolumeMLMC &up, double delta_t,
                          double total_time, int vis_step, Level level,
                          const std::string& caption);

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
    if (hybridization)
    {
        fvupscale.SetAbsTol(1e-15);
        fvupscale.SetRelTol(1e-12);
    }

    // Fine scale transport based on fine flux
    auto S_fine = TwoPhaseFlow(
                spe10problem, fvupscale, delta_t, total_time, vis_step,
                Fine, "saturation based on fine scale upwind");

    auto S_coarse = TwoPhaseFlow(
                spe10problem, fvupscale, delta_t, total_time, vis_step,
                Coarse, "saturation based on fastRAP");

    double sat_err = CompareError(comm, S_coarse, S_fine);
    if (myid == 0)
    {
        std::cout << "Saturation errors: " << sat_err << "\n";
    }

    return EXIT_SUCCESS;
}

TwoPhaseOp::TwoPhaseOp(const SPE10Problem &spe10problem, FiniteVolumeMLMC &up, Level level)
    : mfem::TimeDependentOperator(up.GetNumTotalDofs(level)+up.GetNumVertexDofs(level), 0.0, IMPLICIT),
      comm_(up.GetComm()), level_(level), up_(up), solve_type_(Picard), transport_solver_(comm_)
{
    MPI_Comm_rank(comm_, &myid_);
    offsets_.SetSize(3, 0);
    offsets_[1] = up.GetMatrix(level_).GetNumTotalDofs();
    offsets_[2] = offsets_[1] + up.GetMatrix(level_).GetNumVertexDofs();

    rhs_ = make_unique<mfem::BlockVector>(offsets_);

    mfem::SparseMatrix M_s = SparseIdentity(spe10problem.GetVertexRHS().Size());
    M_s *= spe10problem.CellVolume();
    if (level_ == Fine)
    {
        SetupTransportOp(spe10problem.GetVertexEdge(), spe10problem.GetEdgeToTrueEdge());

        rhs_->GetBlock(1) = spe10problem.GetVertexRHS();
        rhs_->GetBlock(0).SetVector(spe10problem.GetEdgeRHS(), 0);
        rhs_->GetBlock(0).SetVector(rhs_->GetBlock(1), up.GetFineMatrix().GetNumEdgeDofs());

        M_s.GetDiag(M_s_inv_);
        flow_solver_ = make_unique<UpscaleFineBlockSolve>(up_);
    }
    else
    {
        SetupTransportOp(up_.GetAggFace(), up_.GetFaceTrueFace());

        mfem::BlockVector tmp(up_.GetFineBlockVector());
        tmp.GetBlock(0) = spe10problem.GetEdgeRHS();
        tmp.GetBlock(1) = spe10problem.GetVertexRHS();
        mfem::BlockVector tmp_c = up_.Restrict(tmp);
        rhs_->GetBlock(0).SetVector(tmp_c, 0);
        rhs_->GetBlock(1) = tmp_c.GetBlock(1);

        unique_ptr<mfem::SparseMatrix> M_s_c(mfem::RAP(up.GetPu(), M_s, up.GetPu()));
        M_s_c->GetDiag(M_s_inv_);
        flow_solver_ = make_unique<UpscaleCoarseBlockSolve>(up_);
    }
    for (int i = 0; i < M_s_inv_.Size(); i++)
    {
        M_s_inv_(i) = 1.0 / M_s_inv_(i);
    }
    RescaleVector(M_s_inv_, rhs_->GetBlock(1));
    rhs_->GetBlock(1) *= -1.0;  // g = -f

    up_.SetMaxIter(level_ == Fine ? 10 : 5000);
    up_.SetPrintLevel(-1);

    solver_ = make_unique<mfem::BlockOperator>(offsets_);
    solver_->SetBlock(0, 0, flow_solver_.get());

    tot_mob_.SetSize(rhs_->BlockSize(1));
    frac_flow_.SetSize(rhs_->BlockSize(1));

    transport_solver_.SetPrintLevel(print_level_);
    transport_solver_.SetMaxIter(5000);
    transport_solver_.SetRelTol(rtol_);
    transport_solver_.SetAbsTol(atol_);
}

void TwoPhaseOp::Mult(const mfem::Vector &x, mfem::Vector &fx) const
{
    mfem::BlockVector in_help(x.GetData(), offsets_);
    mfem::BlockVector out_help(fx.GetData(), offsets_);

//    out_help_.GetBlock(0) = 0.0;

    out_help.GetBlock(1) = rhs_->GetBlock(1);
    FractionalFlow(in_help.GetBlock(1), frac_flow_);
    transport_op_->Mult(-1.0, frac_flow_, 1.0, out_help.GetBlock(1));
}

void TwoPhaseOp::ImplicitSolve(const double dt, const mfem::Vector& x, mfem::Vector& k)
{
    assert(solve_type_ == Newton || solve_type_ == Picard);

    if (solve_type_ == Picard)
    {
        PicardSolve(dt, x, k);
    }
    else
    {
        NewtonSolve(dt, x, k);
    }
}

void TwoPhaseOp::PicardSolve(const double dt, const mfem::Vector& x, mfem::Vector& k)
{
    // in_help_ = state at previous time step t = n
    in_help_.Update(x.GetData(), offsets_);

    // at return out_help_ is the change (x^{n+1} - x^n) / dt
    out_help_.Update(k.GetData(), offsets_);
    out_help_ = in_help_;

    // rhs_s = dt * M_s^{-1}g + S^n
    mfem::Vector rhs_s(in_help_.GetBlock(1));
    rhs_s.Add(dt, rhs_->GetBlock(1));
    double norm = mfem::ParNormlp(rhs_s, 2, comm_);

    FractionalFlow(out_help_.GetBlock(1), frac_flow_);

    mfem::BlockVector residual(offsets_);
    for (iter_ = 0; iter_ < max_num_iter_; iter_++)
    {
        TotalMobility(out_help_.GetBlock(1), tot_mob_);
        up_.RescaleCoefficient(level_, tot_mob_);
        flow_solver_->Mult(rhs_->GetBlock(0), out_help_.GetBlock(0));

        // TransportOp(S) = M_s^{-1} Adv F(S)
        MakeTransportOp(out_help_.GetBlock(0));

        out_help_.GetBlock(1) = rhs_s;
        transport_op_->Mult(-dt, frac_flow_, 1.0, out_help_.GetBlock(1));

        Mult(out_help_, residual); // Note: frac_flow_ is updated in Mult
        residual.GetBlock(1) *= dt;
        residual.GetBlock(1) += in_help_.GetBlock(1);
        residual.GetBlock(1) -= out_help_.GetBlock(1);

        double resid = mfem::ParNormlp(residual.GetBlock(1), 2, comm_);
        double rel_resid = resid / norm;
        if (myid_ == 0 && print_level_ > 0)
        {
            std::cout << "Picard iter " << iter_
                      << ": relative residual norm = " << rel_resid << "\n";
        }
        if (resid < atol_ || rel_resid < rtol_)
        {
            break;
        }
    }

    converged_ = (iter_ != max_num_iter_);
    if (myid_ == 0 && !converged_ && print_level_ >= 0)
    {
        std::cout << "Warning: Picard iteration reached maximum number of iterations!\n";
    }

    // k = dx / dt (expected output of TimeDependentOperator::ImplicitSolve)
    out_help_.GetBlock(1) -= in_help_.GetBlock(1);
    out_help_.GetBlock(1) /= dt;
}

void TwoPhaseOp::NewtonSolve(const double dt, const mfem::Vector& x, mfem::Vector& k)
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
}

void TwoPhaseOp::MakeTransportOp(const mfem::Vector& normal_flux)
{
    mfem::Array<int> facedofs;
    mfem::Vector normal_flux_loc;
    mfem::Vector normal_flux_fine;

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
            const auto& normal_flip = up_.GetCoarseToFineNormalFlip()[face];
            GetTableRow(up_.GetFaceToFaceDof(), face, facedofs);
            normal_flux.GetSubVector(facedofs, normal_flux_loc);
            normal_flux_loc *= -1.0; // P matrix stores negative traces
            normal_flux_fine.SetSize(normal_flip.size());
            up_.GetTraces()[face].Mult(normal_flux_loc, normal_flux_fine);
            for (int i = 0; i < normal_flux_fine.Size(); i++)
            {
                double fine_flux_i = normal_flux_fine(i) * normal_flip[i];
                normal_flux_i_0 += (fabs(fine_flux_i) - fine_flux_i) / 2;
                normal_flux_i_1 += (fabs(fine_flux_i) + fine_flux_i) / 2;
            }
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
    diag.Finalize(0);
    offd.Finalize(0);

    diag.ScaleRows(M_s_inv_);
    offd.ScaleRows(M_s_inv_);

    transport_op_ = make_unique<mfem::HypreParMatrix>(
                comm_, tran_op_starts_.Last(), tran_op_starts_.Last(),
                tran_op_starts_, tran_op_starts_, &diag, &offd, tran_op_colmap_);

    // Adjust ownership
    transport_op_->SetOwnerFlags(3, 3, 0);
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

mfem::socketstream sout;
int option = 0;
bool setup = true;
mfem::Vector TwoPhaseFlow(const SPE10Problem& spe10problem, FiniteVolumeMLMC& up,
                          double delta_t, double total_time, int vis_step, Level level,
                          const std::string& caption)
{
    option++;

    int myid;
    MPI_Comm_rank(up.GetComm(), &myid);

    mfem::StopWatch chrono;

    chrono.Clear();
    MPI_Barrier(up.GetComm());
    chrono.Start();

    double time = 0.0;

    TwoPhaseOp two_phase_op(spe10problem, up, level);
    two_phase_op.SetTime(time);

    mfem::BackwardEulerSolver ode_solver;
    ode_solver.Init(two_phase_op);

    // state = (flux, pressure, saturation)
    mfem::BlockVector state(two_phase_op.GetOffsets());
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

//    std::vector<mfem::Vector> sats(well_vertices.Size(), mfem::Vector(total_time / delta_t + 2));
//    for (unsigned int i = 0; i < sats.size(); i++)
//    {
//        sats[i] = 0.0;
//    }

    bool done = false;
    double dt_real = std::min(delta_t, total_time - time);
    for (int ti = 0; !done; )
    {
        dt_real = std::min(std::min(delta_t, total_time - time), dt_real * 2.0);
        previous_state = state;
        ode_solver.Step(state, time, dt_real);
        while (!two_phase_op.IsConverged()) // TODO: add smallest step size condition
        {
            time -= dt_real;
            dt_real /= 2.0;
            if (myid == 0)
            {
                std::cout << "Restart nonlinear solve with time step size " << dt_real << "\n";
            }
            state = previous_state;
            ode_solver.Step(state, time, dt_real);
        }
        ti++;

        //        for (unsigned int i = 0; i < sats.size(); i++)
        //        {
        //            if (level == Coarse)
        //            {
        //                up.Interpolate(S, S_vis);
        //                sats[i](ti) = S_vis(well_vertices[i]);
        //            }
        //            else
        //            {
        //                sats[i](ti) = S(well_vertices[i]);
        //            }
        //        }

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
        std::cout << "Time stepping done in " << chrono.RealTime() << "s.\n";
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

//    for (unsigned int i = 0; i < sats.size(); i++)
//    {
//        std::ofstream ofs("sat_prod_" + std::to_string(i) + "_" + std::to_string(myid)
//                          + "_" + std::to_string(option) + ".txt");
//        sats[i].Print(ofs, 1);
//    }

    return out;
}

void TotalMobility(const mfem::Vector& S, mfem::Vector& LamS)
{
    for (int i = 0; i < S.Size(); i++)
    {
        double S_w = S(i);
        double S_o = 1.0 - S_w;
        LamS(i)  = S_w * S_w + S_o * S_o / 5.0;
    }
}

void FractionalFlow(const mfem::Vector& S, mfem::Vector& FS)
{
    for (int i = 0; i < S.Size(); i++)
    {
        double S_w = S(i);
        double S_o = 1.0 - S_w;
        double Lam_S  = S_w * S_w + S_o * S_o / 5.0;
        FS(i) = S_w * S_w / Lam_S;
    }
}

TPH::TPH(const SPE10Problem& spe10problem, FiniteVolumeMLMC& up)
    : Hierarchy(2), up_(up), ops_(2), helper_(2), dt_(1.0)
{
    ops_[0] = make_unique<TwoPhaseOp>(spe10problem, up_, Fine);
    ops_[1] = make_unique<TwoPhaseOp>(spe10problem, up_, Coarse);
}

void TPH::Mult(int level, const mfem::Vector& in, mfem::Vector& out) const
{
    ops_[level]->Mult(in, out);
}

void TPH::Solve(int level, const mfem::Vector& rhs, mfem::Vector& sol)
{
    ops_[level]->ImplicitSolve(dt_, rhs, sol);
    sol *= dt_;
    sol += rhs;
}

void TPH::Restrict(int level, const mfem::Vector& fine, mfem::Vector& coarse) const
{
    assert(level == 0);
    helper_[0].Update(fine.GetData(), ops_[0]->GetOffsets());
    helper_[1].Update(coarse.GetData(), ops_[1]->GetOffsets());

    up_.Restrict(helper_[0].GetBlock(0), helper_[1].GetBlock(0));
    up_.Restrict(helper_[0].GetBlock(1), helper_[1].GetBlock(1));
}

void TPH::Interpolate(int level, const mfem::Vector& coarse, mfem::Vector& fine) const
{
    assert(level == 1);
    helper_[0].Update(fine.GetData(), ops_[0]->GetOffsets());
    helper_[1].Update(coarse.GetData(), ops_[1]->GetOffsets());
    up_.Interpolate(helper_[1].GetBlock(0), helper_[0].GetBlock(0));
    up_.Interpolate(helper_[1].GetBlock(1), helper_[0].GetBlock(1));
}

void TPH::Smoothing(int level, const mfem::Vector& in, mfem::Vector& out) const
{
    out = in;
}


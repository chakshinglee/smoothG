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

/** A time-dependent operator for the right-hand side of the ODE. The semi-discrete
    equation of du/dt = b - v.grad(u) is M du/dt = b - K u, where M and K are
    the mass and advection matrices, and b describes the influx source. This can
    be written as a general ODE, du/dt = M^{-1} (K u + b), and this class is
    used to evaluate the right-hand side. */
class FV_Evolution : public mfem::TimeDependentOperator
{
private:
   const mfem::HypreParMatrix& K_;
   mfem::Vector Minv_;
   const mfem::Vector& b_;
public:
   FV_Evolution(const mfem::SparseMatrix& M, const mfem::HypreParMatrix& K,
                const mfem::Vector& b);
   virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const;
   virtual ~FV_Evolution() { }
};

std::unique_ptr<mfem::HypreParMatrix> DiscreteAdvection(
        const mfem::Vector& normal_flux, const mfem::SparseMatrix& elem_facet,
        const mfem::HypreParMatrix& facet_truefacet);

mfem::Vector Transport(const SPE10Problem& spe10problem, const mfem::BlockVector& normal_flux,
                       double delta_t, double total_time, int vis_step, const std::string& caption);

int main(int argc, char* argv[])
{
    int num_procs, myid;
    picojson::object serialize;

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
    bool dual_target = false;
    args.AddOption(&dual_target, "-du", "--dual-target", "-no-du",
                   "--no-dual-target", "Use dual graph Laplacian in trace generation.");
    bool scaled_dual = false;
    args.AddOption(&scaled_dual, "-sd", "--scaled-dual", "-no-sd",
                   "--no-scaled-dual", "Scale dual graph Laplacian by (inverse) edge weight.");
    bool energy_dual = false;
    args.AddOption(&energy_dual, "-ed", "--energy-dual", "-no-ed",
                   "--no-energy-dual", "Use energy matrix in trace generation.");
    bool visualization = false;
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization", "Enable visualization.");
    double delta_t = 1.0;
    args.AddOption(&delta_t, "-dt", "--delta-t", "Time step.");
    double total_time = 1000.0;
    args.AddOption(&total_time, "-time", "--total-time", "Total time to step.");
    int vis_step = 10;
    args.AddOption(&vis_step, "-vs", "--vis-step",
                   "Step size for visualization.");
    int write_step = 0;
    args.AddOption(&write_step, "-ws", "--write-step",
                   "Step size for writing data to file.");
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
    int coarsening_factor = 100;
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
    auto edge_d_td = spe10problem.GetEdgeToTrueEdge();
    auto& weight = spe10problem.GetWeight();
    auto& edge_bdr_att = spe10problem.GetEdgeBoundaryAttributeTable();
    auto& rhs_sigma_fine = spe10problem.GetEdgeRHS();
    auto& rhs_u_fine = spe10problem.GetVertexRHS();
    auto& well_list = spe10problem.GetWells();
    auto& ess_edof_marker = spe10problem.GetEssentialEdgeDofsMarker();

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

    mfem::Array<int> well_vertices;
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

    int nparts = std::max(vertex_edge.Height() / coarsening_factor, 1);
    bool adaptive_part = false;
    bool use_edge_weight = (nDimensions == 3) && (nz > 1);
    mfem::Array<int> partition;
    PartitionVerticesByMetis(vertex_edge, weight, well_vertices, nparts,
                             partition, adaptive_part, use_edge_weight);

    // Create Upscaler and Solve
    FiniteVolumeUpscale fvupscale(comm, vertex_edge, weight, partition, *edge_d_td,
                                  edge_bdr_att, ess_attr, spect_tol, max_evects,
                                  dual_target, scaled_dual, energy_dual, hybridization);
    fvupscale.PrintInfo();
    fvupscale.ShowSetupTime();

    mfem::BlockVector rhs_fine(fvupscale.GetFineBlockVector());
    rhs_fine.GetBlock(0) = rhs_sigma_fine;
    rhs_fine.GetBlock(1) = rhs_u_fine;

    // Fine scale transport based on fine flux
    fvupscale.MakeFineSolver(ess_edof_marker);
    auto sol_fine = fvupscale.SolveFine(rhs_fine);
    fvupscale.ShowFineSolveInfo();
    auto S_fine = Transport(spe10problem, sol_fine, delta_t, total_time,
                            vis_step, "Saturation based on fine scale flux");

    // Fine scale transport based on upscaled flux
    auto sol_upscaled = fvupscale.Solve(rhs_fine);
    fvupscale.ShowCoarseSolveInfo();
    auto S_upscaled = Transport(spe10problem, sol_upscaled, delta_t, total_time,
                                vis_step, "Saturation based on coarse scale flux");

    auto error_info = fvupscale.ComputeErrors(sol_upscaled, sol_fine);
    double sat_err = CompareError(comm, S_upscaled, S_fine);
    if (myid == 0)
    {
        std::cout << "Flow errors:\n";
        ShowErrors(error_info);
        std::cout << "Saturation errors: " << sat_err << "\n";
    }

    return EXIT_SUCCESS;
}

FV_Evolution::FV_Evolution(const mfem::SparseMatrix& M, const mfem::HypreParMatrix& K,
                           const mfem::Vector& b)
   : mfem::TimeDependentOperator(K.Height()), K_(K), b_(b)
{
   M.GetDiag(Minv_); // assume M is diagonal
   for (int i = 0; i < Minv_.Size(); i++)
   {
       Minv_(i) = 1.0 / Minv_(i);
   }
}

void FV_Evolution::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
   // y = M^{-1} (b - K x)
   y = b_;
   K_.Mult(-1.0, x, 1.0, y);
   RescaleVector(Minv_, y);
}

std::unique_ptr<mfem::HypreParMatrix> DiscreteAdvection(
        const mfem::Vector& normal_flux, const mfem::SparseMatrix& elem_facet,
        const mfem::HypreParMatrix& facet_truefacet)
{
    MPI_Comm comm = facet_truefacet.GetComm();
    const int num_elems_diag = elem_facet.Height();
    const int num_facets = elem_facet.Width();

    mfem::Array<int> elem_starts;
    GenerateOffsets(comm, num_elems_diag, elem_starts);

    using ParMatPtr = std::unique_ptr<mfem::HypreParMatrix>;
    ParMatPtr elem_truefacet(facet_truefacet.LeftDiagMult(elem_facet, elem_starts));
    ParMatPtr truefacet_elem(elem_truefacet->Transpose());
    ParMatPtr facet_truefacet_elem(mfem::ParMult(&facet_truefacet, truefacet_elem.get()));

    mfem::SparseMatrix f_tf_e_diag, f_tf_e_offd, f_tf_diag;
    HYPRE_Int* elem_map;
    facet_truefacet_elem->GetDiag(f_tf_e_diag);
    facet_truefacet_elem->GetOffd(f_tf_e_offd, elem_map);
    facet_truefacet.GetDiag(f_tf_diag);

    HYPRE_Int* elem_map_copy = new HYPRE_Int[f_tf_e_offd.Width()];
    std::copy_n(elem_map, f_tf_e_offd.Width(), elem_map_copy);

    mfem::SparseMatrix diag(num_elems_diag, num_elems_diag);
    mfem::SparseMatrix offd(num_elems_diag, f_tf_e_offd.Width());

    for (int ifacet = 0; ifacet < num_facets; ifacet++)
    {
        const double normal_flux_i = normal_flux(ifacet);

        if (f_tf_e_diag.RowSize(ifacet) == 2) // facet is interior
        {
            const int* elem_pair = f_tf_e_diag.GetRowColumns(ifacet);

            if (normal_flux_i > 0)
            {
                diag.Set(elem_pair[0], elem_pair[1], -1.0 * normal_flux_i);
                diag.Add(elem_pair[1], elem_pair[1], normal_flux_i);
            }
            else
            {
                diag.Set(elem_pair[1], elem_pair[0], normal_flux_i);
                diag.Add(elem_pair[0], elem_pair[0], -1.0 * normal_flux_i);
            }
        }
        else
        {
            assert(f_tf_e_diag.RowSize(ifacet) == 1);
            const int diag_elem = f_tf_e_diag.GetRowColumns(ifacet)[0];

            if (f_tf_e_offd.RowSize(ifacet) == 1) // facet is shared
            {
                const int offd_elem = f_tf_e_offd.GetRowColumns(ifacet)[0];

                if (f_tf_diag.RowSize(ifacet) > 0) // facet is owned by local proc
                {
                    if (normal_flux_i > 0)
                    {
                        offd.Set(diag_elem, offd_elem, -1.0 * normal_flux_i);
                    }
                    else
                    {
                        diag.Add(diag_elem, diag_elem, -1.0 * normal_flux_i);
                    }
                }
                else // facet is owned by the neighbor proc
                {
                    if (normal_flux_i > 0)
                    {
                        diag.Add(diag_elem, diag_elem, normal_flux_i);
                    }
                    else
                    {
                        offd.Set(diag_elem, offd_elem, normal_flux_i);
                    }
                }
            }
            else  // global boundary
            {
                if (normal_flux_i < 0)
                {
                    diag.Add(diag_elem, diag_elem, -1.0 * normal_flux_i);
                }
            }
        }
    }

    diag.Finalize(0);
    offd.Finalize(0);

    int num_elems = elem_starts.Last();
    auto out = make_unique<mfem::HypreParMatrix>(comm, num_elems, num_elems, elem_starts,
                                                 elem_starts, &diag, &offd, elem_map_copy);

    // Adjust ownership and copy starts arrays
    out->CopyRowStarts();
    out->CopyColStarts();
    out->SetOwnerFlags(3, 3, 1);

    diag.LoseData();
    offd.LoseData();

    return out;
}

mfem::Vector Transport(const SPE10Problem& spe10problem, const mfem::BlockVector& flow_sol,
                       double delta_t, double total_time, int vis_step, const std::string& caption)
{
    const mfem::Vector& normal_flux = flow_sol.GetBlock(0);
    auto& vertex_edge = spe10problem.GetVertexEdge();
    auto edge_d_td = spe10problem.GetEdgeToTrueEdge();

    auto Adv = DiscreteAdvection(normal_flux, vertex_edge, *edge_d_td);
    mfem::SparseMatrix M = SparseIdentity(Adv->Height());
    M *= spe10problem.CellVolume();

    mfem::Vector influx(spe10problem.GetVertexRHS());
    influx *= -1.0;
    mfem::Vector S = influx;
    S = 0.0;

    mfem::socketstream sout;
    if (vis_step)
    {
        spe10problem.VisSetup(sout, S, 0.0, 1.0, caption);
    }

    double time = 0.0;

    mfem::StopWatch chrono;
    chrono.Start();

    FV_Evolution adv(M, *Adv, influx);
    adv.SetTime(time);

    mfem::ForwardEulerSolver ode_solver;
    ode_solver.Init(adv);

    int myid;
    MPI_Comm_rank(edge_d_td->GetComm(), &myid);
    bool done = false;
    for (int ti = 0; !done; )
    {
       double dt_real = std::min(delta_t, total_time - time);
       ode_solver.Step(S, time, dt_real);
       ti++;

       done = (time >= total_time - 1e-8*delta_t);

       if (vis_step && (done || ti % vis_step == 0))
       {
          if (myid == 0)
          {
             std::cout << "time step: " << ti << ", time: " << time << "\r";//std::endl;
          }
          spe10problem.VisUpdate(sout, S);
       }
    }
    if (myid == 0)
    {
       std::cout << "Time stepping done in " << chrono.RealTime() << "s.\n";
    }

    return S;
}

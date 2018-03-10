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
#include "spe10.hpp"

#include "../src/picojson.h"
#include "../src/smoothG.hpp"

using namespace smoothg;

/** A time-dependent operator for the right-hand side of the ODE. The DG weak
    form of du/dt = -v.grad(u) is M du/dt = K u + b, where M and K are the mass
    and advection matrices, and b describes the flow on the boundary. This can
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

mfem::HypreParMatrix* DiscreteAdvection(const mfem::Vector& normal_flux,
                                        const mfem::SparseMatrix& elem_facet,
                                        const mfem::HypreParMatrix& facet_truefacet);

void VisSetup(MPI_Comm comm, mfem::socketstream& sout, mfem::ParMesh& pmesh,
              mfem::ParGridFunction& saturation, bool& visualization);

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
    double spect_tol = 1.e-3;
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
    args.AddOption(&dual_target, "-dt", "--dual-target", "-no-dt",
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

    mfem::Array<int> coarseningFactor(nDimensions);
    coarseningFactor[0] = 10;
    coarseningFactor[1] = 10;
    if (nDimensions == 3)
        coarseningFactor[2] = 5;

    int nbdr;
    if (nDimensions == 3)
        nbdr = 6;
    else
        nbdr = 4;
    mfem::Array<int> ess_zeros(nbdr);
    ess_zeros = 1;

    mfem::Array<int> ess_attr;
    mfem::Vector weight;
    mfem::Vector rhs_u_fine;

    // Setting up finite volume discretization problem
    SPE10Problem spe10problem(permFile, nDimensions, spe10_scale, slice,
                              metis_agglomeration, coarseningFactor);
    mfem::ParMesh* pmesh = spe10problem.GetParMesh();

    ess_attr.SetSize(nbdr);
    for (int i(0); i < nbdr; ++i)
        ess_attr[i] = ess_zeros[i];

    // Construct "finite volume mass" matrix using mfem instead of parelag
    mfem::RT_FECollection sigmafec(0, nDimensions);
    mfem::ParFiniteElementSpace sigmafespace(pmesh, &sigmafec);

    mfem::ParBilinearForm a(&sigmafespace);
    a.AddDomainIntegrator(
        new FiniteVolumeMassIntegrator(*spe10problem.GetKInv()) );
    a.Assemble();
    a.Finalize();
    a.SpMat().GetDiag(weight);

    for (int i = 0; i < weight.Size(); ++i)
    {
        weight[i] = 1.0 / weight[i];
    }

    mfem::L2_FECollection ufec(0, nDimensions);
    mfem::ParFiniteElementSpace ufespace(pmesh, &ufec);

    mfem::LinearForm q(&ufespace);
    q.AddDomainIntegrator(new mfem::DomainLFIntegrator(*spe10problem.GetForceCoeff()));
    q.Assemble();
    rhs_u_fine = q;

    // Construct vertex_edge table in mfem::SparseMatrix format
    const mfem::Table& ve_table = (nDimensions == 2) ?
                pmesh->ElementToEdgeTable() : pmesh->ElementToFaceTable();
    mfem::SparseMatrix vertex_edge = TableToMatrix(ve_table);

    // Construct agglomerated topology based on METIS or Cartesion aggloemration
    mfem::Array<int> partitioning;
    if (metis_agglomeration)
    {
        MetisPart(partitioning, sigmafespace, ufespace, coarseningFactor);
    }
    else
    {
        auto num_procs_xyz = spe10problem.GetNumProcsXYZ();
        CartPart(partitioning, num_procs_xyz, *pmesh, coarseningFactor);
    }

    const auto& edge_d_td(sigmafespace.Dof_TrueDof_Matrix());
    auto edge_boundary_att = GenerateBoundaryAttributeTable(pmesh);

    // Create Upscaler and Solve
    FiniteVolumeUpscale fvupscale(comm, vertex_edge, weight, partitioning, *edge_d_td,
                                  edge_boundary_att, ess_attr, spect_tol, max_evects,
                                  dual_target, scaled_dual, energy_dual, hybridization);

    mfem::Array<int> marker(fvupscale.GetFineMatrix().getD().Width());
    marker = 0;
    sigmafespace.GetEssentialVDofs(ess_attr, marker);
    fvupscale.MakeFineSolver(marker);

    fvupscale.PrintInfo();
    fvupscale.ShowSetupTime();

    mfem::BlockVector rhs_fine(fvupscale.GetFineBlockVector());
    rhs_fine.GetBlock(0) = 0.0;
    rhs_fine.GetBlock(1) = rhs_u_fine;

    auto sol_fine = fvupscale.SolveFine(rhs_fine);
    fvupscale.ShowFineSolveInfo();

//    auto sol_upscaled = fvupscale.Solve(rhs_fine);
//    fvupscale.ShowCoarseSolveInfo();

//    auto error_info = fvupscale.ComputeErrors(sol_upscaled, sol_fine);

//    if (myid == 0)
//    {
//        ShowErrors(error_info);
//    }

    std::unique_ptr<mfem::HypreParMatrix> Adv(
                DiscreteAdvection(sol_fine.GetBlock(0), vertex_edge, *edge_d_td));

    mfem::SparseMatrix M = SparseIdentity(Adv->Height());
    M *= spe10problem.CellVolume(nDimensions);

    mfem::Vector influx(rhs_u_fine);
    for (int i = 0; i < influx.Size(); i++)
    {
        if (influx[i] < 0.0)
        {
            influx[i] = 0.0;
        }
    }

    mfem::Vector S = influx;
    S = 0.0;

    mfem::ParGridFunction saturation(&ufespace);    
    saturation = S;

    mfem::socketstream sout;
    VisSetup(comm, sout, *pmesh, saturation, visualization);


    double time = 0.0;
    double total_time = 1000.0;
    double delta_t = 1.0;
    int vis_steps = 10;

    mfem::StopWatch chrono;
    chrono.Start();

    FV_Evolution adv(M, *Adv, influx);
    adv.SetTime(time);

    mfem::ForwardEulerSolver ode_solver;
    ode_solver.Init(adv);

    bool done = false;
    for (int ti = 0; !done; )
    {
       double dt_real = std::min(delta_t, total_time - time);
       ode_solver.Step(S, time, dt_real);
       ti++;

       done = (time >= total_time - 1e-8*delta_t);

       if (done || ti % vis_steps == 0)
       {
          if (myid == 0)
          {
             std::cout << "time step: " << ti << ", time: " << time << "\r";//std::endl;
          }

          // 11. Extract the parallel grid function corresponding to the finite
          //     element approximation U (the local solution on each processor).
          saturation = S;

          if (visualization)
          {
             sout << "parallel " << num_procs << " " << myid << "\n";
             sout << "solution\n" << *pmesh << saturation << std::flush;
          }
       }
    }
    if (myid == 0)
    {
       std::cout << "Final time " << time << " reached, time stepping finished.\n";
    }

    return EXIT_SUCCESS;
}

// Implementation of class FE_Evolution
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
   // y = M^{-1} (K x + b)
   K_.Mult(x, y);
   y += b_;
   RescaleVector(Minv_, y);
}

mfem::HypreParMatrix* DiscreteAdvection(const mfem::Vector& normal_flux,
                                        const mfem::SparseMatrix& elem_facet,
                                        const mfem::HypreParMatrix& facet_truefacet)
{
    MPI_Comm comm = facet_truefacet.GetComm();
    const int num_elems_diag = elem_facet.Height();
    const int num_facets = elem_facet.Width();

    mfem::SparseMatrix f_tf_diag, f_tf_offd;
    HYPRE_Int * tf_map;
    facet_truefacet.GetDiag(f_tf_diag);
    facet_truefacet.GetOffd(f_tf_offd, tf_map);

    mfem::Array<int> elem_starts;
    GenerateOffsets(comm, num_elems_diag, elem_starts);

    using ParMatPtr = std::unique_ptr<mfem::HypreParMatrix>;
    ParMatPtr elem_truefacet(facet_truefacet.LeftDiagMult(elem_facet, elem_starts));
    ParMatPtr truefacet_elem(elem_truefacet->Transpose());

    mfem::SparseMatrix tf_e_diag, tf_e_offd;
    HYPRE_Int* copy_map;
    truefacet_elem->GetDiag(tf_e_diag);
    truefacet_elem->GetOffd(tf_e_offd, copy_map);
    HYPRE_Int* elem_map = new HYPRE_Int[tf_e_offd.Width()];
    std::copy_n(copy_map, tf_e_offd.Width(), elem_map);

    mfem::SparseMatrix diag(num_elems_diag, num_elems_diag);
    mfem::SparseMatrix offd(num_elems_diag, tf_e_offd.Width());

    for (int i = 0; i < num_facets; i++)
    {
        double normal_flux_i = normal_flux(i);

        if (f_tf_diag.RowSize(i) != 0) // facet is owned by local processor
        {
            assert(f_tf_diag.RowSize(i) == 1);
            int truefacet = f_tf_diag.GetRowColumns(i)[0];

            if (tf_e_offd.RowSize(truefacet) == 1) // facet is shared
            {
                assert(tf_e_diag.RowSize(truefacet) == 1);
                int diag_elem = tf_e_diag.GetRowColumns(truefacet)[0];
                int offd_elem = tf_e_offd.GetRowColumns(truefacet)[0];

                if (normal_flux_i > 0)
                {
                    offd.Set(diag_elem, offd_elem, normal_flux_i);
                }
                else
                {
                    diag.Add(diag_elem, diag_elem, normal_flux_i);
                }
            }
            else if (tf_e_diag.RowSize(truefacet) == 2) // facet is interior
            {
                const int* elem_pair = tf_e_diag.GetRowColumns(truefacet);

                if (normal_flux_i > 0)
                {
                    diag.Set(elem_pair[0], elem_pair[1], normal_flux_i);
                    diag.Add(elem_pair[1], elem_pair[1], -1.0 * normal_flux_i);
                }
                else
                {
                    diag.Set(elem_pair[1], elem_pair[0], -1.0 * normal_flux_i);
                    diag.Add(elem_pair[0], elem_pair[0], normal_flux_i);
                }
            }
            else // global boundary
            {
                assert(tf_e_diag.RowSize(truefacet) == 1);
                const int elem0 = tf_e_diag.GetRowColumns(truefacet)[0];
                if (normal_flux_i < 0)
                {
                    diag.Add(elem0, elem0, normal_flux_i);
                }
            }
        }
        // else? I think we need to do something when the facet is not owned,
        // although I dont know how at the moment, and the parallel run seems
        // match with serial run
    }

    diag.Finalize(0);
    offd.Finalize(0);

    int num_elems = elem_starts.Last();
    auto out = new mfem::HypreParMatrix(comm, num_elems, num_elems, elem_starts,
                                        elem_starts, &diag, &offd, elem_map);

    // Adjust ownership and copy starts arrays
    out->CopyRowStarts();
    out->CopyColStarts();
    out->SetOwnerFlags(3, 3, 1);

    diag.LoseData();
    offd.LoseData();

    return out;
}

void VisSetup(MPI_Comm comm, mfem::socketstream& sout, mfem::ParMesh& pmesh,
              mfem::ParGridFunction& saturation, bool& visualization)
{
   char vishost[] = "localhost";
   int  visport   = 19916;
   sout.open(vishost, visport);
   if (!sout)
   {
      if (pmesh.GetNRanks() == 0)
         std::cout << "Unable to connect to GLVis server at "
              << vishost << ':' << visport << std::endl;
      visualization = false;
      if (pmesh.GetNRanks() == 0)
      {
         std::cout << "GLVis visualization disabled.\n";
      }
   }
   else
   {
      sout << "parallel " << pmesh.GetNRanks() << " " << pmesh.GetMyRank() << "\n";
      sout.precision(8);
      sout << "solution\n" << pmesh << saturation;
      sout << "window_size 500 800\n";
      sout << "window_title 'Saturation'\n";
      sout << "autoscale off\n";
      sout << "valuerange " << 0.0 << " " << 1.0 << "\n";

      if (pmesh.SpaceDimension() == 2)
      {
          sout << "view 0 0\n"; // view from top
          sout << "keys jl\n";  // turn off perspective and light
          sout << "keys ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]\n";  // increase size
      }
      else
      {
          sout << "keys ]]]]]]]]]]]]]\n";  // increase size
      }
      sout << "keys c\n";         // show colorbar and mesh
   }
}

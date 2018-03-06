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
class FE_Evolution : public mfem::TimeDependentOperator
{
private:
   mfem::HypreParMatrix &M, &K;
   const mfem::Vector &b;
   mfem::HypreSmoother M_prec;
   mfem::CGSolver M_solver;

   mutable mfem::Vector z;

public:
   FE_Evolution(mfem::HypreParMatrix &_M, mfem::HypreParMatrix &_K, const mfem::Vector &_b);

   virtual void Mult(const mfem::Vector &x, mfem::Vector &y) const;

   virtual ~FE_Evolution() { }
};

void MetisPart(mfem::Array<int>& partitioning,
               mfem::ParFiniteElementSpace& sigmafespace,
               mfem::ParFiniteElementSpace& ufespace,
               mfem::Array<int>& coarsening_factor);

void CartPart(mfem::Array<int>& partitioning, std::vector<int>& num_procs_xyz,
              mfem::ParMesh& pmesh, mfem::Array<int>& coarsening_factor);

mfem::SparseMatrix DiscreteAdvection(const mfem::Vector& normal_flux,
                                     const mfem::SparseMatrix& elem_facet);

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
    mfem::Array<int> nat_one(nbdr);
    mfem::Array<int> nat_zeros(nbdr);
    ess_zeros = 1;
    nat_one = 0;
    nat_zeros = 0;

    mfem::Array<int> ess_attr;
    mfem::Vector weight;
    mfem::Vector rhs_u_fine;

    // Setting up finite volume discretization problem
    SPE10Problem spe10problem(permFile, nDimensions, spe10_scale, slice,
                              metis_agglomeration, coarseningFactor);

    mfem::ParMesh* pmesh = spe10problem.GetParMesh();

    if (myid == 0)
    {
        std::cout << pmesh->GetNEdges() << " fine edges, " <<
                  pmesh->GetNFaces() << " fine faces, " <<
                  pmesh->GetNE() << " fine elements\n";
    }

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
    q.AddDomainIntegrator(
        new mfem::DomainLFIntegrator(*spe10problem.GetForceCoeff()) );
    q.Assemble();
    rhs_u_fine = q;

    // Construct vertex_edge table in mfem::SparseMatrix format
//    mfem::SparseMatrix vertex_edge;
//    if (nDimensions == 2)
//    {
//        mfem::SparseMatrix tmp = TableToSparse(pmesh->ElementToEdgeTable());
//        vertex_edge.Swap(tmp);
//    }
//    else
//    {
//        mfem::SparseMatrix tmp = TableToSparse(pmesh->ElementToFaceTable());
//        vertex_edge.Swap(tmp);
//    }

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

    mfem::ParMixedBilinearForm *bVarf(new mfem::ParMixedBilinearForm(&sigmafespace, &ufespace));
    bVarf->AddDomainIntegrator(new mfem::VectorFEDivergenceIntegrator);
    bVarf->Assemble();
    bVarf->Finalize();
    mfem::SparseMatrix& vertex_edge = bVarf->SpMat();

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

    mfem::GridFunction flux_gf(&sigmafespace, sol_fine.GetBlock(0).GetData());
    mfem::VectorGridFunctionCoefficient flux_coeff(&flux_gf);
    mfem::ParBilinearForm advection(&ufespace);
    advection.AddInteriorFaceIntegrator(
        new mfem::DGTraceIntegrator(flux_coeff, 1.0, -0.5));
    advection.Assemble(0);
    advection.Finalize(0);
    mfem::HypreParMatrix *Adv_ref = advection.ParallelAssemble();

    mfem::SparseMatrix diag;
    Adv_ref->GetDiag(diag);
//    diag.Print();

    mfem::SparseMatrix Adv_diag = DiscreteAdvection(sol_fine.GetBlock(0), vertex_edge);
//    Adv_diag.Print();

    diag.Add(-1.0, Adv_diag);
diag.Print();
    double* data = diag.GetData();
    double fnorm = 0.;
    for (int i = 0; i < diag.NumNonZeroElems(); i++)
        fnorm += (data[i] * data[i]);
    fnorm = std::sqrt(fnorm);

    std::cout<<"diag diff = "<<fnorm<<"\n";


    auto Adv = new mfem::HypreParMatrix(comm, Adv_ref->N(), Adv_ref->RowPart(), &Adv_diag);

    mfem::ParBilinearForm mass(&ufespace);
    mass.AddDomainIntegrator(new mfem::MassIntegrator);
    mass.Assemble();
    mass.Finalize();
    mfem::HypreParMatrix *M = mass.ParallelAssemble();

    mfem::Vector influx(rhs_u_fine);
    for (int i = 0; i < influx.Size(); i++)
    {
        if (influx[i] < 0.0)
        {
            influx[i] = 0.0;
        }
    }

    mfem::ParGridFunction saturation(&ufespace);
    mfem::Vector S = influx;
    S = 0.0;

    mfem::socketstream sout;
    if (visualization)
    {
       char vishost[] = "localhost";
       int  visport   = 19916;
       sout.open(vishost, visport);
       if (!sout)
       {
          if (myid == 0)
             std::cout << "Unable to connect to GLVis server at "
                  << vishost << ':' << visport << std::endl;
          visualization = false;
          if (myid == 0)
          {
             std::cout << "GLVis visualization disabled.\n";
          }
       }
       else
       {
          saturation = S;
          sout << "parallel " << num_procs << " " << myid << "\n";
          sout.precision(8);
          sout << "solution\n" << *pmesh << saturation;
          sout << "window_size 500 800\n";
          sout << "window_title 'Saturation'\n";
          sout << "autoscale off\n";
          sout << "valuerange " << 0.0 << " " << 1.0 << "\n";

          if (pmesh->SpaceDimension() == 2)
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

//          sout << "pause\n";
          sout << std::flush;
          if (myid == 0)
             std::cout << "GLVis visualization paused."
                  << " Press space (in the GLVis window) to resume it.\n";
       }
    }

    double time = 0.0;
    double total_time = 1000.0;
    double delta_t = 1.0;
    int vis_steps = 10;

    mfem::StopWatch chrono;
    chrono.Start();

    FE_Evolution adv(*M, *Adv, influx);
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

    return EXIT_SUCCESS;
}

// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(mfem::HypreParMatrix &_M, mfem::HypreParMatrix &_K,
                           const mfem::Vector &_b)
   : mfem::TimeDependentOperator(_M.Height()),
     M(_M), K(_K), b(_b), M_solver(M.GetComm()), z(_M.Height())
{
   M_prec.SetType(mfem::HypreSmoother::Jacobi);
   M_solver.SetPreconditioner(M_prec);
   M_solver.SetOperator(M);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-9);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(100);
   M_solver.SetPrintLevel(0);
}

void FE_Evolution::Mult(const mfem::Vector &x, mfem::Vector &y) const
{
   // y = M^{-1} (K x + b)
   K.Mult(x, z);
   z += b;
   M_solver.Mult(z, y);
}


void MetisPart(mfem::Array<int>& partitioning,
               mfem::ParFiniteElementSpace& sigmafespace,
               mfem::ParFiniteElementSpace& ufespace,
               mfem::Array<int>& coarsening_factor)
{
    mfem::DiscreteLinearOperator DivOp(&sigmafespace, &ufespace);
    DivOp.AddDomainInterpolator(new mfem::DivergenceInterpolator);
    DivOp.Assemble();
    DivOp.Finalize();

    const mfem::SparseMatrix& DivMat = DivOp.SpMat();
    const mfem::SparseMatrix DivMatT = smoothg::Transpose(DivMat);
    const mfem::SparseMatrix vertex_vertex = smoothg::Mult(DivMat, DivMatT);

    int metis_coarsening_factor = 1;
    for (const auto factor : coarsening_factor)
        metis_coarsening_factor *= factor;

    const int nvertices = vertex_vertex.Height();
    int num_partitions = std::max(1, nvertices / metis_coarsening_factor);

    Partition(vertex_vertex, partitioning, num_partitions);
}

void CartPart(mfem::Array<int>& partitioning, std::vector<int>& num_procs_xyz,
              mfem::ParMesh& pmesh, mfem::Array<int>& coarsening_factor)
{
    const int nDimensions = num_procs_xyz.size();

    mfem::Array<int> nxyz(nDimensions);
    nxyz[0] = 60 / num_procs_xyz[0] / coarsening_factor[0];
    nxyz[1] = 220 / num_procs_xyz[1] / coarsening_factor[1];
    if (nDimensions == 3)
        nxyz[2] = 85 / num_procs_xyz[2] / coarsening_factor[2];

    for (int& i : nxyz)
    {
        i = std::max(1, i);
    }

    mfem::Array<int> cart_part(pmesh.CartesianPartitioning(nxyz.GetData()), pmesh.GetNE());
    partitioning.Append(cart_part);

    cart_part.MakeDataOwner();
}


mfem::SparseMatrix DiscreteAdvection(const mfem::Vector& normal_flux,
                                     const mfem::SparseMatrix& elem_facet)
{
    const int num_elems = elem_facet.Height();
    const int num_facets = elem_facet.Width();
    mfem::SparseMatrix out(num_elems, num_elems);

    mfem::SparseMatrix facet_elem = smoothg::Transpose(elem_facet);

    for (int i = 0; i < num_facets; i++)
    {
        if (facet_elem.RowSize(i) == 2) // assume v.n = 0 on boundary
        {
            const int* elem_pair = facet_elem.GetRowColumns(i);

            if (normal_flux(i) > 0)
            {
                out.Set(elem_pair[0], elem_pair[1], normal_flux(i));
                out.Add(elem_pair[1], elem_pair[1], -1.0 * normal_flux(i));
            }
            else
            {
                out.Set(elem_pair[1], elem_pair[0], -1.0 * normal_flux(i));
                out.Add(elem_pair[0], elem_pair[0], normal_flux(i));
            }
        }
    }
    out.Finalize(0);

    return out;
}

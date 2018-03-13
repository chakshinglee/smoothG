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
   @file finitevolume.cpp
   @brief This is an example for upscaling a graph Laplacian coming from a finite
   volume discretization of a simple reservior model in parallel.

   A simple way to run the example:

   mpirun -n 4 ./finitevolume
*/

#include <fstream>
#include <sstream>
#include <mpi.h>

#include "mfem.hpp"
#include "spe10.hpp"

#include "../src/picojson.h"
#include "../src/smoothG.hpp"

using namespace smoothg;

int main(int argc, char* argv[])
{
    int num_procs, myid;
    picojson::object serialize;

    // 1. Initialize MPI
    MPI_Init(&argc, &argv);
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
    int slice = 1;
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
    coarseningFactor[1] = 22;
    if (nDimensions == 3)
        coarseningFactor[2] = 5;

    int nbdr;
    if (nDimensions == 3)
        nbdr = 6;
    else
        nbdr = 4;
    mfem::Array<int> ess_zeros(nbdr);
    mfem::Array<int> nat_negative_one(nbdr);
    ess_zeros = 1;
    ess_zeros[0] = ess_zeros[2] = 0;
    nat_negative_one = 0;
    nat_negative_one[0] = 1;

    mfem::Array<int> ess_attr;
    mfem::Vector weight;
    mfem::Vector rhs_sigma_fine;

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

    mfem::ConstantCoefficient negative_one(-1.0);
    mfem::RestrictedCoefficient pinflow_coeff(negative_one, nat_negative_one);

    mfem::LinearForm g(&sigmafespace);
    g.AddBoundaryIntegrator(
        new mfem::VectorFEBoundaryFluxLFIntegrator(pinflow_coeff) );
    g.Assemble();
    rhs_sigma_fine = g;

    mfem::Vector rhs_u_fine(ufespace.GetVSize());
    rhs_u_fine = 0.0;

    // Construct vertex_edge table in mfem::SparseMatrix format
    auto& vertex_edge_table = pmesh->Dimension() == 2 ?
            pmesh->ElementToEdgeTable() : pmesh->ElementToFaceTable();
    mfem::SparseMatrix vertex_edge = TableToMatrix(vertex_edge_table);

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

    auto edge_d_td(sigmafespace.Dof_TrueDof_Matrix());

    auto edge_boundary_att = GenerateBoundaryAttributeTable(pmesh);

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
    rhs_fine.GetBlock(0) = rhs_sigma_fine;
    rhs_fine.GetBlock(1) = rhs_u_fine;

    auto sol_upscaled = fvupscale.Solve(rhs_fine);
    fvupscale.ShowCoarseSolveInfo();

    auto sol_fine = fvupscale.SolveFine(rhs_fine);
    fvupscale.ShowFineSolveInfo();

    auto error_info = fvupscale.ComputeErrors(sol_upscaled, sol_fine);
    if (myid == 0)
    {
        ShowErrors(error_info);
    }

    double QoI = mfem::InnerProduct(comm, sol_fine.GetBlock(0), rhs_sigma_fine);
    if (myid == 0)
    {
        std::cout<< "Fine level QoI = " << QoI << "\n";
    }

    QoI = mfem::InnerProduct(comm, sol_upscaled.GetBlock(0), rhs_sigma_fine);
    if (myid == 0)
    {
        std::cout<< "Coarse level QoI = " << QoI << "\n";
    }

    // Visualize the solution
    if (visualization)
    {
        mfem::ParGridFunction field(&ufespace);

        auto Visualize = [&](const mfem::Vector & sol)
        {
            char vishost[] = "localhost";
            int  visport   = 19916;

            mfem::socketstream vis_v;
            vis_v.open(vishost, visport);
            vis_v.precision(8);

            field = sol;

            vis_v << "parallel " << pmesh->GetNRanks() << " " << pmesh->GetMyRank() << "\n";
            vis_v << "solution\n" << *pmesh << field;
            vis_v << "window_size 500 800\n";
            vis_v << "window_title 'pressure'\n";
            vis_v << "autoscale values\n";

            if (nDimensions == 2)
            {
                vis_v << "view 0 0\n"; // view from top
                vis_v << "keys ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]\n";  // increase size
            }

            vis_v << "keys cjl\n";

            MPI_Barrier(comm);
        };

        Visualize(sol_upscaled.GetBlock(1));
        Visualize(sol_fine.GetBlock(1));
    }

    return EXIT_SUCCESS;
}

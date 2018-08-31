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
   @file mlmc.cpp
   @brief This is an example for upscaling a graph Laplacian,
   where we change coefficients in the model without re-coarsening.
*/

#include <fstream>
#include <sstream>
#include <mpi.h>
#include <random>

#include "smoothG.hpp"
#include "Sampler.hpp"

using namespace smoothg;

using linalgcpp::ReadText;
using linalgcpp::WriteText;
using linalgcpp::ReadCSR;

using parlinalgcpp::LOBPCG;
using parlinalgcpp::BoomerAMG;

std::vector<int> MetisPart(const SparseMatrix& vertex_edge, int num_parts);

int main(int argc, char* argv[])
{
    // Initialize MPI
    MpiSession mpi_info(argc, argv);
    MPI_Comm comm = mpi_info.comm_;
    int myid = mpi_info.myid_;
    int num_procs = mpi_info.num_procs_;

    // program options from command line
    std::string graph_filename = "../../graphdata/fe_vertex_edge.txt";
    std::string fiedler_filename = "../../graphdata/fe_rhs.txt";
    std::string partition_filename = "../../graphdata/fe_part.txt";
    //std::string weight_filename = "../../graphdata/fe_weight_0.txt";
    std::string weight_filename = "";
    std::string w_block_filename = "";
    bool save_output = false;

    int isolate = -1;
    int max_evects = 4;
    double spect_tol = 1e-3;
    int num_partitions = 12;
    bool hybridization = false;
    bool metis_agglomeration = false;

    int initial_seed = 1;
    int num_samples = 3;
    int dimension = 2;
    double kappa = 0.001;
    double cell_volume = 200.0;
    bool coarse_sample = false;

    linalgcpp::ArgParser arg_parser(argc, argv);

    arg_parser.Parse(graph_filename, "--g", "Graph connection data.");
    arg_parser.Parse(fiedler_filename, "--f", "Fiedler vector data.");
    arg_parser.Parse(partition_filename, "--p", "Partition data.");
    arg_parser.Parse(weight_filename, "--w", "Edge weight data.");
    arg_parser.Parse(w_block_filename, "--wb", "W block data.");
    arg_parser.Parse(save_output, "--save", "Save solutions.");
    arg_parser.Parse(max_evects, "--m", "Maximum eigenvectors per aggregate.");
    arg_parser.Parse(spect_tol, "--t", "Spectral tolerance for eigenvalue problem.");
    arg_parser.Parse(num_partitions, "--np", "Number of partitions to generate.");
    arg_parser.Parse(hybridization, "--hb", "Enable hybridization.");
    arg_parser.Parse(metis_agglomeration, "--ma", "Enable Metis partitioning.");
    arg_parser.Parse(initial_seed, "--seed", "Seed for random number generator.");
    arg_parser.Parse(num_samples, "--num-samples", "Number of samples.");
    arg_parser.Parse(dimension, "--dim", "Graph Dimension");
    arg_parser.Parse(kappa, "--kappa", "Correlation length for Gaussian samples.");
    arg_parser.Parse(cell_volume, "--cell-volume", "Graph Cell volume");
    arg_parser.Parse(coarse_sample, "--coarse-sample", "Sample on the coarse level.");

    if (!arg_parser.IsGood())
    {
        ParPrint(myid, arg_parser.ShowHelp());
        ParPrint(myid, arg_parser.ShowErrors());

        return EXIT_FAILURE;
    }

    ParPrint(myid, arg_parser.ShowOptions());

    /// [Load graph from file]
    SparseMatrix vertex_edge_global = ReadCSR(graph_filename);

    const int nvertices_global = vertex_edge_global.Rows();
    const int nedges_global = vertex_edge_global.Cols();
    /// [Load graph from file]

    /// [Partitioning]
    std::vector<int> part;
    if (metis_agglomeration)
    {
        assert(num_partitions >= num_procs);
        part = MetisPart(vertex_edge_global, num_partitions);
    }
    else
    {
        part = ReadText<int>(partition_filename);
    }
    /// [Partitioning]

    /// [Load the edge weights]
    std::vector<double> weight;
    if (!weight_filename.empty())
    {
        weight = linalgcpp::ReadText(weight_filename);
    }
    else
    {
        weight = std::vector<double>(nedges_global, 1.0);
    }

    std::vector<double> one_weight(weight.size(), 1.0);

    SparseMatrix W_block = SparseIdentity(nvertices_global);
    W_block *= cell_volume * kappa * kappa;

    // Set up GraphUpscale
    /// [Upscale]
    Graph sampler_graph(comm, vertex_edge_global, part, one_weight, W_block);
    Graph graph(comm, vertex_edge_global, part, weight);

    int sampler_seed = initial_seed + myid;
    PDESampler sampler(sampler_graph, {spect_tol, max_evects, hybridization},
                       dimension, kappa, cell_volume, sampler_seed);
    GraphUpscale upscale(graph, {spect_tol, max_evects, hybridization});

    /// [Upscale]

    /// [Right Hand Side]
    BlockVector fine_rhs = upscale.GetBlockVector(0);

    fine_rhs.GetBlock(0) = 1.0;
    fine_rhs.GetBlock(1) = ReadVertexVector(graph, fiedler_filename);
    /// [Right Hand Side]

    /// [Solve]

    BlockVector fine_sol = upscale.GetBlockVector(0);
    BlockVector upscaled_sol = upscale.GetBlockVector(0);

    for (int i = 1; i <= num_samples; ++i)
    {
        ParPrint(myid, std::cout << "\n---------------------\n\n");
        ParPrint(myid, std::cout << "Sample " << i << " :\n");

        sampler.Sample(coarse_sample);

        const auto& fine_coeff = sampler.GetCoefficientFine();
        const auto& coarse_coeff = sampler.GetCoefficientCoarse();
        const auto& upscaled_coeff = sampler.GetCoefficientUpscaled();

        upscale.MakeSolver(1, coarse_coeff);
        upscale.MakeSolver(0, fine_coeff);

        fine_sol = 0.0;
        upscaled_sol = 0.0;

        upscale.Solve(1, fine_rhs, upscaled_sol);
        upscale.Solve(0, fine_rhs, fine_sol);

        if (save_output)
        {
            SaveOutput(graph, upscaled_sol.GetBlock(1), "coarse_sol_", i);
            SaveOutput(graph, fine_sol.GetBlock(1), "fine_sol_", i);
            SaveOutput(graph, upscaled_coeff, "coarse_coeff_", i);
            SaveOutput(graph, fine_coeff, "fine_coeff_", i);
        }

        upscale.ShowCoarseSolveInfo();
        upscale.ShowFineSolveInfo();

        /// [Check Error]
        upscale.ShowErrors(upscaled_sol, fine_sol);
        /// [Check Error]

    }

    ParPrint(myid, std::cout << "\n---------------------\n\n");

    /// [Solve]

    return 0;
}

std::vector<int> MetisPart(const SparseMatrix& vertex_edge, int num_parts)
{
    SparseMatrix edge_vertex = vertex_edge.Transpose();
    SparseMatrix vertex_vertex = vertex_edge.Mult(edge_vertex);

    double ubal_tol = 2.0;

    return Partition(vertex_vertex, num_parts, ubal_tol);
}

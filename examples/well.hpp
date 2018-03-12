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
   @file well.hpp
   @brief Implementation of well models.

   Build somewhat sophisticated well models, integrate them with a reservoir model
*/

#include "../src/smoothG.hpp"

using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

using namespace smoothg;

enum WellType { Injector, Producer };
enum WellDirection { X = 0, Y, Z };

class Well
{
public:
    Well(const WellType type,
         const double value,
         const std::vector<int>& cell_indices,
         const std::vector<double>& permeabilities,
         const std::vector<double>& cell_sizes,
         const double r_e = 0.28 / std::sqrt(2),
         const double r_w = 0.01,
         const double density = 1.0,
         const double viscosity = 1.0);

    Well(const WellType type,
         const double value,
         const std::vector<int>& cell_indices,
         const std::vector<double>& permeabilities,
         const std::vector<double>& cell_sizes,
         const std::vector<double>& r_e,
         const double r_w = 0.01,
         const double density = 1.0,
         const double viscosity = 1.0);

    WellType GetType() const { return type_; }
    double GetValue() const { return value_; }
    double GetNumberOfCells() const { return cell_indices_.size(); }
    const std::vector<int>& GetWellCells() const { return cell_indices_; }
    const std::vector<double>& GetWellCoeff() const { return well_coefficients_; }

private:
    void Init(const std::vector<double>& permeabilities,
              const std::vector<double>& cell_sizes,
              const std::vector<double>& r_e,
              const double r_w,
              const double density,
              const double viscosity);
    WellType type_;

    // For injectors, value_ is the total injection rate of the well
    // For producers, value_ is the bottom hole pressure
    double value_;
    WellDirection direction_;
    std::vector<int> cell_indices_; // cells that belong to this well
    std::vector<double> well_coefficients_;
};

Well::Well(const WellType type,
           const double value,
           const std::vector<int>& cell_indices,
           const std::vector<double>& permeabilities,
           const std::vector<double>& cell_sizes,
           const double r_e,
           const double r_w,
           const double density,
           const double viscosity)
    :
    type_(type),
    value_(value),
    cell_indices_(cell_indices)
{
    std::vector<double> r_e_vec(cell_indices.size(), r_e);
    Init(permeabilities, cell_sizes, r_e_vec, r_w, density, viscosity);
}

Well::Well(const WellType type,
           const double value,
           const std::vector<int>& cell_indices,
           const std::vector<double>& permeabilities,
           const std::vector<double>& cell_sizes,
           const std::vector<double>& r_e,
           const double r_w,
           const double density,
           const double viscosity)
    :
    type_(type),
    value_(value),
    cell_indices_(cell_indices)
{
    Init(permeabilities, cell_sizes, r_e, r_w, density, viscosity);
}

void Well::Init(const std::vector<double>& permeabilities,
                const std::vector<double>& cell_sizes,
                const std::vector<double>& r_e,
                const double r_w,
                const double density,
                const double viscosity)
{
    const unsigned int num_well_cells = cell_indices_.size();
    well_coefficients_.resize(num_well_cells);

    assert(permeabilities.size() == num_well_cells);
    assert(cell_sizes.size() == num_well_cells);
    assert(r_e.size() == num_well_cells);

    for (unsigned int i = 0; i < num_well_cells; ++i)
    {
        double numerator = 2 * M_PI * density * permeabilities[i] * cell_sizes[i];
        double denominator = viscosity * std::log(r_e[i] / r_w);
        well_coefficients_[i] = numerator / denominator;
    }
}

class WellManager
{
public:
    WellManager(mfem::Mesh& mesh,
                mfem::VectorCoefficient& perm_inv_coeff)
        :
        mesh_(&mesh),
        perm_inv_coeff_(&perm_inv_coeff)
    { }

    void AddWell(const WellType type,
                 const double value,
                 const std::vector<int>& cell_indices,
                 const WellDirection direction = WellDirection::Z,
                 const double r_w = 0.01,
                 const double density = 1.0,
                 const double viscosity = 1.0);

    const std::vector<Well>& GetWells() { return wells_; }

private:
    mfem::Mesh* mesh_;
    mfem::VectorCoefficient* perm_inv_coeff_;

    std::vector<Well> wells_;

    WellManager() : mesh_(nullptr), perm_inv_coeff_(nullptr) {}
};

void WellManager::AddWell(const WellType type,
                          const double value,
                          const std::vector<int>& cell_indices,
                          const WellDirection direction,
                          const double r_w,
                          const double density,
                          const double viscosity)
{
    const int nDim = mesh_->Dimension();
    assert (nDim == 3 || (nDim == 2 && direction == WellDirection::Z) );

    const int num_well_cells = cell_indices.size();
    std::vector<double> permeabilities(num_well_cells);
    std::vector<double> cell_sizes(num_well_cells);
    std::vector<double> r_e(num_well_cells);
    mfem::Vector perm_inv;

    // directions other than the direction of the well
    int perp_dir1 = (direction + 1) % 3;
    int perp_dir2 = (direction + 2) % 3;

    // Find out the size of the well cells
    mfem::Vector dir_vec(nDim), perp_dir_vec1(nDim), perp_dir_vec2(nDim);
    if (nDim == 2)
    {
        std::fill(cell_sizes.begin(), cell_sizes.end(), 2.0);
    }
    else
    {
        dir_vec = 0.0;
        //dir_vec[direction] = -1.0;
        dir_vec[direction] = 1.0;

        for (int i = 0; i < num_well_cells; i++)
            cell_sizes[i] = mesh_->GetElementSize(cell_indices[i], dir_vec);
    }

    perp_dir_vec1 = 0.0;
    perp_dir_vec1[perp_dir1] = 1.0;

    perp_dir_vec2 = 0.0;
    perp_dir_vec2[perp_dir2] = 1.0;

    const mfem::IntegrationRule* ir =
        &(mfem::IntRules.Get(mesh_->GetElementType(0), 1));
    const mfem::IntegrationPoint& ip = ir->IntPoint(0);

    // Find out the effective permeability and equivalent radius of the cells
    for (int i = 0; i < num_well_cells; i++)
    {
        auto Tr = mesh_->GetElementTransformation(cell_indices[i]);
        Tr->SetIntPoint (&ip);
        perm_inv_coeff_->Eval(perm_inv, *Tr, ip);

        permeabilities[i] = 1. / sqrt(perm_inv[perp_dir1] * perm_inv[perp_dir2]);

        double perp_h1 = mesh_->GetElementSize(cell_indices[i], perp_dir_vec1);
        double perp_h2 = mesh_->GetElementSize(cell_indices[i], perp_dir_vec2);

        double numerator = 0.28 * sqrt(sqrt(perm_inv[perp_dir1] / perm_inv[perp_dir2]) * perp_h1 * perp_h1 +
                                       sqrt(perm_inv[perp_dir2] / perm_inv[perp_dir1]) * perp_h2 * perp_h2);
        double denominator = pow(perm_inv[perp_dir1] / perm_inv[perp_dir2], 0.25) +
                             pow(perm_inv[perp_dir2] / perm_inv[perp_dir1], 0.25);
        r_e[i] = numerator / denominator;
    }

    wells_.emplace_back(type, value, cell_indices, permeabilities,
                        cell_sizes, r_e, r_w, density, viscosity);
}

mfem::SparseMatrix BuildReservoirGraph(const mfem::ParMesh& pmesh)
{
    const mfem::Table& vertex_edge_table = pmesh.Dimension() == 2 ?
                                           pmesh.ElementToEdgeTable() : pmesh.ElementToFaceTable();

    return TableToMatrix(vertex_edge_table);
}

shared_ptr<mfem::HypreParMatrix> ExtendMatrixByIdentity(
    const mfem::HypreParMatrix& pmat, const int id_size)
{
    mfem::SparseMatrix diag, offd;
    HYPRE_Int* old_colmap;
    pmat.GetDiag(diag);
    pmat.GetOffd(offd, old_colmap);

    const int nrows = diag.Height() + id_size;
    const int ncols_diag = diag.Width() + id_size;
    const int nnz_diag = diag.NumNonZeroElems() + id_size;


    mfem::Array<HYPRE_Int> row_starts, col_starts;
    mfem::Array<HYPRE_Int>* starts[2] = {&row_starts, &col_starts};
    HYPRE_Int sizes[2] = {nrows, ncols_diag};
    GenerateOffsets(pmat.GetComm(), 2, sizes, starts);


    int myid_;
    int num_procs;
    MPI_Comm comm = pmat.GetComm();
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid_);

    int col_diff = col_starts[0] - pmat.ColPart()[0];

    int global_true_dofs = pmat.N();
    mfem::Array<int> col_change(global_true_dofs);
    col_change = 0;

    int start = pmat.ColPart()[0];
    int end = pmat.ColPart()[1];

    for (int i = start; i < end; ++i)
    {
        col_change[i] = col_diff;
    }

    mfem::Array<int> col_remap(global_true_dofs);
    col_remap = 0;

    MPI_Scan(col_change.GetData(), col_remap.GetData(), global_true_dofs, HYPRE_MPI_INT, MPI_SUM, comm);
    MPI_Bcast(col_remap.GetData(), global_true_dofs, HYPRE_MPI_INT, num_procs - 1, comm);

    // Append identity matrix to the bottom left of diag
    int* diag_i = new int[nrows + 1];
    std::copy_n(diag.GetI(), diag.Height() + 1, diag_i);
    std::iota(diag_i + diag.Height(), diag_i + nrows + 1, diag_i[diag.Height()]);

    int* diag_j = new int[nnz_diag];
    std::copy_n(diag.GetJ(), diag.NumNonZeroElems(), diag_j);

    for (int i = 0; i < id_size; i++)
    {
        diag_j[diag.NumNonZeroElems() + i] = diag.Width() + i;
    }

    double* diag_data = new double[nnz_diag];
    std::copy_n(diag.GetData(), diag.NumNonZeroElems(), diag_data);
    std::fill_n(diag_data + diag.NumNonZeroElems(), id_size, 1.0);

    // Append zero matrix to the bottom of offd
    const int ncols_offd = offd.Width();
    const int nnz_offd = offd.NumNonZeroElems() + id_size;

    int* offd_i = new int[nrows + 1];
    std::copy_n(offd.GetI(), offd.Height() + 1, offd_i);
    std::fill_n(offd_i + offd.Height() + 1, id_size, offd_i[offd.Height()]);

    int* offd_j = new int[nnz_offd];
    std::copy_n(offd.GetJ(), offd.NumNonZeroElems(), offd_j);

    double* offd_data = new double[nnz_offd];
    std::copy_n(offd.GetData(), offd.NumNonZeroElems(), offd_data);

    HYPRE_Int* colmap = new HYPRE_Int[ncols_offd]();
    std::copy_n(old_colmap, ncols_offd, colmap);

    for (int i = 0; i < ncols_offd; ++i)
    {
        colmap[i] += col_remap[colmap[i]];
    }

    auto out = make_shared<mfem::HypreParMatrix>(
                   pmat.GetComm(), row_starts.Last(), col_starts.Last(),
                   row_starts, col_starts, diag_i, diag_j, diag_data,
                   offd_i, offd_j, offd_data, ncols_offd, colmap);

    out->CopyRowStarts();
    out->CopyColStarts();

    return out;
}

shared_ptr<mfem::HypreParMatrix> IntegrateReservoirAndWellModels(
    const std::vector<Well>& well_list,
    mfem::SparseMatrix& vertex_edge, mfem::Vector& weight,
    mfem::HypreParMatrix& edge_d_td, mfem::Vector& rhs_sigma, mfem::Vector& rhs_u)
{
    int num_well_cells = 0;
    for (const auto& well : well_list)
        num_well_cells += well.GetNumberOfCells();

    int num_injectors = 0;
    for (const auto& well : well_list)
    {
        num_injectors += (well.GetType() == WellType::Injector);
    }

    mfem::SparseMatrix edge_vertex(smoothg::Transpose(vertex_edge));

    const int num_reservoir_cells = edge_vertex.Width();
    const int num_reservoir_faces = edge_vertex.Height();
    const int new_nedges = num_reservoir_faces + num_well_cells;
    const int new_nvertices = num_reservoir_cells + num_injectors;
    const int edge_vertex_nnz = edge_vertex.NumNonZeroElems();
    const int new_edge_vertex_nnz = edge_vertex_nnz + num_well_cells * 2; // this is over estimated

    // Copying the old data
    mfem::Vector new_weight(new_nedges);
    std::copy_n(weight.GetData(), weight.Size(), new_weight.GetData());
    mfem::Vector new_rhs_sigma(new_nedges);
    std::copy_n(rhs_sigma.GetData(), rhs_sigma.Size(), new_rhs_sigma.GetData());
    std::fill_n(new_rhs_sigma.GetData() + rhs_sigma.Size(), num_well_cells, 0.0);
    mfem::Vector new_rhs_u(new_nvertices);
    std::copy_n(rhs_u.GetData(), rhs_u.Size(), new_rhs_u.GetData());

    int* new_edge_vertex_i = new int[new_nedges + 1];
    std::copy_n(edge_vertex.GetI(), num_reservoir_faces + 1, new_edge_vertex_i);
    int* new_edge_vertex_j = new int[new_edge_vertex_nnz];
    std::copy_n(edge_vertex.GetJ(), edge_vertex_nnz, new_edge_vertex_j);
    double* new_edge_vertex_data = new double[new_edge_vertex_nnz];
    std::fill_n(new_edge_vertex_data, new_edge_vertex_nnz, 1.0);

    // Adding well equations to the system
    int edge_counter = num_reservoir_faces;
    int nnz_counter = edge_vertex_nnz;
    int injector_counter = 0;
    for (unsigned int i = 0; i < well_list.size(); i++)
    {
        const auto& well_cells = well_list[i].GetWellCells();
        const auto& well_coeff = well_list[i].GetWellCoeff();

        if (well_list[i].GetType() == WellType::Producer)
        {
            for (unsigned int j = 0; j < well_cells.size(); j++)
            {
                new_edge_vertex_j[nnz_counter] = well_cells[j];
                new_weight[edge_counter] = well_coeff[j];
                new_rhs_sigma[edge_counter] = well_list[i].GetValue();
                nnz_counter++;

                new_edge_vertex_i[edge_counter + 1] = nnz_counter;
                edge_counter++;
            }
        }
        else
        {
            for (unsigned int j = 0; j < well_cells.size(); j++)
            {
                new_edge_vertex_j[nnz_counter] = well_cells[j];
                new_edge_vertex_j[nnz_counter + 1] = num_reservoir_cells + injector_counter;
                new_weight[edge_counter] = well_coeff[j];
                nnz_counter += 2;

                new_edge_vertex_i[edge_counter + 1] = nnz_counter;
                edge_counter++;
            }
            new_rhs_u[num_reservoir_cells + injector_counter] = -1.0 * well_list[i].GetValue();
            injector_counter++;
        }
    }

    mfem::SparseMatrix new_edge_vertex(new_edge_vertex_i, new_edge_vertex_j,
                                       new_edge_vertex_data,
                                       new_nedges, new_nvertices);
    //new_edge_vertex.Print();
    mfem::SparseMatrix new_vertex_edge(smoothg::Transpose(new_edge_vertex));

    vertex_edge.Swap(new_vertex_edge);
    weight.Swap(new_weight);
    rhs_sigma.Swap(new_rhs_sigma);
    rhs_u.Swap(new_rhs_u);

    return ExtendMatrixByIdentity(edge_d_td, num_well_cells);
}

void RemoveWellDofs(const std::vector<Well>& well_list,
                      const mfem::BlockVector& vec,
                      mfem::Array<int>& offset, mfem::BlockVector& new_vec)
{
    int num_well_cells = 0;
    for (const auto& well : well_list)
        num_well_cells += well.GetNumberOfCells();

    offset.SetSize(3);
    offset[0] = 0;
    offset[1] = vec.GetBlock(0).Size() - num_well_cells;
    offset[2] = offset[1] + vec.GetBlock(1).Size() - well_list.size();

    double* data = new double[offset[2]];
    new_vec.Update(data, offset);
    new_vec.MakeDataOwner();

    for (int k = 0; k < 2; k++)
        for (int i = 0; i < new_vec.BlockSize(k); i++)
            new_vec.GetBlock(k)[i] = vec.GetBlock(k)[i];
}

void MakeWellMaps(const std::vector<Well>& well_list,
                  const mfem::BlockVector& vec,
                  mfem::Array<int>& no_well_offset,
                  mfem::Array<int>& no_well_map,
                  mfem::Array<int>& well_offset,
                  mfem::Array<int>& well_map)
{
    int num_well_cells = 0;
    for (auto& well : well_list)
        num_well_cells += well.GetNumberOfCells();

    int num_injectors = 0;
    for (const auto& well : well_list)
    {
        num_injectors += (well.GetType() == WellType::Injector);
    }

    no_well_offset.SetSize(3);
    no_well_offset[0] = 0;
    no_well_offset[1] = vec.GetBlock(0).Size() - num_well_cells;
    no_well_offset[2] = no_well_offset[1] + vec.GetBlock(1).Size() - num_injectors;

    well_offset.SetSize(3);
    well_offset[0] = 0;
    well_offset[1] = num_well_cells;
    well_offset[2] = well_offset[1] + num_injectors;

    no_well_map.SetSize(no_well_offset[2]);
    std::iota(no_well_map.begin(), no_well_map.begin() + no_well_offset[1], 0);
    std::iota(no_well_map.begin() + no_well_offset[1], no_well_map.end(), vec.GetBlock(0).Size());

    well_map.SetSize(num_well_cells + num_injectors);
    std::iota(well_map.begin(), well_map.begin() + num_well_cells, no_well_offset[1]);
    std::iota(well_map.begin() + num_well_cells, well_map.end(), no_well_offset[2] + num_well_cells);
}

void WritePoints(double time, std::ofstream& output, const mfem::Vector& values)
{
    output << time;

    for (int i = 0; i < values.Size(); ++i)
    {
        output << "\t" << values[i];
    }

    output << std::endl;
}


void PartitionVerticesByMetis(
    const mfem::SparseMatrix& vertex_edge,
    const mfem::Array<int>& isolate_vertices,
    int num_partitions,
    mfem::Array<int>& partition,
    int degree = 1)
{
    mfem::SparseMatrix e_v = smoothg::Transpose(vertex_edge);
    mfem::SparseMatrix vert_vert = smoothg::Mult(vertex_edge, e_v);

    MetisGraphPartitioner partitioner;
    partitioner.setUnbalanceTol(2.0);

    mfem::SparseMatrix vert_vert_ext(vert_vert);

    for (int i = 1; i < degree; ++i)
    {
        auto tmp = smoothg::Mult(vert_vert_ext, vert_vert);
        vert_vert_ext.Swap(tmp);
    }

    mfem::Array<int> connected_vertices;
    for (auto i : isolate_vertices)
    {
        GetTableRow(vert_vert_ext, i, connected_vertices);

        for (auto connection : connected_vertices)
        {
            if (connection != i)
            {
                partitioner.SetPostIsolateVertices(connection);
            }
        }
    }

    partitioner.SetPostIsolateVertices(isolate_vertices);

    partitioner.doPartition(vert_vert, num_partitions, partition);
}

// extend edge_boundaryattr by adding empty rows corresponding to wells edges
void ExtendEdgeBoundaryattr(const std::vector<Well>& well_list,
                            mfem::SparseMatrix& edge_boundaryattr)
{
    const int old_nedges = edge_boundaryattr.Height();
    int num_well_cells = 0;
    for (auto& well : well_list)
        num_well_cells += well.GetNumberOfCells();

    int* new_i = new int[old_nedges + num_well_cells + 1];
    std::copy_n(edge_boundaryattr.GetI(), old_nedges + 1, new_i);
    std::fill_n(new_i + old_nedges + 1, num_well_cells, new_i[old_nedges]);

    mfem::SparseMatrix new_edge_boundaryattr(
        new_i, edge_boundaryattr.GetJ(), edge_boundaryattr.GetData(),
        old_nedges + num_well_cells, edge_boundaryattr.Width());

    edge_boundaryattr.Swap(new_edge_boundaryattr);
    new_edge_boundaryattr.SetGraphOwner(false);
    new_edge_boundaryattr.SetDataOwner(false);
    delete[] new_edge_boundaryattr.GetI();
}

// extend edge_boundaryattr by adding new attribute to rows corresponding to wells edges
void ExtendEdgeBoundaryattr2(const std::vector<Well>& well_list,
                             mfem::SparseMatrix& edge_boundaryattr,
                             mfem::Array<int>& well_marker)
{
    const int old_nedges = edge_boundaryattr.Height();
    const int old_nnz = edge_boundaryattr.NumNonZeroElems();

    std::vector<int> well_cells;
    std::vector<WellType> well_type;
    int num_well_cells = 0;
    int new_nnz = old_nnz;

    for (auto& well : well_list)
    {
        int num_wells_i = well.GetNumberOfCells();
        num_well_cells += num_wells_i;
        if (well.GetType() == WellType::Producer)
            new_nnz += num_wells_i;

        well_cells.push_back(num_wells_i);
        well_type.push_back(well.GetType());
    }

    int* new_i = new int[old_nedges + num_well_cells + 1];
    int* new_j = new int[new_nnz];
    double* new_data = new double[new_nnz];
    well_marker.SetSize(old_nedges + num_well_cells);
    well_marker = 0;

    std::fill_n(new_j, old_nnz, -5);
    std::copy_n(edge_boundaryattr.GetI(), old_nedges + 1, new_i);
    std::copy_n(edge_boundaryattr.GetJ(), old_nnz, new_j);
    std::copy_n(edge_boundaryattr.GetData(), old_nnz, new_data);

    //int counter = old_nedges;
    int counter = 0;
    int new_attr = edge_boundaryattr.Width();

    for (size_t i = 0; i < well_list.size(); ++i)
    {
        int num_cells = well_cells[i];

        if (well_type[i] == WellType::Producer)
        {
            for (int j = 0; j < num_cells; ++j)
            {
                new_j[old_nnz + counter + j] = new_attr;
                new_data[old_nnz + counter + j] = 1.0;
                new_i[old_nedges + counter + j + 1] = new_i[old_nedges + counter] + j + 1;
                //printf("Setting: %d, %d to %d %d\n", i, j, new_i[old_nedges + counter+ j + 1], new_i[old_nedges + counter]);
                well_marker[old_nedges + counter + j] = 1;
            }
        }
        else
        {
            for (int j = 0; j < num_cells; ++j)
            {
                new_i[old_nedges + counter + j + 1] = new_i[old_nedges + counter];
                //printf("Extending: %d, %d to %d %d\n", i, j, new_i[old_nedges + counter+ j + 1], new_i[old_nedges + counter]);
            }
        }

        counter += num_cells;
    }

//    for (int i = old_nnz; i < new_nnz; ++i)
//    {
//        printf("j %d: %d\n", i, new_j[i]);
//    }
//    for (int i = old_nedges; i < old_nedges + num_well_cells + 1; ++i)
//    {
//        printf("i %d: %d\n", i, new_i[i]);
//    }

    assert(new_i[old_nedges + num_well_cells] == new_nnz);

    mfem::SparseMatrix new_edge_boundaryattr(
        new_i, new_j, new_data,
        old_nedges + num_well_cells, edge_boundaryattr.Width() + 1);

    edge_boundaryattr.Swap(new_edge_boundaryattr);
}

void CoarsenVertexEssentialCondition(
    const int num_wells, const int new_size,
    mfem::Array<int>& ess_marker, mfem::Vector& ess_data)
{
    mfem::Array<int> new_ess_marker(new_size);
    mfem::Vector new_ess_data(new_size);
    new_ess_marker = 0;
    new_ess_data = 0.0;

    const int old_size = ess_data.Size();

    for (int i = 0; i < num_wells; i++)
    {
        if (ess_marker[old_size - 1 - i])
        {
            new_ess_marker[new_size - 1 - i] = 1;
            new_ess_data(new_size - 1 - i) = ess_data(old_size - 1 - i);
        }
    }
    mfem::Swap(ess_marker, new_ess_marker);
    ess_data.Swap(new_ess_data);
}

void CoarsenSigmaEssentialCondition(
    const int num_wells, const int new_size,
    mfem::Array<int>& ess_marker)
{
    mfem::Array<int> new_ess_marker(new_size);
    new_ess_marker = 0;

    const int old_size = ess_marker.Size();

    for (int i = 0; i < num_wells; i++)
    {
        if (ess_marker[old_size - 1 - i])
        {
            new_ess_marker[new_size - 1 - i] = 1;
        }
    }

    mfem::Swap(ess_marker, new_ess_marker);
}

class SPE10Problem
{
public:
    SPE10Problem(const char* permFile, const int nDimensions,
                 const int spe10_scale, const int slice, const mfem::Array<int>& ess_attr,
                 int nz = 15, int well_height = 5,
                 double inject_rate = 1.0, double bottom_hole_pressure = 0.0, double well_shift = 1.0);

    ~SPE10Problem()
    {
        InversePermeabilityFunction::ClearMemory();
    }
    mfem::ParMesh* GetParMesh()
    {
        return pmesh_.get();
    }
    const std::vector<int>& GetNumProcsXYZ()
    {
        return num_procs_xyz_;
    }
    static double CellVolume(int nDimensions)
    {
        return (20.0 * 10.0 * 2.0);// (nDimensions == 2 ) ? (20.0 * 10.0) : (20.0 * 10.0 * 2.0);
    }
    mfem::ParFiniteElementSpace* GetEdgeFES()
    {
        return sigmafespace_.get();
    }
    mfem::ParFiniteElementSpace* GetVertexFES()
    {
        return ufespace_.get();
    }

    mfem::SparseMatrix& GetVertexEdge() { return vertex_edge_; }
    const mfem::SparseMatrix& GetVertexEdge() const { return vertex_edge_; }

    shared_ptr<mfem::HypreParMatrix> GetEdgeToTrueEdge()
    {
        return edge_d_td_;
    }
    const mfem::Vector& GetWeight()
    {
        return weight_;
    }
    const mfem::Vector& GetEdgeRHS()
    {
        return rhs_sigma_;
    }
    const mfem::Vector& GetVertexRHS()
    {
        return rhs_u_;
    }
    const std::vector<Well>& GetWells()
    {
        return well_manager_->GetWells();
    }
    const mfem::SparseMatrix& GetEdgeBoundaryAttributeTable()
    {
        return edge_bdr_att_;
    }

    const mfem::Array<int>& GetEssentialEdgeDofsMarker()
    {
        return ess_edof_marker_;
    }
    void PrintMeshWithPartitioning(mfem::Array<int>& partition);

    void VisualizeSolution(int k, const mfem::BlockVector& interp_sol);

    void setup_five_spot_pattern(const mfem::Array<int>& N, const int nDim,
                                 WellManager& well_manager, int well_height,
                                 double injection_rate, double bottom_hole_pressure = 0.0);
    void setup_five_spot_pattern_center(const mfem::Array<int>& N, const int nDim,
                                 WellManager& well_manager, int well_height,
                                 double injection_rate, double bottom_hole_pressure = 0.0, double shift_in = 1.0);
    void setup_nine_spot_pattern(const mfem::Array<int>& N, const int nDim,
                                 WellManager& well_manager, int well_height,
                                 double injection_rate, double bottom_hole_pressure = 0.0);
    void setup_ten_spot_pattern(const mfem::Array<int>& N, const int nDim,
                                 WellManager& well_manager, int well_height,
                                 double injection_rate, double bottom_hole_pressure = 0.0);
private:
    unique_ptr<mfem::ParMesh> pmesh_;
    unique_ptr<WellManager> well_manager_;

    std::vector<int> num_procs_xyz_;
    unique_ptr<mfem::RT_FECollection> sigmafec_;
    unique_ptr<mfem::L2_FECollection> ufec_;
    unique_ptr<mfem::ParFiniteElementSpace> sigmafespace_;
    unique_ptr<mfem::ParFiniteElementSpace> ufespace_;

    mfem::SparseMatrix vertex_edge_;
    shared_ptr<mfem::HypreParMatrix> edge_d_td_;

    mfem::Vector weight_;
    mfem::Vector rhs_sigma_;
    mfem::Vector rhs_u_;

    mfem::Vector bbmin_;
    mfem::Vector bbmax_;

    mfem::SparseMatrix edge_bdr_att_;
    mfem::Array<int> ess_edof_marker_;

    int myid_;
};

SPE10Problem::SPE10Problem(const char* permFile, const int nDimensions,
                           const int spe10_scale, const int slice, const mfem::Array<int>& ess_attr, int nz,
                           int well_height, double inject_rate, double bottom_hole_pressure, double well_shift)
{
    int num_procs;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid_);

    mfem::Array<int> N(3);
    N[0] = 12 * spe10_scale; // 60
    N[1] = 44 * spe10_scale; // 220
    N[2] = 17 * spe10_scale; // 85

    // SPE10 grid cell dimensions
    mfem::Vector h(3);
    h(0) = 20.0;
    h(1) = 10.0;
    h(2) = 2.0;

    using IPF = InversePermeabilityFunction;

    IPF::SetNumberCells(N[0], N[1], N[2]);
    IPF::SetMeshSizes(h(0), h(1), h(2));
    IPF::ReadPermeabilityFile(permFile, MPI_COMM_WORLD);

    const int num_cell_z = (nDimensions == 2) ? 1 : nz;
//    IPF::z_offset_ = std::min(84, std::max(84 - slice, 0));

    auto mesh = make_unique<mfem::Mesh>(N[0], N[1], num_cell_z,
                                        mfem::Element::HEXAHEDRON, 1,
                                        h(0) * N[0], h(1) * N[1], h(2) * num_cell_z);

    pmesh_  = make_unique<mfem::ParMesh>(comm, *mesh);

    if (myid_ == 0)
    {
        std::cout << pmesh_->GetNEdges() << " fine edges, " <<
                  pmesh_->GetNFaces() << " fine faces, " <<
                  pmesh_->GetNE() << " fine elements\n";
    }

    sigmafec_ = make_unique<mfem::RT_FECollection>(0, 3);
    sigmafespace_ = make_unique<mfem::ParFiniteElementSpace>(pmesh_.get(),
                                                             sigmafec_.get());
    ufec_ = make_unique<mfem::L2_FECollection>(0, 3);
    ufespace_ = make_unique<mfem::ParFiniteElementSpace>(pmesh_.get(),
                                                         ufec_.get());

    auto kinv = make_unique<mfem::VectorFunctionCoefficient>(nDimensions, IPF::InversePermeability);

    // Construct "finite volume mass" matrix
    mfem::ParBilinearForm a(sigmafespace_.get());
    a.AddDomainIntegrator(new FiniteVolumeMassIntegrator(*kinv));
    a.Assemble();
    a.Finalize();
    a.SpMat().GetDiag(weight_);
    for (int i = 0; i < weight_.Size(); ++i)
    {
        assert(mfem::IsFinite(weight_[i]) && weight_[i] != 0.0);
        weight_[i] = 1.0 / weight_[i];
    }

    auto tmp = BuildReservoirGraph(*pmesh_);
    vertex_edge_.Swap(tmp);

    rhs_sigma_.SetSize(vertex_edge_.Width());
    rhs_sigma_ = 0.0;
    rhs_u_.SetSize(vertex_edge_.Height());
    rhs_u_ = 0.0;

    mesh->GetBoundingBox(bbmin_, bbmax_, 1);


    // Build wells (Peaceman's five-spot pattern)
    well_manager_ = make_unique<WellManager>(*pmesh_, *kinv);
    //setup_five_spot_pattern(N, nDimensions, *well_manager_, well_height, inject_rate,
    //setup_nine_spot_pattern(N, nDimensions, *well_manager_, well_height, inject_rate,
    //setup_ten_spot_pattern(N, nDimensions, *well_manager_, well_height, inject_rate,
    setup_five_spot_pattern(N, nDimensions, *well_manager_, well_height, inject_rate,
                                   bottom_hole_pressure);

    edge_d_td_ = IntegrateReservoirAndWellModels(
                     well_manager_->GetWells(), vertex_edge_, weight_,
                     *(sigmafespace_->Dof_TrueDof_Matrix()), rhs_sigma_, rhs_u_);

    auto edge_bdr_att_tmp = GenerateBoundaryAttributeTable(pmesh_.get());
    edge_bdr_att_.Swap(edge_bdr_att_tmp);
    sigmafespace_->GetEssentialVDofs(ess_attr, ess_edof_marker_);

    mfem::Array<int> well_marker;
    ExtendEdgeBoundaryattr2(well_manager_->GetWells(), edge_bdr_att_, well_marker);

    well_marker = 0;
    std::copy_n(ess_edof_marker_.GetData(), ess_edof_marker_.Size(), well_marker.GetData());
    mfem::Swap(ess_edof_marker_, well_marker);
}

void SPE10Problem::setup_nine_spot_pattern(const mfem::Array<int>& N, const int nDim,
                                           WellManager& well_manager, int well_height,
                                           double injection_rate, double bottom_hole_pressure)
{
    if (nDim == 2)
    {
        well_height = 1;
    }
    else
    {
        well_height = std::min(N[2], well_height);
    }

    const int num_wells = 9;

    std::vector<std::vector<int>> producer_well_cells(num_wells - 1);
    std::vector<std::vector<int>>injector_well_cells(1);

    mfem::DenseMatrix point(3, num_wells);
    // Producers
    // Bottom 4
    point(0, 0) = bbmin_[0] + 1.0;
    point(1, 0) = bbmin_[1] + 1.0;
    point(2, 0) = bbmin_[2] + 1.0;

    point(0, 1) = bbmax_[0] - 1.0;
    point(1, 1) = bbmin_[1] + 1.0;
    point(2, 1) = bbmin_[2] + 1.0;

    point(0, 2) = bbmin_[0] + 1.0;
    point(1, 2) = bbmax_[1] - 1.0;
    point(2, 2) = bbmin_[2] + 1.0;

    point(0, 3) = bbmax_[0] - 1.0;
    point(1, 3) = bbmax_[1] - 1.0;
    point(2, 3) = bbmin_[2] + 1.0;

    // Top 4
    point(0, 4) = bbmin_[0] + 1.0;
    point(1, 4) = bbmin_[1] + 1.0;
    point(2, 4) = bbmax_[2] - 1.0;

    point(0, 5) = bbmax_[0] - 1.0;
    point(1, 5) = bbmin_[1] + 1.0;
    point(2, 5) = bbmax_[2] - 1.0;

    point(0, 6) = bbmin_[0] + 1.0;
    point(1, 6) = bbmax_[1] - 1.0;
    point(2, 6) = bbmax_[2] - 1.0;

    point(0, 7) = bbmax_[0] - 1.0;
    point(1, 7) = bbmax_[1] - 1.0;
    point(2, 7) = bbmax_[2] - 1.0;

    // Injector, Shifted to avoid middle,
    // Since probably processor boundary
    point(0, 8) = ((bbmax_[0] - bbmin_[0]) / 2.0) + 1.0;
    point(1, 8) = ((bbmax_[1] - bbmin_[1]) / 2.0) + 1.0;
    point(2, 8) = std::min(((bbmax_[2] - bbmin_[2]) / 2.0) + (well_height / 2.0), bbmax_[2]);


    for (int j = 0; j < well_height; ++j)
    {

        mfem::Array<int> ids;
        mfem::Array<mfem::IntegrationPoint> ips;

        pmesh_->FindPoints(point, ids, ips, false);

        // Producers
        for (int i = 0; i < num_wells - 1; ++i)
        {
            if (ids[i] >= 0)
            {
                producer_well_cells[i].push_back(ids[i]);
            }
        }

        // Injector
        if (ids[num_wells - 1] >= 0)
        {
            injector_well_cells[0].push_back(ids[num_wells - 1]);
        }


        // Shift Points for next layer
        for (int i = 0; i < 4; ++i)
        {
            point(2, i) += 2.0;
        }

        for (int i = 4; i < 8; ++i)
        {
            point(2, i) -= 2.0;
        }
        for (int i = 8; i < 9; ++i)
        {
            point(2, i) -= 2.0;
        }
    }

    for (const auto& cells : producer_well_cells)
    {
        if (cells.size() > 0)
        {
            well_manager.AddWell(WellType::Producer, bottom_hole_pressure, cells);
        }
    }

    for (const auto& cells : injector_well_cells)
    {
        if (cells.size() > 0)
        {
            well_manager.AddWell(WellType::Injector, injection_rate, cells);
        }
    }
}

void SPE10Problem::setup_ten_spot_pattern(const mfem::Array<int>& N, const int nDim,
                                           WellManager& well_manager, int well_height,
                                           double injection_rate, double bottom_hole_pressure)
{
    if (nDim == 2)
    {
        well_height = 1;
    }
    else
    {
        well_height = std::min(N[2], well_height);
    }

    const int num_wells = 10;

    std::vector<std::vector<int>> producer_well_cells(num_wells - 2);
    std::vector<std::vector<int>> injector_well_cells(2);

    mfem::DenseMatrix point(3, num_wells);
    // Producers
    // Bottom 4
    point(0, 0) = bbmin_[0] + 1.0;
    point(1, 0) = bbmin_[1] + 1.0;
    point(2, 0) = bbmin_[2] + 1.0;

    point(0, 1) = bbmax_[0] - 1.0;
    point(1, 1) = bbmin_[1] + 1.0;
    point(2, 1) = bbmin_[2] + 1.0;

    point(0, 2) = bbmin_[0] + 1.0;
    point(1, 2) = bbmax_[1] - 1.0;
    point(2, 2) = bbmin_[2] + 1.0;

    point(0, 3) = bbmax_[0] - 1.0;
    point(1, 3) = bbmax_[1] - 1.0;
    point(2, 3) = bbmin_[2] + 1.0;

    // Top 4
    point(0, 4) = bbmin_[0] + 1.0;
    point(1, 4) = bbmin_[1] + 1.0;
    point(2, 4) = bbmax_[2] - 1.0;

    point(0, 5) = bbmax_[0] - 1.0;
    point(1, 5) = bbmin_[1] + 1.0;
    point(2, 5) = bbmax_[2] - 1.0;

    point(0, 6) = bbmin_[0] + 1.0;
    point(1, 6) = bbmax_[1] - 1.0;
    point(2, 6) = bbmax_[2] - 1.0;

    point(0, 7) = bbmax_[0] - 1.0;
    point(1, 7) = bbmax_[1] - 1.0;
    point(2, 7) = bbmax_[2] - 1.0;

    // Injector, Shifted to avoid middle,
    // Since probably processor boundary
    point(0, 8) = ((bbmax_[0] - bbmin_[0]) / 2.0) + 1.0;
    point(1, 8) = (1.0 * (bbmax_[1]) / 3.0) - 1.0;
    point(2, 8) = std::min(((bbmax_[2] - bbmin_[2]) / 2.0) + (well_height / 2.0), bbmax_[2]);

    point(0, 9) = ((bbmax_[0] - bbmin_[0]) / 2.0) + 1.0;
    point(1, 9) = (2.0 * (bbmax_[1]) / 3.0) - 1.0;
    point(2, 9) = std::min(((bbmax_[2] - bbmin_[2]) / 2.0) + (well_height / 2.0), bbmax_[2]);


    for (int j = 0; j < well_height; ++j)
    {

        mfem::Array<int> ids;
        mfem::Array<mfem::IntegrationPoint> ips;

        pmesh_->FindPoints(point, ids, ips, false);

        // Producers
        for (int i = 0; i < num_wells - 2; ++i)
        {
            if (ids[i] >= 0)
            {
                producer_well_cells[i].push_back(ids[i]);
            }
        }

        // Injector
        for (int i = 0; i < 2; ++i)
        {
            if (ids[num_wells - i - 1] >= 0)
            {
                injector_well_cells[i].push_back(ids[num_wells - i - 1]);
            }
        }


        // Shift Points for next layer
        for (int i = 0; i < 4; ++i)
        {
            point(2, i) += 2.0;
        }

        for (int i = 4; i < 8; ++i)
        {
            point(2, i) -= 2.0;
        }
        for (int i = 8; i < 10; ++i)
        {
            point(2, i) -= 2.0;
        }
    }

    for (const auto& cells : producer_well_cells)
    {
        if (cells.size() > 0)
        {
            well_manager.AddWell(WellType::Producer, bottom_hole_pressure, cells);
        }
    }

    for (const auto& cells : injector_well_cells)
    {
        if (cells.size() > 0)
        {
            well_manager.AddWell(WellType::Injector, injection_rate, cells);
        }
    }
}

void SPE10Problem::setup_five_spot_pattern(const mfem::Array<int>& N, const int nDim,
                                           WellManager& well_manager, int well_height,
                                           double injection_rate, double bottom_hole_pressure)
{
    if (nDim == 2)
    {
        well_height = 1;
    }
    else
    {
        well_height = std::min(N[2], well_height);
    }

    const int num_wells = 5;

    std::vector<std::vector<int>> producer_well_cells(num_wells - 1);
    std::vector<std::vector<int>>injector_well_cells(1);

    mfem::DenseMatrix point(3, num_wells);
    // Producers
    // Bottom 4
    point(0, 0) = bbmin_[0] + 1.0;
    point(1, 0) = bbmin_[1] + 1.0;
    point(2, 0) = bbmin_[2] + 1.0;

    point(0, 1) = bbmax_[0] - 1.0;
    point(1, 1) = bbmin_[1] + 1.0;
    point(2, 1) = bbmin_[2] + 1.0;

    point(0, 2) = bbmin_[0] + 1.0;
    point(1, 2) = bbmax_[1] - 1.0;
    point(2, 2) = bbmin_[2] + 1.0;

    point(0, 3) = bbmax_[0] - 1.0;
    point(1, 3) = bbmax_[1] - 1.0;
    point(2, 3) = bbmin_[2] + 1.0;

    // Injector, Shifted to avoid middle,
    // Since probably processor boundary
    point(0, 4) = ((bbmax_[0] - bbmin_[0]) / 2.0) + 1.0;
    point(1, 4) = ((bbmax_[1] - bbmin_[1]) / 2.0) + 1.0;
    point(2, 4) = bbmax_[2] - 1.0;


    for (int j = 0; j < well_height; ++j)
    {

        mfem::Array<int> ids;
        mfem::Array<mfem::IntegrationPoint> ips;

        pmesh_->FindPoints(point, ids, ips, false);

        // Producers
        for (int i = 0; i < num_wells - 1; ++i)
        {
            if (ids[i] >= 0)
            {
                producer_well_cells[i].push_back(ids[i]);
            }
        }

        // Injector
        if (ids[num_wells - 1] >= 0)
        {
            injector_well_cells[0].push_back(ids[num_wells - 1]);
        }

        // Shift Points for next layer
        for (int i = 0; i < 4; ++i)
        {
            point(2, i) += 2.0;
        }
        for (int i = 4; i < 5; ++i)
        {
            point(2, i) -= 2.0;
        }
    }

    for (const auto& cells : producer_well_cells)
    {
        if (cells.size() > 0)
        {
            well_manager.AddWell(WellType::Producer, bottom_hole_pressure, cells);
        }
    }

    for (const auto& cells : injector_well_cells)
    {
        if (cells.size() > 0)
        {
            well_manager.AddWell(WellType::Injector, injection_rate, cells);
        }
    }
}

void SPE10Problem::setup_five_spot_pattern_center(const mfem::Array<int>& N, const int nDim,
                                           WellManager& well_manager, int well_height,
                                           double injection_rate, double bottom_hole_pressure, double shift_in)
{
    if (nDim == 2)
    {
        well_height = 1;
    }
    else
    {
        well_height = std::min(N[2], well_height);
    }

    const int num_wells = 5;

    std::vector<std::vector<int>> producer_well_cells(num_wells - 1);
    std::vector<std::vector<int>>injector_well_cells(1);

    mfem::DenseMatrix point(3, num_wells);
    // Producers
    // Bottom 4
    point(0, 0) = bbmin_[0] + shift_in;
    point(1, 0) = bbmin_[1] + shift_in;
    point(2, 0) = ((bbmax_[2] - bbmin_[2]) / 2.0) - (well_height / 2.0) + 1.0;

    point(0, 1) = bbmax_[0] - shift_in;
    point(1, 1) = bbmin_[1] + shift_in;
    point(2, 1) = ((bbmax_[2] - bbmin_[2]) / 2.0) - (well_height / 2.0) + 1.0;

    point(0, 2) = bbmin_[0] + shift_in;
    point(1, 2) = bbmax_[1] - shift_in;
    point(2, 2) = ((bbmax_[2] - bbmin_[2]) / 2.0) - (well_height / 2.0) + 1.0;

    point(0, 3) = bbmax_[0] - shift_in;
    point(1, 3) = bbmax_[1] - shift_in;
    point(2, 3) = ((bbmax_[2] - bbmin_[2]) / 2.0) - (well_height / 2.0) + 1.0;

    // Injector, Shifted to avoid middle,
    // Since probably processor boundary
    point(0, 4) = ((bbmax_[0] - bbmin_[0]) / 2.0) + 1.0;
    point(1, 4) = ((bbmax_[1] - bbmin_[1]) / 2.0) + 1.0;
    point(2, 4) = ((bbmax_[2] - bbmin_[2]) / 2.0) - (well_height / 2.0) + 1.0;


    for (int j = 0; j < well_height; ++j)
    {

        mfem::Array<int> ids;
        mfem::Array<mfem::IntegrationPoint> ips;

        pmesh_->FindPoints(point, ids, ips, false);

        // Producers
        for (int i = 0; i < num_wells - 1; ++i)
        {
            if (ids[i] >= 0)
            {
                producer_well_cells[i].push_back(ids[i]);
            }
        }

        // Injector
        if (ids[num_wells - 1] >= 0)
        {
            injector_well_cells[0].push_back(ids[num_wells - 1]);
        }

        // Shift Points for next layer
        for (int i = 0; i < 4; ++i)
        {
            point(2, i) += 2.0;
        }
        for (int i = 4; i < 5; ++i)
        {
            point(2, i) -= 2.0;
        }
    }

    for (const auto& cells : producer_well_cells)
    {
        if (cells.size() > 0)
        {
            well_manager.AddWell(WellType::Producer, bottom_hole_pressure, cells);
        }
    }

    for (const auto& cells : injector_well_cells)
    {
        if (cells.size() > 0)
        {
            well_manager.AddWell(WellType::Injector, injection_rate, cells);
        }
    }
}


void SPE10Problem::PrintMeshWithPartitioning(mfem::Array<int>& partition)
{
    std::stringstream fname;
    fname << "mesh0.mesh." << std::setfill('0') << std::setw(6) << myid_;
    std::ofstream ofid(fname.str().c_str());
    ofid.precision(8);
    pmesh_->PrintWithPartitioning(partition.GetData(), ofid, 1);
}

//void SPE10Problem::VisualizeSolution(int k, const mfem::BlockVector& interp_sol)
//{
//    mfem::Array<int> offset;
//    mfem::BlockVector vis_vec;
//    remove_well_dofs(well_manager_->GetWells(), interp_sol, offset, vis_vec);
//    smoothg::VisualizeSolution(k, sigmafespace_.get(), ufespace_.get(), vis_vec);
//}

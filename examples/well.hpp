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

#include "smoothG.hpp"
#include "/Users/lee1029/Codes/mfem-install/include/mfem.hpp"

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
        perm_inv_coeff_(&perm_inv_coeff),
        num_producers_(0),
        num_injectors_(0)
    { }

    void AddWell(const WellType type,
                 const double value,
                 const std::vector<int>& cell_indices,
                 const WellDirection direction = WellDirection::Z,
                 const double r_w = 0.01,
                 const double density = 1.0,
                 const double viscosity = 1.0);

    const std::vector<Well>& GetWells() { return wells_; }

    int GetNumProducers() { return num_producers_; }
    int GetNumInjectors() { return num_injectors_; }
private:
    mfem::Mesh* mesh_;
    mfem::VectorCoefficient* perm_inv_coeff_;

    int num_producers_;
    int num_injectors_;

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
        std::fill(cell_sizes.begin(), cell_sizes.end(), 1.0);
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

    num_producers_ += (type == WellType::Producer);
    num_injectors_ += (type == WellType::Injector);
}

ParMatrix ExtendMatrixByIdentity(
    const ParMatrix& pmat, const int id_size)
{
    const std::vector<HYPRE_Int>& old_colmap = pmat.GetColMap();
    const SparseMatrix& diag = pmat.GetDiag();
    const SparseMatrix& offd = pmat.GetOffd();

    const int nrows = diag.Rows() + id_size;
    const int ncols_diag = diag.Cols() + id_size;
    const int nnz_diag = diag.nnz() + id_size;

    auto row_starts = parlinalgcpp::GenerateOffsets(pmat.GetComm(), nrows);
    auto col_starts = parlinalgcpp::GenerateOffsets(pmat.GetComm(), ncols_diag);

    int myid_;
    int num_procs;
    MPI_Comm comm = pmat.GetComm();
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid_);

    int col_diff = col_starts[0] - pmat.GetColStarts()[0];

    int global_true_dofs = pmat.GlobalCols();
    std::vector<int> col_change(global_true_dofs, 0);

    int start = pmat.GetColStarts()[0];
    int end = pmat.GetColStarts()[1];

    for (int i = start; i < end; ++i)
    {
        col_change[i] = col_diff;
    }

    std::vector<int> col_remap(global_true_dofs, 0);
    MPI_Scan(col_change.data(), col_remap.data(), global_true_dofs, HYPRE_MPI_INT, MPI_SUM, comm);
    MPI_Bcast(col_remap.data(), global_true_dofs, HYPRE_MPI_INT, num_procs - 1, comm);

    // Append identity matrix to the bottom left of diag
    std::vector<int> diag_i(nrows + 1);
    std::copy_n(diag.GetIndptr().begin(), diag.Rows() + 1, diag_i.begin());
    std::iota(diag_i.begin() + diag.Rows(), diag_i.end(), diag_i[diag.Rows()]);

    std::vector<int> diag_j(nnz_diag);
    std::copy_n(diag.GetIndices().begin(), diag.nnz(), diag_j.begin());

    for (int i = 0; i < id_size; i++)
    {
        diag_j[diag.nnz() + i] = diag.Cols() + i;
    }

    std::vector<double> diag_data(nnz_diag);
    std::copy_n(diag.GetData().begin(), diag.nnz(), diag_data.begin());
    std::fill_n(diag_data.begin() + diag.nnz(), id_size, 1.0);

    // Append zero matrix to the bottom of offd
    const int ncols_offd = offd.Cols();
    const int nnz_offd = offd.nnz() + id_size;

    std::vector<int> offd_i(nrows + 1);
    std::copy_n(offd.GetIndptr().begin(), offd.Rows() + 1, offd_i.begin());
    std::fill_n(offd_i.begin() + offd.Rows() + 1, id_size, offd_i[offd.Rows()]);

    std::vector<int> offd_j(nnz_offd);
    std::copy_n(offd.GetIndices().begin(), offd.nnz(), offd_j.begin());

    std::vector<double> offd_data(nnz_offd);
    std::copy_n(offd.GetData().begin(), offd.nnz(), offd_data.begin());

    std::vector<HYPRE_Int> colmap(ncols_offd, 0);
    std::copy_n(old_colmap.begin(), ncols_offd, colmap.begin());

    for (int i = 0; i < ncols_offd; ++i)
    {
        colmap[i] += col_remap[colmap[i]];
    }

    SparseMatrix ext_diag(std::move(diag_i), std::move(diag_j),
                          std::move(diag_data), nrows, ncols_diag);
    SparseMatrix ext_offd(std::move(offd_i), std::move(offd_j),
                          std::move(offd_data), nrows, ncols_offd);
    return ParMatrix(pmat.GetComm(), std::move(row_starts), std::move(col_starts),
                     std::move(ext_diag), std::move(ext_offd), std::move(colmap));
}

ParMatrix IntegrateReservoirAndWellModels(
    const std::vector<Well>& well_list, SparseMatrix& vertex_edge,
    std::vector<double>& weight, std::vector<Vector>& local_weight,
    ParMatrix& edge_d_td, Vector& rhs_sigma, Vector& rhs_u)
{
    int num_well_cells = 0;
    for (const auto& well : well_list)
        num_well_cells += well.GetNumberOfCells();

    int num_injectors = 0;
    for (const auto& well : well_list)
    {
        num_injectors += (well.GetType() == WellType::Injector);
    }

    const int num_reservoir_cells = vertex_edge.Rows();
    const int num_reservoir_faces = vertex_edge.Cols();
    const int new_nedges = num_reservoir_faces + num_well_cells;
    const int new_nvertices = num_reservoir_cells + num_injectors;

    // Copying the old data
    std::vector<double> new_weight(new_nedges);
    std::copy(weight.begin(), weight.end(), new_weight.begin());
    Vector new_rhs_sigma(new_nedges);
    std::copy(rhs_sigma.begin(), rhs_sigma.end(), new_rhs_sigma.begin());
    std::fill_n(new_rhs_sigma.begin() + rhs_sigma.size(), num_well_cells, 0.0);
    Vector new_rhs_u(new_nvertices);
    std::copy(rhs_u.begin(), rhs_u.end(), new_rhs_u.begin());

    smoothg::CooMatrix new_vertex_edge(new_nvertices, new_nedges);
    {
        const std::vector<int>& vertex_edge_i = vertex_edge.GetIndptr();
        const std::vector<int>& vertex_edge_j = vertex_edge.GetIndices();
        for (int i = 0; i < num_reservoir_cells; i++)
        {
            for (int j = vertex_edge_i[i]; j < vertex_edge_i[i + 1]; j++)
            {
                new_vertex_edge.Add(i, vertex_edge_j[j], 1.0);
            }
        }
    }

    // Adding well equations to the system
    int edge_counter = num_reservoir_faces;
    int injector_counter = 0;
    for (unsigned int i = 0; i < well_list.size(); i++)
    {
        const auto& well_cells = well_list[i].GetWellCells();
        const auto& well_coeff = well_list[i].GetWellCoeff();

        if (well_list[i].GetType() == WellType::Producer)
        {
            for (unsigned int j = 0; j < well_cells.size(); j++)
            {
                new_vertex_edge.Add(well_cells[j], edge_counter, 1.0);
                new_weight[edge_counter] = well_coeff[j];
                new_rhs_sigma[edge_counter] = well_list[i].GetValue();

                auto& local_weight_j = local_weight[well_cells[j]];
                Vector new_local_weight_j(local_weight_j.size() + 1);
                for (int k = 0; k < local_weight_j.size(); k++)
                {
                    new_local_weight_j[k] = local_weight_j[k];
                }
                new_local_weight_j[local_weight_j.size()] = well_coeff[j];
                swap(local_weight_j, new_local_weight_j);

                edge_counter++;
            }
        }
        else
        {
            auto& local_weight_j = local_weight[num_reservoir_cells + injector_counter];
            local_weight_j.SetSize(well_cells.size());
            local_weight_j = 1e10;//INFINITY; // Not sure if this is ok
            for (unsigned int j = 0; j < well_cells.size(); j++)
            {
                new_vertex_edge.Add(well_cells[j], edge_counter, 1.0);
                new_vertex_edge.Add(num_reservoir_cells + injector_counter, edge_counter, 1.0);
                new_weight[edge_counter] = well_coeff[j];
                auto& local_weight_j = local_weight[well_cells[j]];
                Vector new_local_weight_j(local_weight_j.size() + 1);
                for (int k = 0; k < local_weight_j.size(); k++)
                {
                    new_local_weight_j[k] = local_weight_j[k];
                }
                new_local_weight_j[local_weight_j.size()] = well_coeff[j];
                swap(local_weight_j, new_local_weight_j);

                edge_counter++;
            }
            new_rhs_u[num_reservoir_cells + injector_counter] = -1.0 * well_list[i].GetValue();
            injector_counter++;
        }
    }

    vertex_edge = new_vertex_edge.ToSparse();
    swap(weight, new_weight);
    swap(rhs_sigma, new_rhs_sigma);
    swap(rhs_u, new_rhs_u);

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
                             SparseMatrix& edge_boundaryattr,
                             std::vector<int>& well_marker)
{
    const int old_nedges = edge_boundaryattr.Rows();
    const int old_nnz = edge_boundaryattr.nnz();

    int num_well_cells = 0;
    int new_nnz = old_nnz;

    for (auto& well : well_list)
    {
        int num_wells_i = well.GetNumberOfCells();
        num_well_cells += num_wells_i;
        if (well.GetType() == WellType::Producer)
            new_nnz += num_wells_i;
    }

    std::vector<int> new_i(old_nedges + num_well_cells + 1);
    std::vector<int> new_j(new_nnz);
    std::vector<double> new_data(new_nnz);
    well_marker.resize(old_nedges + num_well_cells, 0);

    std::fill_n(new_j.begin(), old_nnz, -5);
    std::copy_n(edge_boundaryattr.GetIndptr().begin(), old_nedges + 1, new_i.begin());
    std::copy_n(edge_boundaryattr.GetIndices().begin(), old_nnz, new_j.begin());
    std::copy_n(edge_boundaryattr.GetData().begin(), old_nnz, new_data.begin());

    //int counter = old_nedges;
    int counter = 0;
    int new_attr = edge_boundaryattr.Cols();

    for (const Well& well : well_list)
    {
        int num_cells = well.GetNumberOfCells();

        if (well.GetType() == WellType::Producer)
        {
            for (int j = 0; j < num_cells; ++j)
            {
                new_j[old_nnz + counter + j] = new_attr;
                new_data[old_nnz + counter + j] = 1.0;
                new_i[old_nedges + counter + j + 1] = new_i[old_nedges + counter] + j + 1;
                well_marker[old_nedges + counter + j] = 1;
            }
        }
        else
        {
            for (int j = 0; j < num_cells; ++j)
            {
                new_i[old_nedges + counter + j + 1] = new_i[old_nedges + counter];
            }
        }

        counter += num_cells;
    }

    assert(new_i[old_nedges + num_well_cells] == new_nnz);

    SparseMatrix new_edge_boundaryattr(new_i, new_j, new_data, new_nnz,
                                       edge_boundaryattr.Cols() + 1);
    swap(edge_boundaryattr, new_edge_boundaryattr);
}

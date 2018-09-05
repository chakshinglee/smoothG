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

/** @file

    @brief Implementations of some utility routines for linear algebra.

    These are implemented with and operate on linalgcpp data structures.
*/

#include "Utilities.hpp"

namespace smoothg
{

int MyId(MPI_Comm comm)
{
    int myid;
    MPI_Comm_rank(comm, &myid);

    return myid;
}

ParMatrix MakeEdgeTrueEdge(MPI_Comm comm, const SparseMatrix& proc_edge,
                           const std::vector<int>& edge_map)
{
    int myid;
    int num_procs;

    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    auto edge_proc = proc_edge.Transpose();

    int num_edges_local = proc_edge.RowSize(myid);
    int num_tedges_global = proc_edge.Cols();

    std::vector<int> tedge_counter(num_procs + 1, 0);

    for (int i = 0; i < num_tedges_global; ++i)
    {
        tedge_counter[edge_proc.GetIndices(i)[0] + 1]++;
    }

    int num_tedges_local = tedge_counter[myid + 1];
    int num_edge_diff = num_edges_local - num_tedges_local;
    std::partial_sum(std::begin(tedge_counter), std::end(tedge_counter),
                     std::begin(tedge_counter));

    assert(tedge_counter.back() == static_cast<int>(num_tedges_global));

    std::vector<int> edge_perm(num_tedges_global);

    for (int i = 0; i < num_tedges_global; ++i)
    {
        edge_perm[i] = tedge_counter[edge_proc.GetIndices(i)[0]]++;
    }

    for (int i = num_procs - 1; i > 0; i--)
    {
        tedge_counter[i] = tedge_counter[i - 1];
    }
    tedge_counter[0] = 0;

    std::vector<int> diag_indptr(num_edges_local + 1);
    std::vector<int> diag_indices(num_tedges_local);
    std::vector<double> diag_data(num_tedges_local, 1.0);

    std::vector<int> offd_indptr(num_edges_local + 1);
    std::vector<int> offd_indices(num_edge_diff);
    std::vector<double> offd_data(num_edge_diff, 1.0);
    std::vector<HYPRE_Int> col_map(num_edge_diff);
    std::vector<std::pair<HYPRE_Int, int>> offd_map(num_edge_diff);

    diag_indptr[0] = 0;
    offd_indptr[0] = 0;

    int tedge_begin = tedge_counter[myid];
    int tedge_end = tedge_counter[myid + 1];

    int diag_counter = 0;
    int offd_counter = 0;

    for (int i = 0; i < num_edges_local; ++i)
    {
        int tedge = edge_perm[edge_map[i]];

        if ((tedge >= tedge_begin) && (tedge < tedge_end))
        {
            diag_indices[diag_counter++] = tedge - tedge_begin;
        }
        else
        {
            offd_map[offd_counter].first = tedge;
            offd_map[offd_counter].second = offd_counter;
            offd_counter++;
        }

        diag_indptr[i + 1] = diag_counter;
        offd_indptr[i + 1] = offd_counter;
    }

    assert(offd_counter == static_cast<int>(num_edge_diff));

    auto compare = [] (const std::pair<HYPRE_Int, int>& lhs,
                       const std::pair<HYPRE_Int, int>& rhs)
    {
        return lhs.first < rhs.first;
    };

    std::sort(std::begin(offd_map), std::end(offd_map), compare);

    for (int i = 0; i < offd_counter; ++i)
    {
        offd_indices[offd_map[i].second] = i;
        col_map[i] = offd_map[i].first;
    }

    auto starts = parlinalgcpp::GenerateOffsets(comm, {num_edges_local, num_tedges_local});

    SparseMatrix diag(std::move(diag_indptr), std::move(diag_indices), std::move(diag_data),
                      num_edges_local, num_tedges_local);

    SparseMatrix offd(std::move(offd_indptr), std::move(offd_indices), std::move(offd_data),
                      num_edges_local, num_edge_diff);

    return ParMatrix(comm, starts[0], starts[1],
                     std::move(diag), std::move(offd),
                     std::move(col_map));
}

SparseMatrix RemoveLargeEntries(const SparseMatrix& mat, double tol)
{
    int rows = mat.Rows();
    int cols = mat.Cols();

    const auto& mat_indptr(mat.GetIndptr());
    const auto& mat_indices(mat.GetIndices());
    const auto& mat_data(mat.GetData());

    std::vector<int> indptr(rows + 1);
    std::vector<int> indices;

    indices.reserve(mat.nnz());

    for (int i = 0; i < rows; ++i)
    {
        indptr[i] = indices.size();

        for (int j = mat_indptr[i]; j < mat_indptr[i + 1]; ++j)
        {
            if (std::fabs(mat_data[j]) > tol)
            {
                indices.push_back(mat_indices[j]);
            }
        }
    }

    indptr[rows] = indices.size();

    std::vector<double> data(indices.size(), 1);

    return SparseMatrix(std::move(indptr), std::move(indices), std::move(data), rows, cols);
}

ParMatrix RemoveLargeEntries(const ParMatrix& mat, double tol)
{
    int num_rows = mat.Rows();

    const auto& diag_ext = mat.GetDiag();
    const auto& offd_ext = mat.GetOffd();
    const auto& colmap_ext = mat.GetColMap();

    const auto& offd_indptr = offd_ext.GetIndptr();
    const auto& offd_indices = offd_ext.GetIndices();
    const auto& offd_data = offd_ext.GetData();
    const int num_offd = offd_ext.Cols();

    std::vector<int> indptr(num_rows + 1);
    std::vector<int> offd_marker(num_offd, -1);

    int offd_nnz = 0;

    for (int i = 0; i < num_rows; ++i)
    {
        indptr[i] = offd_nnz;

        for (int j = offd_indptr[i]; j < offd_indptr[i + 1]; ++j)
        {
            if (std::fabs(offd_data[j]) > tol)
            {
                offd_marker[offd_indices[j]] = 1;
                offd_nnz++;
            }
        }
    }

    indptr[num_rows] = offd_nnz;

    int offd_num_cols = std::count_if(std::begin(offd_marker), std::end(offd_marker),
    [](int i) { return i > 0; });

    std::vector<HYPRE_Int> col_map(offd_num_cols);
    int count = 0;

    for (int i = 0; i < num_offd; ++i)
    {
        if (offd_marker[i] > 0)
        {
            offd_marker[i] = count;
            col_map[count] = colmap_ext[i];

            count++;
        }
    }

    assert(count == offd_num_cols);

    std::vector<int> indices(offd_nnz);
    std::vector<double> data(offd_nnz, 1.0);

    count = 0;

    for (int i = 0; i < num_rows; i++)
    {
        for (int j = offd_indptr[i]; j < offd_indptr[i + 1]; ++j)
        {
            if (offd_data[j] > 1)
            {
                indices[count++] = offd_marker[offd_indices[j]];
            }
        }
    }

    assert(count == offd_nnz);

    SparseMatrix diag = RemoveLargeEntries(diag_ext);
    SparseMatrix offd(std::move(indptr), std::move(indices), std::move(data),
                      num_rows, offd_num_cols);

    return ParMatrix(mat.GetComm(), mat.GetRowStarts(), mat.GetColStarts(),
                     std::move(diag), std::move(offd), std::move(col_map));
}

ParMatrix MakeEntityTrueEntity(const ParMatrix& entity_entity)
{
    const auto& offd = entity_entity.GetOffd();

    const auto& offd_indptr = offd.GetIndptr();
    const auto& offd_indices = offd.GetIndices();
    const auto& offd_colmap = entity_entity.GetColMap();

    HYPRE_Int last_row = entity_entity.GetColStarts()[1];

    int num_entities = entity_entity.Rows();
    std::vector<int> select_indptr(num_entities + 1);

    int num_true_entities = 0;

    for (int i = 0; i < num_entities; ++i)
    {
        select_indptr[i] = num_true_entities;

        int row_size = offd.RowSize(i);

        if (row_size == 0 || offd_colmap[offd_indices[offd_indptr[i]]] >= last_row )
        {
            assert(row_size == 0 || row_size == 1);
            num_true_entities++;
        }
    }

    select_indptr[num_entities] = num_true_entities;

    std::vector<int> select_indices(num_true_entities);
    std::iota(std::begin(select_indices), std::end(select_indices), 0);

    std::vector<double> select_data(num_true_entities, 1.0);

    SparseMatrix select(std::move(select_indptr), std::move(select_indices), std::move(select_data),
                        num_entities, num_true_entities);

    MPI_Comm comm = entity_entity.GetComm();
    auto true_starts = parlinalgcpp::GenerateOffsets(comm, num_true_entities);

    ParMatrix select_d(comm, entity_entity.GetRowStarts(), true_starts, std::move(select));

    return entity_entity.Mult(select_d);
}

SparseMatrix SparseIdentity(int size)
{
    assert(size >= 0);

    return SparseMatrix(std::vector<double>(size, 1.0));
}

SparseMatrix SparseIdentity(int rows, int cols, int row_offset, int col_offset)
{
    assert(rows >= 0);
    assert(cols >= 0);
    assert(row_offset <= rows);
    assert(row_offset >= 0);
    assert(col_offset <= cols);
    assert(col_offset >= 0);

    const int diag_size = std::min(rows - row_offset, cols - col_offset);

    std::vector<int> indptr(rows + 1);

    std::fill(std::begin(indptr), std::begin(indptr) + row_offset, 0);
    std::iota(std::begin(indptr) + row_offset, std::begin(indptr) + row_offset + diag_size, 0);
    std::fill(std::begin(indptr) + row_offset + diag_size, std::begin(indptr) + rows + 1, diag_size);

    std::vector<int> indices(diag_size);
    std::iota(std::begin(indices), std::begin(indices) + diag_size, col_offset);

    std::vector<double> data(diag_size, 1.0);

    return SparseMatrix(std::move(indptr), std::move(indices), std::move(data), rows, cols);
}

void SetMarker(std::vector<int>& marker, const std::vector<int>& indices)
{
    const int size = indices.size();

    for (int i = 0; i < size; ++i)
    {
        assert(indices[i] < static_cast<int>(marker.size()));

        marker[indices[i]] = i;
    }
}

void ClearMarker(std::vector<int>& marker, const std::vector<int>& indices)
{
    const int size = indices.size();

    for (int i = 0; i < size; ++i)
    {
        assert(indices[i] < static_cast<int>(marker.size()));

        marker[indices[i]] = -1;
    }
}

DenseMatrix Orthogonalize(DenseMatrix& mat, VectorView vect_view, int offset, int max_keep)
{
    assert(mat.Rows() == vect_view.size());

    // If the view is of mat, deflate will destroy it,
    // so copy is needed
    Normalize(vect_view);
    Vector vect(vect_view);

    int sz = 0;

    if (mat.Cols() > offset) // 0 or 1
    {
        Deflate(mat, vect);
        auto singular_values = mat.SVD();
        int num_values = singular_values.size();

        if (singular_values[0] > 1e-8)
        {
            double zero_tol = 1e-8 * singular_values[0];
            for (; sz < num_values; ++sz)
            {
                if (singular_values[sz] < zero_tol)
                {
                    break;
                }
            }
        }
    }

    sz = std::min(max_keep - 1, sz);
    DenseMatrix out(mat.Rows(), sz + 1);

    out.SetCol(0, vect);

    for (int i = 0; i < sz; ++i)
    {
        out.SetCol(i + 1, mat.GetColView(i));
    }

    return out;
}

void OrthoConstant(DenseMatrix& mat)
{
    int cols = mat.Cols();

    for (int i = 0; i < cols; ++i)
    {
        VectorView col = mat.GetColView(i);
        SubAvg(col);
    }
}

void OrthoConstant(DenseMatrix& mat, const VectorView& constant)
{
    int cols = mat.Cols();

    Vector vect(constant);
    Normalize(vect);

    for (int i = 0; i < cols; ++i)
    {
        VectorView col = mat.GetColView(i);
        col.Sub(col.Mult(vect), vect);
    }
}

void OrthoConstant(VectorView vect)
{
    SubAvg(vect);
}

void OrthoConstant(MPI_Comm comm, VectorView vect, int global_size)
{
    double local_sum = vect.Sum();
    double global_sum = 0.0;

    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);

    vect -= global_sum / global_size;
}

void OrthoConstant(MPI_Comm comm, VectorView vect, const VectorView& constant)
{
    double local_sum = constant.Mult(vect);
    double global_sum = 0.0;

    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);

    vect.Sub(global_sum, constant);
}

void Deflate(DenseMatrix& A, const VectorView& v)
{
    int rows = A.Rows();
    int cols = A.Cols();

    assert(v.size() == rows);

    Vector v_T_A = A.MultAT(v);

    for (int j = 0; j < cols; ++j)
    {
        double vt_A_j = v_T_A[j];

        for (int i = 0; i < rows; ++i)
        {
            A(i, j) -= v[i] * vt_A_j;
        }
    }
}

double DivError(MPI_Comm comm, const SparseMatrix& D, const VectorView& numer,
                const VectorView& denom)
{
    Vector sigma_diff(denom);
    sigma_diff -= numer;

    Vector Dfine = D.Mult(denom);
    Vector Ddiff = D.Mult(sigma_diff);

    const double error = parlinalgcpp::ParL2Norm(comm, Ddiff) /
                         parlinalgcpp::ParL2Norm(comm, Dfine);

    return error;
}

double CompareError(MPI_Comm comm, const VectorView& numer, const VectorView& denom)
{
    Vector diff(denom);
    diff -= numer;

    const double error = parlinalgcpp::ParL2Norm(comm, diff) /
                         parlinalgcpp::ParL2Norm(comm, denom);

    return error;
}

void ShowErrors(const std::vector<double>& error_info, std::ostream& out, bool pretty)
{
    assert(error_info.size() >= 3);

    std::map<std::string, double> values =
    {
        {"finest-p-error", error_info[0]},
        {"finest-u-error", error_info[1]},
        {"finest-div-error", error_info[2]}
    };

    if (error_info.size() > 3)
    {
        values.emplace(std::make_pair("operator-complexity", error_info[3]));
    }

    PrintJSON(values, out, pretty);
}

std::vector<double> ComputeErrors(MPI_Comm comm, const SparseMatrix& M,
                                  const SparseMatrix& D,
                                  const BlockVector& upscaled_sol,
                                  const BlockVector& fine_sol)
{
    BlockVector M_scaled_up_sol(upscaled_sol);
    BlockVector M_scaled_fine_sol(fine_sol);

    const std::vector<double>& M_data = M.GetData();

    const int num_edges = upscaled_sol.GetBlock(0).size();

    for (int i = 0; i < num_edges; ++i)
    {
        assert(M_data[i] >= 0);

        M_scaled_up_sol[i] *= std::sqrt(M_data[i]);
        M_scaled_fine_sol[i] *= std::sqrt(M_data[i]);
    }

    std::vector<double> info(3);

    info[0] = CompareError(comm, M_scaled_up_sol.GetBlock(1), M_scaled_fine_sol.GetBlock(1));  // vertex
    info[1] = CompareError(comm, M_scaled_up_sol.GetBlock(0), M_scaled_fine_sol.GetBlock(0));  // edge
    info[2] = DivError(comm, D, upscaled_sol.GetBlock(0), fine_sol.GetBlock(0));   // div

    return info;
}

void PrintJSON(const std::map<std::string, double>& values, std::ostream& out,
               bool pretty)
{
    const std::string new_line = pretty ? "\n" : "";
    const std::string indent = pretty ? "  " : "";
    std::stringstream ss;

    out << "{" << new_line;

    for (const auto& pair : values)
    {
        ss.str("");
        ss << indent << "\"" << std::right << pair.first << "\": "
           << std::left << std::setprecision(16) << pair.second;

        if (&pair != &(*values.rbegin()))
        {
            ss << std::left << ",";
        }

        ss << new_line;

        out << ss.str();
    }

    out << "}" << new_line;
}

SparseMatrix MakeProcAgg(int num_procs, int num_aggs_global)
{
    int num_aggs_local = num_aggs_global / num_procs;
    int num_left = num_aggs_global % num_procs;

    std::vector<int> indptr(num_procs + 1);
    std::vector<int> indices(num_aggs_global);
    std::vector<double> data(num_aggs_global, 1.0);

    std::iota(std::begin(indices), std::end(indices), 0);

    for (int i = 0; i <= num_left; ++i)
    {
        indptr[i] = i * (num_aggs_local + 1);
    }

    for (int i = num_left + 1; i <= num_procs; ++i)
    {
        indptr[i] = indptr[i - 1] + num_aggs_local;
    }

    return SparseMatrix(std::move(indptr), std::move(indices), std::move(data),
                        num_procs, num_aggs_global);
}

SparseMatrix MakeAggVertex(const std::vector<int>& partition)
{
    assert(partition.size() > 0);

    const int num_parts = *std::max_element(std::begin(partition), std::end(partition)) + 1;
    const int num_vert = partition.size();

    std::vector<int> indptr(num_vert + 1);
    std::vector<double> data(num_vert, 1);

    std::iota(std::begin(indptr), std::end(indptr), 0);

    SparseMatrix vertex_agg(std::move(indptr), partition, std::move(data), num_vert, num_parts);

    return vertex_agg.Transpose();
}

double PowerIterate(MPI_Comm comm, const linalgcpp::Operator& A, VectorView result,
                    int max_iter, double tol, bool verbose)
{
    using parlinalgcpp::ParMult;
    using parlinalgcpp::ParL2Norm;

    int myid;
    MPI_Comm_rank(comm, &myid);

    Vector temp(result.size());

    double rayleigh = 0.0;
    double old_rayleigh = 0.0;

    for (int i = 0; i < max_iter; ++i)
    {
        A.Mult(result, temp);

        rayleigh = ParMult(comm, temp, result) / ParMult(comm, result, result);
        temp /= ParL2Norm(comm, temp);

        swap(temp, result);

        if (verbose && myid == 0)
        {
            std::cout << std::scientific;
            std::cout << " i: " << i << " ray: " << rayleigh;
            std::cout << " inverse: " << (1.0 / rayleigh);
            std::cout << " rate: " << (std::fabs(rayleigh - old_rayleigh) / rayleigh) << "\n";
        }

        if (std::fabs(rayleigh - old_rayleigh) / std::fabs(rayleigh) < tol)
        {
            break;
        }

        old_rayleigh = rayleigh;
    }

    return rayleigh;
}

void BroadCast(MPI_Comm comm, SparseMatrix& mat)
{
    int myid;
    MPI_Comm_rank(comm, &myid);

    int sizes[3];

    if (myid == 0)
    {
        sizes[0] = mat.Rows();
        sizes[1] = mat.Cols();
        sizes[2] = mat.nnz();
    }

    MPI_Bcast(sizes, 3, MPI_INT, 0, comm);

    bool master = (myid == 0);

    std::vector<int> indptr(master ? 0 : sizes[0] + 1);
    std::vector<int> indices(master ? 0 : sizes[2]);
    std::vector<double> data(master ? 0 : sizes[2]);

    int* I_ptr = master ? mat.GetIndptr().data() : indptr.data();
    int* J_ptr = master ? mat.GetIndices().data() : indices.data();
    double* Data_ptr = master ? mat.GetData().data() : data.data();

    MPI_Bcast(I_ptr, sizes[0] + 1, MPI_INT, 0, comm);
    MPI_Bcast(J_ptr, sizes[2], MPI_INT, 0, comm);
    MPI_Bcast(Data_ptr, sizes[2], MPI_DOUBLE, 0, comm);

    if (myid != 0)
    {
        mat = SparseMatrix(std::move(indptr), std::move(indices), std::move(data),
                           sizes[0], sizes[1]);
    }
}

//TODO(gelever1): Define this inplace in linalgcpp
SparseMatrix Add(double alpha, const SparseMatrix& A, double beta, const SparseMatrix& B)
{
    assert(A.Rows() == B.Rows());
    assert(A.Cols() == B.Cols());

    CooMatrix coo(A.Rows(), A.Cols());

    auto add_mat = [&coo](double scale, const SparseMatrix & mat)
    {
        const auto& indptr = mat.GetIndptr();
        const auto& indices = mat.GetIndices();
        const auto& data = mat.GetData();

        int rows = mat.Rows();

        for (int i = 0; i < rows; ++i)
        {
            for (int j = indptr[i]; j < indptr[i + 1]; ++j)
            {
                coo.Add(i, indices[j], scale * data[j]);
            }
        }
    };

    add_mat(alpha, A);
    add_mat(beta, B);

    return coo.ToSparse();
}

void IsolatePreProcess(const SparseMatrix& wtable,
                       const std::vector<int>& isolated_vertices,
                       SparseMatrix& sub_table,
                       std::vector<int>& sub_to_global)
{
    sub_to_global.resize(0);
    sub_to_global.reserve(wtable.Rows());
    std::vector<int> full_indices(wtable.Rows());
    std::iota(full_indices.begin(), full_indices.end(), 0);
    std::set_difference(full_indices.begin(), full_indices.end(),
                        isolated_vertices.begin(), isolated_vertices.end(),
                        std::back_inserter(sub_to_global));

    sub_table = wtable.GetSubMatrix(sub_to_global, sub_to_global);
}

std::vector<int> PartitionAAT(const SparseMatrix& A, double coarsening_factor, double ubal,
                              bool contig, const std::vector<int>& isolated_vertices)
{
    std::vector<double> weight(A.Cols(), 1.0);
    return PartitionAAT(A, weight, coarsening_factor, ubal, contig, false, isolated_vertices);
}

std::vector<int> PartitionAAT(const SparseMatrix& A, const std::vector<double>& weight,
                              double coarsening_factor, double ubal, bool contig,
                              bool weighted, const std::vector<int>& isolated_vertices)
{
    std::vector<double> weight_inv_vec(weight);
    for (auto& w : weight_inv_vec)
    {
        w = 1./w;
    }
    SparseMatrix weight_inv(weight_inv_vec);

    SparseMatrix A_T = A.Transpose();
    SparseMatrix AA_T = RescaleLog(Mult(A, weight_inv, A_T));

    int num_parts = std::max(1.0, (A.Rows() / coarsening_factor) + 0.5);
    if (isolated_vertices.size() == 0)
    {
        return Partition(AA_T, num_parts, ubal, contig, weighted);
    }
    else
    {
        SparseMatrix sub_AA_T;
        std::vector<int> sub_verts;
        IsolatePreProcess(AA_T, isolated_vertices, sub_AA_T, sub_verts);

        auto sub_part = Partition(sub_AA_T, num_parts, ubal, contig, weighted);

        std::vector<int> partitioning(A.Rows());
        for (unsigned int i = 0; i < sub_part.size(); ++i)
        {
            partitioning[sub_verts[i]] = sub_part[i];
        }

        num_parts = *(std::max_element(sub_part.begin(), sub_part.end()));
        for (auto vertex : isolated_vertices)
        {
            partitioning[vertex] = ++num_parts;
        }

        return partitioning;
    }
}

Vector ReadVector(const std::string& filename,
                  const std::vector<int>& local_to_global)
{
    std::vector<double> global_vect = linalgcpp::ReadText(filename);

    int local_size = local_to_global.size();

    Vector local_vect(local_size);

    for (int i = 0; i < local_size; ++i)
    {
        local_vect[i] = global_vect[local_to_global[i]];
    }

    return local_vect;
}

std::vector<int> GetElementColoring(const SparseMatrix& el_el)
{
    int num_el = el_el.Rows();
    int stack_p = 0;
    int stack_top_p = 0;
    int max_num_colors = 1;

    std::vector<int> el_stack(num_el);

    const auto& i_el_el = el_el.GetIndptr();
    const auto& j_el_el = el_el.GetIndices();

    std::vector<int> colors(num_el, -2);

    for (int el = 0; stack_top_p < num_el; el = (el + 1) % num_el)
    {
        if (colors[el] != -2)
        {
            continue;
        }

        colors[el] = -1;
        el_stack[stack_top_p++] = el;

        for ( ; stack_p < stack_top_p; stack_p++)
        {
            int i = el_stack[stack_p];
            int num_nb = i_el_el[i + 1] - i_el_el[i] - 1; // assume nonzero diagonal
            max_num_colors = std::max(max_num_colors, num_nb + 1);

            for (int j = i_el_el[i]; j < i_el_el[i + 1]; j++)
            {
                int k = j_el_el[j];

                if (j == i)
                {
                    continue; // skip self-interaction
                }

                if (colors[k] == -2)
                {
                    colors[k] = -1;
                    el_stack[stack_top_p++] = k;
                }
            }
        }
    }

    std::vector<int> color_marker(max_num_colors);

    for (int stack_p = 0; stack_p < stack_top_p; stack_p++)
    {
        int i = el_stack[stack_p];
        std::fill(std::begin(color_marker), std::end(color_marker), 0);

        for (int j = i_el_el[i]; j < i_el_el[i + 1]; j++)
        {
            if (j_el_el[j] == i)
            {
                continue;          // skip self-interaction
            }

            int color = colors[j_el_el[j]];

            if (color != -1)
            {
                color_marker[color] = 1;
            }
        }

        int color = 0;

        while (color < max_num_colors && color_marker[color] != 0)
        {
            color++;
        }

        colors[i] = color;
    }

    return colors;
}

bool IsDiag(const SparseMatrix& mat)
{
    if (mat.Rows() != mat.Cols() || mat.nnz() != mat.Rows())
    {
        return false;
    }

    const auto& indptr = mat.GetIndptr();
    const auto& indices = mat.GetIndices();

    int rows = mat.Rows();

    for (int i = 0; i < rows; ++i)
    {
        for (int j = indptr[i]; j < indptr[i + 1]; ++j)
        {
            if (indices[j] != i)
            {
                return false;
            }
        }
    }

    return true;
}

void GetSubMatrix(const SparseMatrix& mat, const std::vector<int>& rows,
                  const std::vector<int>& cols, std::vector<int>& col_map,
                  DenseMatrix& dense_mat)
{
    SetMarker(col_map, cols);

    int num_rows = rows.size();
    int num_cols = cols.size();

    dense_mat.SetSize(num_rows, num_cols);
    dense_mat = 0.0;

    const auto& indptr = mat.GetIndptr();
    const auto& indices = mat.GetIndices();
    const auto& data = mat.GetData();

    for (int i = 0; i < num_rows; ++i)
    {
        int row = rows[i];

        for (int j = indptr[row]; j < indptr[row + 1]; ++j)
        {
            int col = col_map[indices[j]];

            if (col > -1)
            {
                dense_mat(i, col) = data[j];
            }
        }
    }

    ClearMarker(col_map, cols);
}

void OffsetMult(const linalgcpp::Operator& A, const DenseMatrix& input, DenseMatrix& output, int offset)
{
    assert(offset >= 0);
    assert(offset < input.Cols());

    int cols = input.Cols();
    int off_cols = cols - offset;

    output.SetSize(A.Rows(), off_cols);

    for (int i = 0; i < off_cols; ++i)
    {
        A.Mult(input.GetColView(i + offset), output.GetColView(i));
    }
}

void OffsetMultAT(const linalgcpp::Operator& A, const DenseMatrix& input, DenseMatrix& output, int offset)
{
    assert(offset >= 0);
    assert(offset < input.Cols());

    int cols = input.Cols();
    int off_cols = cols - offset;

    output.SetSize(A.Cols(), off_cols);

    for (int i = 0; i < off_cols; ++i)
    {
        A.MultAT(input.GetColView(i + offset), output.GetColView(i));
    }
}

DenseMatrix OuterProduct(const VectorView& lhs, const VectorView& rhs)
{
    DenseMatrix out;

    OuterProduct(lhs, rhs, out);

    return out;
}

void OuterProduct(const VectorView& lhs, const VectorView& rhs, DenseMatrix& product)
{
    int rows = lhs.size();
    int cols = rhs.size();

    product.SetSize(rows, cols);

    for (int j = 0; j < cols; ++j)
    {
        for (int i = 0; i < rows; ++i)
        {
            product(i, j) = lhs[i] * rhs[j];
        }
    }
}

double Min(MPI_Comm comm, const VectorView& vect)
{
    double local_min = Min(vect);

    double global_min;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, comm);

    return global_min;
}

SparseMatrix RescaleLog(SparseMatrix A)
{
    std::vector<int>& indptr = A.GetIndptr();
    std::vector<int>& indices = A.GetIndices();
    std::vector<double>& data = A.GetData();
    int num_rows = A.Rows();
    double weight_min = *std::min_element(std::begin(data), std::end(data));
    assert(weight_min != 0);
    for (int i = 0; i < num_rows; ++i)
    {
        for (int j = indptr[i]; j < indptr[i + 1]; ++j)
        {
            if (i != indices[j])
            {
                data[j] = std::floor(std::log2(data[j] / weight_min)) + 1;
            }
        }
    }
    return std::move(A);
}

int FindFirstNotShared(const SparseMatrix& is_shared)
{
    for (int i = 0; i < is_shared.Rows(); ++i)
    {
        if (is_shared.RowSize(i) == 0)
        {
            return i;
        }
    }
    return -1;
}

} // namespace smoothg

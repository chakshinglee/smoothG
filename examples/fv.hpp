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
   @file fv.hpp
   @brief Implementation of finite volume discretization of PDEs.

   Build somewhat sophisticated well models, integrate them with a reservoir model
*/

#include "well.hpp"

using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;

namespace smoothg
{

smoothg::SparseMatrix SparseToSparse(const mfem::SparseMatrix& sparse)
{
    const int height = sparse.Height();
    const int width = sparse.Width();
    const int nnz = sparse.NumNonZeroElems();

    std::vector<int> indptr(height + 1);
    std::vector<int> indices(nnz);
    std::vector<double> data(nnz);

    std::copy_n(sparse.GetI(), height + 1, std::begin(indptr));
    std::copy_n(sparse.GetJ(), nnz, std::begin(indices));
    std::copy_n(sparse.GetData(), nnz, std::begin(data));

    return smoothg::SparseMatrix(std::move(indptr), std::move(indices), std::move(data), height, width);
}

smoothg::ParMatrix ParMatrixToParMatrix(const mfem::HypreParMatrix& mat)
{
    mfem::SparseMatrix mfem_diag;
    mfem::SparseMatrix mfem_offd;
    HYPRE_Int* mfem_map;

    mat.GetDiag(mfem_diag);
    mat.GetOffd(mfem_offd, mfem_map);

    smoothg::SparseMatrix diag = SparseToSparse(mfem_diag);
    smoothg::SparseMatrix offd = SparseToSparse(mfem_offd);

    int col_map_size = offd.Cols();
    std::vector<HYPRE_Int> col_map(mfem_map, mfem_map + col_map_size);

    std::vector<HYPRE_Int> row_starts(mat.RowPart(), mat.RowPart() + 2);
    std::vector<HYPRE_Int> col_starts(mat.ColPart(), mat.ColPart() + 2);

    row_starts.push_back(mat.M());
    col_starts.push_back(mat.N());

    MPI_Comm comm = mat.GetComm();

    return smoothg::ParMatrix(comm, row_starts, col_starts,
                              std::move(diag), std::move(offd), std::move(col_map));
}

smoothg::SparseMatrix GenerateBoundaryAttributeTable(const mfem::Mesh& mesh)
{
    int nedges = mesh.Dimension() == 2 ? mesh.GetNEdges() : mesh.GetNFaces();
    int nbdr = mesh.bdr_attributes.Max();
    int nbdr_edges = mesh.GetNBE();

    std::vector<int> indptr(nedges + 1, 0);
    std::vector<int> indices(nbdr_edges);
    std::vector<double> data(nbdr_edges, 1.0);

    for (int j = 0; j < nbdr_edges; j++)
    {
        int edge = mesh.GetBdrElementEdgeIndex(j);
        indptr[edge + 1] = mesh.GetBdrAttribute(j);
    }

    int count = 0;

    for (int j = 1; j <= nedges; j++)
    {
        if (indptr[j])
        {
            indices[count++] = indptr[j] - 1;
            indptr[j] = indptr[j - 1] + 1;
        }
        else
        {
            indptr[j] = indptr[j - 1];
        }
    }

    return smoothg::SparseMatrix(std::move(indptr), std::move(indices), std::move(data),
                                 nedges, nbdr);
}

/**
   @brief Finite volume integrator

   This is the integrator for the artificial mass matrix in a finite
   volume discretization, tricking MFEM into doing finite volumes instead
   of finite elements.
*/
class FiniteVolumeMassIntegrator: public mfem::BilinearFormIntegrator
{
protected:
    mfem::Coefficient* Q;
    mfem::VectorCoefficient* VQ;
    mfem::MatrixCoefficient* MQ;

    // these are not thread-safe!
    mfem::Vector nor, ni;
    mfem::Vector unitnormal; // ATB 25 February 2015
    double sq;
    mfem::Vector vq;
    mfem::DenseMatrix mq;

public:
    ///@name Constructors differ by whether the coefficient (permeability) is scalar, vector, or full tensor
    ///@{
    FiniteVolumeMassIntegrator() :
        Q(NULL), VQ(NULL), MQ(NULL)
    {
    }
    FiniteVolumeMassIntegrator(mfem::Coefficient& q) :
        Q(&q), VQ(NULL), MQ(NULL)
    {
    }
    FiniteVolumeMassIntegrator(mfem::VectorCoefficient& q) :
        Q(NULL), VQ(&q), MQ(NULL)
    {
    }
    FiniteVolumeMassIntegrator(mfem::MatrixCoefficient& q) :
        Q(NULL), VQ(NULL), MQ(&q)
    {
    }
    ///@}

    using mfem::BilinearFormIntegrator::AssembleElementMatrix;
    /// Implements interface for MFEM's BilinearForm
    virtual void AssembleElementMatrix (const mfem::FiniteElement& el,
                                        mfem::ElementTransformation& Trans,
                                        mfem::DenseMatrix& elmat);
}; // class FiniteVolumeMassIntegrator


void FiniteVolumeMassIntegrator::AssembleElementMatrix(
    const mfem::FiniteElement& el,
    mfem::ElementTransformation& Trans,
    mfem::DenseMatrix& elmat)
{
    int dim = el.GetDim();
    int ndof = el.GetDof();
    elmat.SetSize(ndof);
    elmat = 0.0;

    mq.SetSize(dim);

    int order = 1;
    const mfem::IntegrationRule* ir = &mfem::IntRules.Get(el.GetGeomType(), order);

    MFEM_ASSERT(ir->GetNPoints() == 1, "Only implemented for piecewise "
                "constants!");

    int p = 0;
    const mfem::IntegrationPoint& ip = ir->IntPoint(p);

    if (VQ)
    {
        vq.SetSize(dim);
        VQ->Eval(vq, Trans, ip);
        for (int i = 0; i < dim; i++)
            mq(i, i) = vq(i);
    }
    else if (Q)
    {
        sq = Q->Eval(Trans, ip);
        for (int i = 0; i < dim; i++)
            mq(i, i) = sq;
    }
    else if (MQ)
        MQ->Eval(mq, Trans, ip);
    else
    {
        for (int i = 0; i < dim; i++)
            mq(i, i) = 1.0;
    }

    // Compute face area of each face
    mfem::DenseMatrix vshape;
    vshape.SetSize(ndof, dim);
    Trans.SetIntPoint(&ip);
    el.CalcVShape(Trans, vshape);
    vshape *= 2.;

    mfem::DenseMatrix vshapeT(vshape, 't');
    mfem::DenseMatrix tmp(ndof);
    Mult(vshape, vshapeT, tmp);

    mfem::Vector FaceAreaSquareInv(ndof);
    tmp.GetDiag(FaceAreaSquareInv);
    mfem::Vector FaceArea(ndof);

    for (int i = 0; i < ndof; i++)
        FaceArea(i) = 1. / std::sqrt(FaceAreaSquareInv(i));

    vshape.LeftScaling(FaceArea);
    vshapeT.RightScaling(FaceArea);

    // Compute k_{ii}
    mfem::DenseMatrix nk(ndof, dim);
    Mult(vshape, mq, nk);

    mfem::DenseMatrix nkn(ndof);
    Mult(nk, vshapeT, nkn);

    // this is right for grid-aligned permeability, maybe not for full tensor?
    mfem::Vector k(ndof);
    nkn.GetDiag(k);

    // here assume the input is k^{-1};
    mfem::Vector mii(ndof);
    for (int i = 0; i < ndof; i++)
        // Trans.Weight()/FaceArea(i)=Volume/face area=h (for rectangular grid)
        mii(i) = (Trans.Weight() / FaceArea(i)) * k(i) / FaceArea(i) / 2;
    elmat.Diag(mii.GetData(), ndof);
}

std::vector<int> BooleanMult(const SparseMatrix& mat, const std::vector<int>& vec)
{
    std::vector<int> out(mat.Rows(), 0);
    for (int i = 0; i < mat.Rows(); i++)
    {
        for (int j = mat.GetIndptr()[i]; j < mat.GetIndptr()[i + 1]; j++)
        {
            if (vec[mat.GetIndices()[j]])
            {
                out[i] = 1;
                break;
            }
        }
    }
    return out;
}

/**
   @brief A utility class for working with the SPE10 data set.

   The SPE10 data set can be found at: http://www.spe.org/web/csp/datasets/set02.htm
*/
class InversePermeabilityFunction
{
public:
    enum SliceOrientation {NONE, XY, XZ, YZ};
    static void SetNumberCells(int Nx_, int Ny_, int Nz_);
    static void SetMeshSizes(double hx, double hy, double hz);
    static void Set2DSlice(SliceOrientation o, int npos );
    static void ReadPermeabilityFile(const std::string& fileName);
    static void ReadPermeabilityFile(const std::string& fileName, MPI_Comm comm);
    static void InversePermeability(const mfem::Vector& x, mfem::Vector& val);
    static double InvNorm2(const mfem::Vector& x);
    static void ClearMemory();
private:
    static int Nx;
    static int Ny;
    static int Nz;
    static double hx;
    static double hy;
    static double hz;
    static double* inversePermeability;
    static SliceOrientation orientation;
    static int npos;
};


void InversePermeabilityFunction::SetNumberCells(int Nx_, int Ny_, int Nz_)
{
    Nx = Nx_;
    Ny = Ny_;
    Nz = Nz_;
}

void InversePermeabilityFunction::SetMeshSizes(double hx_, double hy_,
                                               double hz_)
{
    hx = hx_;
    hy = hy_;
    hz = hz_;
}

void InversePermeabilityFunction::Set2DSlice(SliceOrientation o, int npos_ )
{
    orientation = o;
    npos = npos_;
}

void InversePermeabilityFunction::ReadPermeabilityFile(const std::string& fileName)
{
    std::ifstream permfile(fileName.c_str());

    if (!permfile.is_open())
    {
        std::cerr << "Error in opening file " << fileName << std::endl;
        mfem::mfem_error("File does not exist");
    }

    inversePermeability = new double [3 * Nx * Ny * Nz];
    double* ip = inversePermeability;
    double tmp;
    for (int l = 0; l < 3; l++)
    {
        for (int k = 0; k < Nz; k++)
        {
            for (int j = 0; j < Ny; j++)
            {
                for (int i = 0; i < Nx; i++)
                {
                    permfile >> *ip;
                    *ip = 1. / (*ip);
                    ip++;
                }
                for (int i = 0; i < 60 - Nx; i++)
                    permfile >> tmp; // skip unneeded part
            }
            for (int j = 0; j < 60 - Ny; j++) //220
                for (int i = 0; i < 60; i++)
                    permfile >> tmp;  // skip unneeded part
        }

        if (l < 2) // if not processing Kz, skip unneeded part
            for (int k = 0; k < 7 - Nz; k++) //85
                for (int j = 0; j < 60; j++) //220
                    for (int i = 0; i < 60; i++)
                        permfile >> tmp;
    }

}

void InversePermeabilityFunction::ReadPermeabilityFile(const std::string& fileName,
                                                       MPI_Comm comm)
{
    int num_procs, myid;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    mfem::StopWatch chrono;

    chrono.Start();
    if (myid == 0)
        ReadPermeabilityFile(fileName);
    else
        inversePermeability = new double [3 * Nx * Ny * Nz];
    chrono.Stop();

    if (myid == 0)
        std::cout << "Permeability file read in " << chrono.RealTime() << ".s \n";

    chrono.Clear();

    chrono.Start();
    MPI_Bcast(inversePermeability, 3 * Nx * Ny * Nz, MPI_DOUBLE, 0, comm);
    chrono.Stop();

    if (myid == 0)
        std::cout << "Permeability field distributed in " << chrono.RealTime() << ".s \n";

}

void InversePermeabilityFunction::InversePermeability(const mfem::Vector& x,
                                                      mfem::Vector& val)
{
    val.SetSize(x.Size());

    unsigned int i = 0, j = 0, k = 0;

    switch (orientation)
    {
        case NONE:
            i = Nx - 1 - (int)floor(x[0] / hx / (1. + 3e-16));
            j = (int)floor(x[1] / hy / (1. + 3e-16));
            k = Nz - 1 - (int)floor(x[2] / hz / (1. + 3e-16));
            break;
        case XY:
            i = Nx - 1 - (int)floor(x[0] / hx / (1. + 3e-16));
            j = (int)floor(x[1] / hy / (1. + 3e-16));
            k = npos;
            break;
        case XZ:
            i = Nx - 1 - (int)floor(x[0] / hx / (1. + 3e-16));
            j = npos;
            k = Nz - 1 - (int)floor(x[2] / hz / (1. + 3e-16));
            break;
        case YZ:
            i = npos;
            j = (int)floor(x[1] / hy / (1. + 3e-16));
            k = Nz - 1 - (int)floor(x[2] / hz / (1. + 3e-16));
            break;
        default:
            mfem::mfem_error("InversePermeabilityFunction::InversePermeability");
    }

    val[0] = inversePermeability[Ny * Nx * k + Nx * j + i];
    val[1] = inversePermeability[Ny * Nx * k + Nx * j + i + Nx * Ny * Nz];

    if (orientation == NONE)
        val[2] = inversePermeability[Ny * Nx * k + Nx * j + i + 2 * Nx * Ny * Nz];

}

double InversePermeabilityFunction::InvNorm2(const mfem::Vector& x)
{
    mfem::Vector val(3);
    InversePermeability(x, val);
    return 1.0 / val.Norml2();
}

void InversePermeabilityFunction::ClearMemory()
{
    delete[] inversePermeability;
}

int InversePermeabilityFunction::Nx(60);
int InversePermeabilityFunction::Ny(220);
int InversePermeabilityFunction::Nz(85);
double InversePermeabilityFunction::hx(20);
double InversePermeabilityFunction::hy(10);
double InversePermeabilityFunction::hz(2);
double* InversePermeabilityFunction::inversePermeability(NULL);
InversePermeabilityFunction::SliceOrientation InversePermeabilityFunction::orientation(
    InversePermeabilityFunction::NONE );
int InversePermeabilityFunction::npos(-1);

SparseMatrix TableToMatrix(const mfem::Table& table)
{
    const int height = table.Size();
    const int width = table.Width();
    const int nnz = table.Size_of_connections();


    std::vector<int> indptr(height + 1);
    std::vector<int> indices(nnz);
    std::vector<double> data(nnz, 1.0);

    std::copy_n(table.GetI(), height + 1, std::begin(indptr));
    std::copy_n(table.GetJ(), nnz, std::begin(indices));

    return SparseMatrix(std::move(indptr), std::move(indices), std::move(data), height, width);
}

/**
   @brief Darcy's flow problem discretized in finite volume (TPFA)
*/
class DarcyProblem
{
public:
    DarcyProblem(MPI_Comm comm, int nDimensions, const std::vector<int>& ess_v_attr);
    DarcyProblem(const mfem::ParMesh& pmesh, const std::vector<int> &ess_v_attr);

    Graph GetFVGraph(int coarsening_factor, bool use_local_weight, SparseMatrix W_block = {});

    const Vector& GetVertexRHS() const
    {
        return rhs_u_;
    }
    double CellVolume() const
    {
        assert(pmesh_);
        return pmesh_->GetElementVolume(0);
    }
    const std::vector<int>& GetEssentialVertDofs() const
    {
        return ess_vdofs_;
    }
    void PrintMeshWithPartitioning(mfem::Array<int>& partition);
    void VisSetup(mfem::socketstream& vis_v, mfem::Vector& vec, double range_min,
                  double range_max, const std::string& caption = "", int coef = 0) const;
    void VisUpdate(mfem::socketstream& vis_v, mfem::Vector& vec) const;
    void CartPart(mfem::Array<int>& partitioning, int nz,
                  const mfem::Array<int>& coarsening_factor,
                  const mfem::Array<int>& isolated_vertices) const;
protected:
    void BuildReservoirGraph();
    void InitGraph();
    void ComputeGraphWeight();

    unique_ptr<mfem::ParMesh> pmesh_;

    std::vector<int> num_procs_xyz_;
    unique_ptr<mfem::RT_FECollection> sigma_fec_;
    unique_ptr<mfem::L2_FECollection> u_fec_;
    unique_ptr<mfem::ParFiniteElementSpace> sigma_fes_;
    unique_ptr<mfem::ParFiniteElementSpace> u_fes_;

    unique_ptr<mfem::GridFunction> coeff_gf_;

    smoothg::SparseMatrix vertex_edge_;
    ParMatrix edge_trueedge_;

    std::vector<double> weight_;
    std::vector<Vector> local_weight_;

    unique_ptr<mfem::VectorCoefficient> kinv_vector_;
    unique_ptr<mfem::Coefficient> kinv_scalar_;

    Vector rhs_sigma_;
    Vector rhs_u_;

    std::vector<int> ess_v_attr_;
    std::vector<int> ess_vdofs_;
    int num_ess_vdof_;

    ParMatrix edge__mfem_dof_;

    mutable mfem::ParGridFunction u_fes_gf_;

    int myid_;
    MPI_Comm comm_;
};

DarcyProblem::DarcyProblem(MPI_Comm comm, int nDimensions, const std::vector<int> &ess_v_attr)
    : comm_(comm), ess_v_attr_(ess_v_attr), num_ess_vdof_(0)
{
    int num_procs;
    MPI_Comm_size(comm_, &num_procs);
    MPI_Comm_rank(comm_, &myid_);

    num_procs_xyz_.resize(nDimensions, 1);
    num_procs_xyz_[0] = std::sqrt(num_procs);
    num_procs_xyz_[1] = std::sqrt(num_procs);
}

DarcyProblem::DarcyProblem(const mfem::ParMesh& pmesh, const std::vector<int>& ess_v_attr)
    : DarcyProblem(pmesh.GetComm(), pmesh.Dimension(), ess_v_attr)
{
    pmesh_ = make_unique<mfem::ParMesh>(pmesh, false);
    InitGraph();
    kinv_scalar_ = make_unique<mfem::ConstantCoefficient>(1.0);
    ComputeGraphWeight();
}

Graph DarcyProblem::GetFVGraph(int coarsening_factor, bool use_local_weight,
                               SparseMatrix W_block)
{
    auto partitioning = PartitionAAT(vertex_edge_, weight_, coarsening_factor,
                                     2., true, true, ess_vdofs_);
    if (use_local_weight)
    {
        std::cout << "use_local_weight is currently not supported! \n";
    }
//    else
    {
        Graph out(vertex_edge_, edge_trueedge_, partitioning, weight_, W_block);
        vertex_edge_ = SparseMatrix();
        edge_trueedge_ = ParMatrix();
        std::vector<double>().swap(weight_);
        std::vector<Vector>().swap(local_weight_);
        return out;
    }
}

// Keep only boundary faces associated with essential pressure condition
// For these faces, add the associated attribute as a (ghost) element
void DarcyProblem::BuildReservoirGraph()
{
    SparseMatrix edge_bdr_att = GenerateBoundaryAttributeTable(*pmesh_);
    assert(edge_bdr_att.Cols() == ess_v_attr_.size());

    const mfem::Table& v_e_table = pmesh_->Dimension() == 2 ?
                pmesh_->ElementToEdgeTable() : pmesh_->ElementToFaceTable();
    SparseMatrix mesh_e_v = TableToMatrix(v_e_table).Transpose();

    std::vector<int> free_faces = BooleanMult(edge_bdr_att, ess_v_attr_);

    const int num_elems = mesh_e_v.Cols();
    const int num_faces = mesh_e_v.Rows();
    const int num_ess_faces = edge_bdr_att.nnz() - Sum(free_faces, 0);

    SparseMatrix bdr_att_edge = edge_bdr_att.Transpose();
    std::vector<int> ess_vdof_map(ess_v_attr_.size(), -1);
    for (int i = 0; i < ess_v_attr_.size(); ++i)
    {
        if (bdr_att_edge.RowSize(i) && ess_v_attr_[i])
        {
            ess_vdof_map[i] = num_elems + num_ess_vdof_++;
        }
    }

    CooMatrix v_e(num_elems + num_ess_vdof_, num_faces - num_ess_faces);
    CooMatrix edof_select(num_faces - num_ess_faces, num_faces);
    int edge_counter = 0;
    for (int i = 0; i < num_faces; ++i)
    {
        if (edge_bdr_att.RowSize(i))
        {
            assert(edge_bdr_att.RowSize(i) == 1);
            int bdr_attr = edge_bdr_att.GetIndices(i)[0];
            if (ess_v_attr_[bdr_attr])
            {
                assert(ess_vdof_map[bdr_attr] != -1);
                v_e.Add(ess_vdof_map[bdr_attr], edge_counter, 1.0);
            }
            else
            {
                continue;
            }
        }

        for (auto j : mesh_e_v.GetIndices(i))
        {
            v_e.Add(j, edge_counter, 1.0);
        }

        edof_select.Add(edge_counter, i, 1.0);

        edge_counter++;
    }
    assert(edge_counter == num_faces - num_ess_faces);
    vertex_edge_ = v_e.ToSparse();

    edge__mfem_dof_ = ParMatrix(comm_, edof_select.ToSparse());
    ParMatrix d__td = ParMatrixToParMatrix(*sigma_fes_->Dof_TrueDof_Matrix());
    ParMatrix edge__mfem_td = edge__mfem_dof_ * d__td;
    edge_trueedge_ = MakeEntityTrueEntity(edge__mfem_td * edge__mfem_td.Transpose());

    ess_vdofs_.resize(num_ess_vdof_);
    std::iota(ess_vdofs_.begin(), ess_vdofs_.end(), num_elems);
}

void DarcyProblem::InitGraph()
{
    sigma_fec_ = make_unique<mfem::RT_FECollection>(0, pmesh_->SpaceDimension());
    sigma_fes_ = make_unique<mfem::ParFiniteElementSpace>(pmesh_.get(), sigma_fec_.get());

    u_fec_ = make_unique<mfem::L2_FECollection>(0, pmesh_->SpaceDimension());
    u_fes_ = make_unique<mfem::ParFiniteElementSpace>(pmesh_.get(), u_fec_.get());
    coeff_gf_ = make_unique<mfem::GridFunction>(u_fes_.get());

    BuildReservoirGraph();

    rhs_sigma_.SetSize(vertex_edge_.Cols(), 0.0);
    rhs_u_.SetSize(vertex_edge_.Rows(), 0.0);
}

void DarcyProblem::ComputeGraphWeight()
{
    // Construct "finite volume mass" matrix
    mfem::ParBilinearForm a(sigma_fes_.get());
    if (kinv_vector_)
    {
        assert(kinv_scalar_ == nullptr);
        a.AddDomainIntegrator(new FiniteVolumeMassIntegrator(*kinv_vector_));
    }
    else
    {
        assert(kinv_scalar_);
        a.AddDomainIntegrator(new FiniteVolumeMassIntegrator(*kinv_scalar_));
    }

    // Compute element mass matrices, assemble mass matrix and edge weight
    mfem::Vector a_diag;
    a.ComputeElementMatrices();
    a.Assemble();
    a.Finalize();
    a.SpMat().GetDiag(a_diag);

    VectorView a_diag_view(a_diag.GetData(), a_diag.Size());
    Vector weight_inv = edge__mfem_dof_.GetDiag().Mult(a_diag_view); //TODO: no diag

    weight_.resize(weight_inv.size());
    SparseMatrix e_v = vertex_edge_.Transpose();

    for (int i = 0; i < weight_inv.size(); ++i)
    {
        bool is_ess_attr = false;
        for (auto&& j : e_v.GetIndices(i))
        {
            if (j >= (vertex_edge_.Rows() - num_ess_vdof_))
                is_ess_attr = true;
        }
        assert(mfem::IsFinite(weight_inv[i]) && weight_inv[i] != 0.0);
        weight_[i] = 1.0 / weight_inv[i];
        if (is_ess_attr)
        {
            weight_[i] /= 2.0;
        }
    }

    // Store element mass matrices and local edge weights (TODO: map old to new)
    local_weight_.resize(vertex_edge_.Rows());
    mfem::DenseMatrix M_el_i;
    for (int i = 0; i < pmesh_->GetNE(); i++)
    {
        a.ComputeElementMatrix(i, M_el_i);
        Vector& local_weight_i = local_weight_[i];
        local_weight_i.SetSize(M_el_i.Height());
        for (int j = 0; j < local_weight_i.size(); j++)
        {
            local_weight_i[j] = 1.0 / M_el_i(j, j);
        }
    }
    for (int i = 0; i < pmesh_->GetNE(); i++)
    {
        a.ComputeElementMatrix(i, M_el_i);
        Vector& local_weight_i = local_weight_[i];
        local_weight_i.SetSize(M_el_i.Height());
        for (int j = 0; j < local_weight_i.size(); j++)
        {
            local_weight_i[j] = 1.0 / M_el_i(j, j);
        }
    }
}

void DarcyProblem::PrintMeshWithPartitioning(mfem::Array<int>& partition)
{
    std::stringstream fname;
    fname << "mesh0.mesh." << std::setfill('0') << std::setw(6) << myid_;
    std::ofstream ofid(fname.str().c_str());
    ofid.precision(8);
    pmesh_->PrintWithPartitioning(partition.GetData(), ofid, 1);
}

void DarcyProblem::VisSetup(mfem::socketstream& vis_v, mfem::Vector& vec, double range_min,
                            double range_max, const std::string& caption, int coef) const
{
    u_fes_gf_.MakeRef(u_fes_.get(), vec.GetData());

    const char vishost[] = "localhost";
    const int  visport   = 19916;
    vis_v.open(vishost, visport);
    vis_v.precision(8);

    vis_v << "parallel " << pmesh_->GetNRanks() << " " << myid_ << "\n";
    vis_v << "solution\n" << *pmesh_ << u_fes_gf_;
    vis_v << "window_size 800 800\n";//500
    vis_v << "window_title 'vertex space unknown'\n";
    vis_v << "autoscale off\n"; // update value-range; keep mesh-extents fixed
    if (range_max > range_min)
    {
        vis_v << "valuerange " << range_min << " " << range_max <<
                 "\n"; // update value-range; keep mesh-extents fixed
    }

    if (pmesh_->SpaceDimension() == 2)
    {
        vis_v << "view 0 0\n"; // view from top
        vis_v << "keys jl\n";  // turn off perspective and light
//        vis_v << "keys ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]\n";  // increase size
        vis_v << "keys ]]]]]]]]]]]]]]]]]]]]\n";  // increase size
    }
    else
    {
        if (!coef)
        {
            vis_v << "keys i\n";  // see interior
        }
        vis_v << "keys [[[[\n";  // decrease size
//        vis_v << "keys i\n";  // see interior
//        vis_v << "keys ]]]]]\n";  // increase size
    }

    if (coef)
    {
        vis_v << "keys fL\n";  // smoothing and logarithmic scale
    }

    vis_v << "keys c\n";         // show colorbar and mesh
    //vis_v << "pause\n"; // Press space to play!

    //    if (!caption.empty())
    //    {
    //        vis_v << "plot_caption '" << caption << "'\n";
    //    }

    MPI_Barrier(pmesh_->GetComm());

    vis_v << "keys S\n";         //Screenshot

    MPI_Barrier(pmesh_->GetComm());
}

void DarcyProblem::VisUpdate(mfem::socketstream& vis_v, mfem::Vector& vec) const
{
    u_fes_gf_.MakeRef(u_fes_.get(), vec.GetData());

    vis_v << "parallel " << pmesh_->GetNRanks() << " " << myid_ << "\n";
    vis_v << "solution\n" << *pmesh_ << u_fes_gf_;

//    MPI_Barrier(pmesh_->GetComm());

//    vis_v << "keys S\n";         //Screenshot

//    MPI_Barrier(pmesh_->GetComm());
}

void DarcyProblem::CartPart(mfem::Array<int>& partitioning, int nz,
                            const mfem::Array<int>& coarsening_factor,
                            const mfem::Array<int>& isolated_vertices) const
{
    const int nDimensions = num_procs_xyz_.size();

    mfem::Array<int> nxyz(nDimensions);
    nxyz[0] = 60 / num_procs_xyz_[0] / coarsening_factor[0];
    nxyz[1] = 220 / num_procs_xyz_[1] / coarsening_factor[1];
    if (nDimensions == 3)
        nxyz[2] = nz / num_procs_xyz_[2] / coarsening_factor[2];

    for (int& i : nxyz)
    {
        i = std::max(1, i);
    }

    mfem::Array<int> cart_part(pmesh_->CartesianPartitioning(nxyz.GetData()),
                               pmesh_->GetNE());
    cart_part.MakeDataOwner();

    partitioning.SetSize(vertex_edge_.Rows());
    for (int i = 0; i < cart_part.Size(); i++)
    {
        partitioning[i] = cart_part[i];
    }
    int num_parts = cart_part.Max() + 1;
    for (int i = 0; i < isolated_vertices.Size(); i++)
    {
        partitioning[isolated_vertices[i]] = num_parts++;
    }
    //    for (int i = cart_part.Size(); i < partitioning.Size(); i++)
    //    {
    //        partitioning[i] = num_parts++;
    //    }
}

class SPE10Problem : public DarcyProblem
{
public:
    SPE10Problem(const char* permFile, int nDimensions, int spe10_scale,
                 int slice, const std::vector<int>& ess_v_attr,
                 int nz = 15, int well_height = 5, double inject_rate = 1.0,
                 double bottom_hole_pressure = 0.0);

    ~SPE10Problem()
    {
        InversePermeabilityFunction::ClearMemory();
    }
    static double CellVolume(int nDimensions)
    {
        return (nDimensions == 2) ? (20.0 * 10.0) : (20.0 * 10.0 * 2.0);
    }
    using DarcyProblem::CellVolume;
    const std::vector<Well>& GetWells()
    {
        return well_manager_->GetWells();
    }
    void setup_five_spot_pattern(int nz, int nDim, WellManager& well_manager, int well_height,
                                 double injection_rate, double bottom_hole_pressure = 0.0);
private:
    void SetupMeshAndCoeff(const char* permFile, int nDimensions,
                           int spe10_scale, int nz, int slice);

    unique_ptr<WellManager> well_manager_;
    mfem::Vector bbmin_;
    mfem::Vector bbmax_;
};

void SPE10Problem::SetupMeshAndCoeff(const char* permFile, int nDimensions,
                                     int spe10_scale, int nz, int slice)
{
    mfem::Array<int> N(3);
    N[0] = 12 * spe10_scale; // 60
    N[1] = 44 * spe10_scale; // 220
    N[2] = 17 * spe10_scale; // 85

    // SPE10 grid cell dimensions
    mfem::Vector h(3);
    h(0) = 20.0;
    h(1) = 10.0;
    h(2) = 2.0;

    unique_ptr<mfem::Mesh> mesh;
    if (nDimensions == 2)
    {
        mesh = make_unique<mfem::Mesh>(N[0], N[1], mfem::Element::QUADRILATERAL,
                                       1, h(0) * N[0], h(1) * N[1]);
        mesh->UniformRefinement();
    }
    else
    {
        mesh = make_unique<mfem::Mesh>(N[0], N[1], nz, mfem::Element::HEXAHEDRON,
                                       1, h(0) * N[0], h(1) * N[1], h(2) * nz);
    }
    mesh->GetBoundingBox(bbmin_, bbmax_, 1);

//    int* partition = mesh->CartesianPartitioning(num_procs_xyz_.data());
    pmesh_ = make_unique<mfem::ParMesh>(comm_, *mesh);
//    delete partition;

    using IPF = InversePermeabilityFunction;
    IPF::SetNumberCells(N[0], N[1], N[2]);
    IPF::SetMeshSizes(h(0), h(1), h(2));
    if (nDimensions == 2)
    {
        IPF::Set2DSlice(IPF::XY, slice);
    }
    IPF::ReadPermeabilityFile(permFile, comm_);
    kinv_vector_ = make_unique<mfem::VectorFunctionCoefficient>(nDimensions, IPF::InversePermeability);
}

SPE10Problem::SPE10Problem(const char* permFile, int nDimensions, int spe10_scale,
                           int slice, const std::vector<int>& ess_v_attr, int nz,
                           int well_height, double inject_rate, double bottom_hole_pressure)
    : DarcyProblem(MPI_COMM_WORLD, nDimensions, ess_v_attr)
{
    SetupMeshAndCoeff(permFile, nDimensions, spe10_scale, nz, slice);

    // Build wells (Peaceman's five-spot pattern)
    well_manager_ = make_unique<WellManager>(*pmesh_, *kinv_vector_);
    setup_five_spot_pattern(nz, nDimensions, *well_manager_, well_height,
                            inject_rate, bottom_hole_pressure);

    local_weight_.reserve(pmesh_->GetNE() + well_manager_->GetNumInjectors());
    InitGraph();

//    mfem::FunctionCoefficient coeff_k(InversePermeabilityFunction::InvNorm2);
//    coeff_gf_->ProjectCoefficient(coeff_k);
//    mfem::socketstream soc;
//    VisSetup(soc, *coeff_gf_, 0., 0., "");

    ComputeGraphWeight();

//    if (well_height > 0)
//    {
//        auto e_te = ParMatrixToParMatrix(*(sigma_fes_->Dof_TrueDof_Matrix()));
//        local_weight_.resize(vertex_edge_.Rows() + well_manager_->GetNumInjectors());
//        edge_trueedge_ = IntegrateReservoirAndWellModels(
//                    well_manager_->GetWells(), vertex_edge_, weight_,
//                    local_weight_, e_te, rhs_sigma_, rhs_u_);

//        std::vector<int> well_marker;
//        ExtendEdgeBoundaryattr2(well_manager_->GetWells(), edge_bdr_att, well_marker);

//        std::fill(well_marker.begin(), well_marker.end(), 0);
//        mfem::Array<int> ess_edof_marker_tmp;
//        sigma_fes_->GetEssentialVDofs(ess_v_attr, ess_edof_marker_tmp);
//        std::copy(ess_edof_marker_tmp.begin(), ess_edof_marker_tmp.end(), well_marker.begin());
//        swap(ess_edof_marker_, well_marker);
//    }
//    else
    {
        rhs_u_ = -1.0 * CellVolume();
        for (int i = 0; i < num_ess_vdof_; ++i)
        {
            rhs_u_[rhs_u_.size()-1-i] = 0.0;
        }
    }

}

void SPE10Problem::setup_five_spot_pattern(int nz, int nDim, WellManager& well_manager,
                                           int well_height, double injection_rate,
                                           double bottom_hole_pressure)
{
    if (nDim == 2)
    {
        well_height = 1;
    }
    else
    {
        well_height = std::min(nz, well_height);
    }

    const int num_wells = 5;

    std::vector<std::vector<int>> producer_well_cells(num_wells - 1);
    std::vector<std::vector<int>> injector_well_cells(1);

    mfem::DenseMatrix point(nDim, num_wells);
    // Producers
    point(0, 0) = bbmin_[0] + 1.0;
    point(1, 0) = bbmin_[1] + 1.0;

    point(0, 1) = bbmax_[0] - 1.0;
    point(1, 1) = bbmin_[1] + 1.0;

    point(0, 2) = bbmin_[0] + 1.0;
    point(1, 2) = bbmax_[1] - 1.0;

    point(0, 3) = bbmax_[0] - 1.0;
    point(1, 3) = bbmax_[1] - 1.0;

    // Injector, Shifted to avoid middle,
    // Since probably processor boundary
    point(0, 4) = ((bbmax_[0] - bbmin_[0]) / 2.0) + 1.0;
    point(1, 4) = ((bbmax_[1] - bbmin_[1]) / 2.0) + 1.0;

    if (nDim == 3)
    {
        point(2, 0) = bbmin_[2] + 1.0;
        point(2, 1) = bbmin_[2] + 1.0;
        point(2, 2) = bbmin_[2] + 1.0;
        point(2, 3) = bbmin_[2] + 1.0;
        point(2, 4) = bbmax_[2] - 1.0;
    }

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
        if (nDim == 3)
        {
            for (int i = 0; i < 4; ++i)
            {
                point(2, i) += 2.0;
            }
            for (int i = 4; i < 5; ++i)
            {
                point(2, i) -= 2.0;
            }
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

/// scalar normal distribution
class NormalDistribution
{
public:
    NormalDistribution(double mean = 0.0, double stddev = 1.0, int seed = 0)
        : generator_(seed), dist_(mean, stddev)
    { }

    double Sample()
    {
        double out = dist_(generator_);
        return out;
    }

private:
    std::mt19937 generator_;
    std::normal_distribution<double> dist_;
};

class LognormalProblem : public DarcyProblem
{
public:
    LognormalProblem(int nDimensions, int num_ser_ref, int num_par_ref,
                     double correlation_length, const std::vector<int>& ess_e_attr);

    ~LognormalProblem() { }
private:
    void SetupMesh(int nDimensions, int num_ser_ref, int num_par_ref);
    void SetupCoeff(int nDimensions, double correlation_length, int more_ref);

    unique_ptr<mfem::ParMesh> pmesh_c_;
};

LognormalProblem::LognormalProblem(int nDimensions, int num_ser_ref,
                                   int num_par_ref, double correlation_length,
                                   const std::vector<int>& ess_v_attr)
    : DarcyProblem(MPI_COMM_WORLD, nDimensions, ess_v_attr)
{
    SetupMesh(nDimensions, num_ser_ref, num_par_ref);
    InitGraph();

    int more_ref = num_par_ref ;
    SetupCoeff(nDimensions, correlation_length, more_ref);
    ComputeGraphWeight();

    rhs_u_ = -1.0 * CellVolume() / 2.5;
    for (int i = 0; i < num_ess_vdof_; ++i)
    {
        rhs_u_[rhs_u_.size()-1-i] = 0.0;
    }
    pmesh_.reset();
    sigma_fec_.reset();
    sigma_fes_.reset();
    u_fec_.reset();
    u_fes_.reset();
    InversePermeabilityFunction::ClearMemory();
}

void LognormalProblem::SetupMesh(int nDimensions, int num_ser_ref, int num_par_ref)
{
    const int N = std::pow(2, num_ser_ref);
    unique_ptr<mfem::Mesh> mesh;
//    if (nDimensions == 2)
//    {
//        mesh = make_unique<mfem::Mesh>(N, N, mfem::Element::QUADRILATERAL, 1);
//    }
//    else
//    {
//        mesh = make_unique<mfem::Mesh>(N, N, N, mfem::Element::HEXAHEDRON, 1);
//    }

    std::ifstream imesh("egg_model.mesh");
    mesh = make_unique<mfem::Mesh>(imesh, 1, 1);

    pmesh_ = make_unique<mfem::ParMesh>(comm_, *mesh);
    for (int i = 0; i < 0; i++)
    {
        pmesh_->UniformRefinement();
    }
    pmesh_c_ = make_unique<mfem::ParMesh>(*pmesh_);
    for (int i = 0; i < num_par_ref ; i++)
    {
        pmesh_->UniformRefinement();
    }
}

void LognormalProblem::SetupCoeff(int nDimensions, double correlation_length, int more_ref)
{
    double nu_parameter = nDimensions == 2 ? 1.0 : 0.5;
    double kappa = std::sqrt(2.0 * nu_parameter) / correlation_length;

    double ddim = static_cast<double>(nDimensions);
    double scalar_g = std::pow(4.0 * M_PI, ddim / 4.0) * std::pow(kappa, nu_parameter) *
            std::sqrt( std::tgamma(nu_parameter + ddim / 2.0) / tgamma(nu_parameter) );

//    std::vector<int> ess_v_attr(ess_v_attr_.size(), 0);

//    mfem::L2_FECollection u_fec(0, pmesh_c_->SpaceDimension());
//    mfem::ParFiniteElementSpace u_fes(pmesh_c_.get(), &u_fec);

//    SparseMatrix P = SparseIdentity(pmesh_c_->GetNE());
//    for (int i = 0; i < more_ref; i++)
//    {
//        pmesh_c_->UniformRefinement();
//        auto& P_l = (const mfem::SparseMatrix&)*u_fes.GetUpdateOperator();
//        P = SparseToSparse(P_l).Mult(P);
//    }
//    SparseMatrix Proj = P.Transpose();

//    auto PTP = Proj.Mult(P);
//    Proj.InverseScaleRows(PTP.GetDiag());

//    DarcyProblem darcy_problem(*pmesh_, ess_v_attr);
//    SparseMatrix W_block = SparseIdentity(pmesh_->GetNE());
//    double cell_vol = pmesh_->GetElementVolume(0);
//    W_block = cell_vol * kappa * kappa;
//    MixedMatrix mgL(darcy_problem.GetFVGraph(50, false, std::move(W_block)), 0);
//    mgL.AssembleM();

//    NormalDistribution normal_dist(0.0, 1.0, 22 + myid_);
//    Vector rhs(mgL.LocalD().Rows());
////    assert(rhs.size() == 262144);
//    for (int i = 0; i < rhs.size(); ++i)
//    {
////        normal_dist.Sample();
////        normal_dist.Sample();
//        rhs[i] = scalar_g * std::sqrt(cell_vol) * normal_dist.Sample();
//    }

//    SPDSolver solver(mgL);
//    Vector sol = solver.Mult(rhs);

//    for (auto& s : sol)
//    {
//        s = std::exp(s);
//    }

//    std::ofstream ofs("coef.txt");
//    sol.Print("", ofs);

    //==================

//    std::vector<double> coef_from_file = linalgcpp::ReadText("coef.txt");
//    Vector coef(coef_from_file.data(), coef_from_file.size());
//    Vector sol = P.Mult(coef);

//    for (int i = 0; i < coeff_gf_->Size(); ++i)
//    {
//        coeff_gf_->Elem(i) = sol[i];//std::exp(projected_sol[i]);
//    }
//    kinv_scalar_ = make_unique<mfem::GridFunctionCoefficient>(coeff_gf_.get());

    // ======  egg model

    using IPF = InversePermeabilityFunction;
    IPF::SetNumberCells(60, 60, 7);
    IPF::SetMeshSizes(8.0, 8.0, 4.0);
    IPF::ReadPermeabilityFile("egg_perm_27.txt", comm_);
    kinv_vector_ = make_unique<mfem::VectorFunctionCoefficient>(nDimensions, IPF::InversePermeability);

//    kinv_scalar_ = make_unique<mfem::FunctionCoefficient>(IPF::InvNorm2);
//    coeff_gf_->ProjectCoefficient(*kinv_scalar_);

    // ======= end egg model

    mfem::socketstream soc;
//    VisSetup(soc, *coeff_gf_, 0., 0., "", 1);
}

}


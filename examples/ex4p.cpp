//                       MFEM Example 4 - Parallel Version
//
// Compile with: make ex4p
//
// Sample runs:  mpirun -np 4 ex4p -m ../data/square-disc.mesh
//               mpirun -np 4 ex4p -m ../data/star.mesh
//               mpirun -np 4 ex4p -m ../data/beam-tet.mesh
//               mpirun -np 4 ex4p -m ../data/beam-hex.mesh
//               mpirun -np 4 ex4p -m ../data/escher.mesh -o 2 -sc
//               mpirun -np 4 ex4p -m ../data/fichera.mesh -o 2 -hb
//               mpirun -np 4 ex4p -m ../data/fichera-q2.vtk
//               mpirun -np 4 ex4p -m ../data/fichera-q3.mesh -o 2 -sc
//               mpirun -np 4 ex4p -m ../data/square-disc-nurbs.mesh -o 3
//               mpirun -np 4 ex4p -m ../data/beam-hex-nurbs.mesh -o 3
//               mpirun -np 4 ex4p -m ../data/periodic-square.mesh -no-bc
//               mpirun -np 4 ex4p -m ../data/periodic-cube.mesh -no-bc
//               mpirun -np 4 ex4p -m ../data/amr-quad.mesh
//               mpirun -np 4 ex4p -m ../data/amr-hex.mesh -o 2 -sc
//               mpirun -np 4 ex4p -m ../data/amr-hex.mesh -o 2 -hb
//               mpirun -np 4 ex4p -m ../data/star-surf.mesh -o 3 -hb
//
// Description:  This example code solves a simple 2D/3D H(div) diffusion
//               problem corresponding to the second order definite equation
//               -grad(alpha div F) + beta F = f with boundary condition F dot n
//               = <given normal field>. Here, we use a given exact solution F
//               and compute the corresponding r.h.s. f.  We discretize with
//               Raviart-Thomas finite elements.
//
//               The example demonstrates the use of H(div) finite element
//               spaces with the grad-div and H(div) vector finite element mass
//               bilinear form, as well as the computation of discretization
//               error when the exact solution is known. Bilinear form
//               hybridization and static condensation are also illustrated.
//
//               We recommend viewing examples 1-3 before viewing this example.

#include "mfem.hpp"
#include "../src/smoothG.hpp"
#include <fstream>
#include <iostream>

#if SMOOTHG_USE_SAAMGE
#include "saamge.hpp"
#endif

using namespace std;
using namespace mfem;
using namespace smoothg;

class VectordivDomainLFIntegrator : public LinearFormIntegrator
{
    Vector divshape;
    Coefficient& Q;
    int oa, ob;
public:
    /// Constructs a domain integrator with a given Coefficient
    VectordivDomainLFIntegrator(Coefficient& QF, int a = 2, int b = 0)
    // the old default was a = 1, b = 1
    // for simple elliptic problems a = 2, b = -2 is ok
        : Q(QF), oa(a), ob(b) { }

    /// Constructs a domain integrator with a given Coefficient
    VectordivDomainLFIntegrator(Coefficient& QF, const IntegrationRule* ir)
        : LinearFormIntegrator(ir), Q(QF), oa(1), ob(1) { }

    /** Given a particular Finite Element and a transformation (Tr)
       computes the element right hand side element vector, elvect. */
    virtual void AssembleRHSElementVect(const FiniteElement& el,
                                        ElementTransformation& Tr,
                                        Vector& elvect);

    using LinearFormIntegrator::AssembleRHSElementVect;
};
//---------

//------------------
void VectordivDomainLFIntegrator::AssembleRHSElementVect(
    const FiniteElement& el, ElementTransformation& Tr,
    Vector& elvect)//don't need the matrix but the vector
{
    int dof = el.GetDof();

    divshape.SetSize(dof);       // vector of size dof
    elvect.SetSize(dof);
    elvect = 0.0;

    const IntegrationRule* ir = IntRule;
    if (ir == NULL)
    {
        // ir = &IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob + Tr.OrderW());
        ir = &IntRules.Get(el.GetGeomType(), oa * el.GetOrder() + ob);
        // int order = 2 * el.GetOrder() ; // <--- OK for RTk
        // ir = &IntRules.Get(el.GetGeomType(), order);
    }

    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        const IntegrationPoint& ip = ir->IntPoint(i);
        el.CalcDivShape(ip, divshape);

        Tr.SetIntPoint (&ip);
        //double val = Tr.Weight() * Q.Eval(Tr, ip);
        // Chak: Looking at how MFEM assembles in VectorFEDivergenceIntegrator, I think you dont need Tr.Weight() here
        // I think this is because the RT (or other vector FE) basis is scaled by the geometry of the mesh
        double val = Q.Eval(Tr, ip);

        add(elvect, ip.weight * val, divshape, elvect);
        //cout << "elvect = " << elvect << endl;
    }
}

double weight = 1.0;

// Exact solution, F, and r.h.s., f. See below for implementation.
double uFun_ex(const Vector& xt); // Exact Solution
double fFun(const Vector& xt); // Source f
void bfFun_ex(const Vector& xt, Vector& gradf);
void gradfFun_ex(const Vector& xt, Vector& gradf);
void sigmaFun_ex(const Vector& xt, Vector& sigma);
double bTb_ex(const Vector& xt);
void bFun_ex (const Vector& xt, Vector& b);
void Ktilda_ex(const Vector& xt, DenseMatrix& Ktilda);
void bbT_ex(const Vector& xt, DenseMatrix& bbT);
int main(int argc, char* argv[])
{
    // 1. Initialize MPI.
    int num_procs, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    // 2. Parse command-line options.
    //   const char *mesh_file = "../data/beam-tet.mesh";
    int order = 1;
    bool set_bc = true;
    bool static_cond = false;
    bool hybridization = false;
    bool use_saamge = false;
    bool visualization = 1;
    int par_ref_levels = 2;

    OptionsParser args(argc, argv);
    //   args.AddOption(&mesh_file, "-m", "--mesh",
    //                  "Mesh file to use.");
    args.AddOption(&order, "-o", "--order",
                   "Finite element order (polynomial degree).");
    args.AddOption(&set_bc, "-bc", "--impose-bc", "-no-bc", "--dont-impose-bc",
                   "Impose or not essential boundary conditions.");
    //   args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the exact"
    //                  " solution.");
    args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                   "--no-static-condensation", "Enable static condensation.");
    args.AddOption(&hybridization, "-hb", "--hybridization", "-no-hb",
                   "--no-hybridization", "Enable hybridization.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.AddOption(&par_ref_levels, "-r", "--ref",
                   "Number of parallel refinement steps.");
    args.AddOption(&use_saamge, "-sa", "--saamge", "-no-sa",
                   "--no-saamge", "Enable SA-AMGe.");
    args.Parse();
    if (!args.Good())
    {
        if (myid == 0)
        {
            args.PrintUsage(cout);
        }
        MPI_Finalize();
        return 1;
    }
    if (myid == 0)
    {
        args.PrintOptions(cout);
    }
    //   kappa = freq * M_PI;

    // 3. Read the (serial) mesh from the given mesh file on all processors.  We
    //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
    //    and volume, as well as periodic meshes with the same code.
    //   Mesh *mesh = new Mesh(mesh_file, 1, 1);
    Mesh* mesh = new Mesh(2, 2, 2, mfem::Element::TETRAHEDRON, 1);
    int dim = mesh->Dimension();
    int sdim = mesh->SpaceDimension();

    // 4. Refine the serial mesh on all processors to increase the resolution. In
    //    this example we do 'ref_levels' of uniform refinement. We choose
    //    'ref_levels' to be the largest number that gives a final mesh with no
    //    more than 1,000 elements.
    {
        int ref_levels = 0;
        //         (int)floor(log(1000./mesh->GetNE())/log(2.)/dim);
        for (int l = 0; l < ref_levels; l++)
        {
            mesh->UniformRefinement();
        }
    }

    // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
    //    this mesh further in parallel to increase the resolution. Once the
    //    parallel mesh is defined, the serial mesh can be deleted. Tetrahedral
    //    meshes need to be reoriented before we can define high-order Nedelec
    //    spaces on them (this is needed in the ADS solver below).
    ParMesh* pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    delete mesh;
    {
        for (int l = 0; l < par_ref_levels; l++)
        {
            pmesh->UniformRefinement();
        }
    }
    pmesh->ReorientTetMesh();

    // 6. Define a parallel finite element space on the parallel mesh. Here we
    //    use the Raviart-Thomas finite elements of the specified order.
    FiniteElementCollection* fec = new RT_FECollection(order - 1, dim);
    ParFiniteElementSpace* fespace = new ParFiniteElementSpace(pmesh, fec);
    HYPRE_Int size = fespace->GlobalTrueVSize();
    if (myid == 0)
    {
        cout << "Number of finite element unknowns: " << size << endl;
    }

    // 7. Determine the list of true (i.e. parallel conforming) essential
    //    boundary dofs. In this example, the boundary conditions are defined
    //    by marking all the boundary attributes from the mesh as essential
    //    (Dirichlet) and converting them to a list of true dofs.
    Array<int> ess_tdof_list;
    if (pmesh->bdr_attributes.Size())
    {
        Array<int> ess_bdr(pmesh->bdr_attributes.Max());
        ess_bdr = 0;
        ess_bdr[0] = 1;
        fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }

    // 8. Set up the parallel linear form b(.) which corresponds to the
    //    right-hand side of the FEM linear system, which in this case is
    //    (f,phi_i) where f is given by the function f_exact and phi_i are the
    //    basis functions in the finite element fespace.
    FunctionCoefficient f(fFun);
    ParLinearForm* b = new ParLinearForm(fespace);
    b->AddDomainIntegrator(new VectordivDomainLFIntegrator(f));
    b->Assemble();

    // 9. Define the solution vector x as a parallel finite element grid function
    //    corresponding to fespace. Initialize x by projecting the exact
    //    solution. Note that only values from the boundary faces will be used
    //    when eliminating the non-homogeneous boundary condition to modify the
    //    r.h.s. vector b.
    ParGridFunction x(fespace);
    VectorFunctionCoefficient F(sdim, sigmaFun_ex);
    x.ProjectCoefficient(F);

    // 10. Set up the parallel bilinear form corresponding to the H(div)
    //     diffusion operator grad alpha div + beta I, by adding the div-div and
    //     the mass domain integrators.
    Coefficient* alpha = new ConstantCoefficient(weight);
    MatrixFunctionCoefficient* beta  = new MatrixFunctionCoefficient(dim, Ktilda_ex);

    //   double h = 1./par_ref_levels;
    //   h *= h;

    double h = 1. / par_ref_levels;
    h *= h;
    if (myid == 0)
        std::cout << "epsilon = " << h << "\n";

    if (myid == 0)
        std::cout << "regularization is ON \n";
    double h_min, h_max, kappa_min, kappa_max;
    pmesh->GetCharacteristics(h_min, h_max, kappa_min, kappa_max);
    if (myid == 0)
        std::cout << "coarse mesh steps: min " << h_min << " max " << h_max << "\n";
    h = h_min * h_min;
    Coefficient* epsilon = new ConstantCoefficient(1.0);
    ParBilinearForm* a = new ParBilinearForm(fespace);
    a->AddDomainIntegrator(new DivDivIntegrator(*alpha));
//    a->AddDomainIntegrator(new VectorFEMassIntegrator(*beta));
    a->AddDomainIntegrator(new VectorFEMassIntegrator(*epsilon));


    StopWatch chrono;
    chrono.Clear(); chrono.Start();

    // 11. Assemble the parallel bilinear form and the corresponding linear
    //     system, applying any necessary transformations such as: parallel
    //     assembly, eliminating boundary conditions, applying conforming
    //     constraints for non-conforming AMR, static condensation,
    //     hybridization, etc.
    FiniteElementCollection* hfec = NULL;
    ParFiniteElementSpace* hfes = NULL;
    if (static_cond)
    {
        a->EnableStaticCondensation();
    }
    else if (hybridization)
    {
        hfec = new DG_Interface_FECollection(order - 1, dim);
        hfes = new ParFiniteElementSpace(pmesh, hfec);
        a->EnableHybridization(hfes, new NormalTraceJumpIntegrator(),
                               ess_tdof_list);
    }
//    a->ComputeElementMatrices();
    a->Assemble();

    HypreParMatrix A;
    Vector B, X;
    a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
    if (myid == 0)
    {
        cout << "\nSystem assembled in " << chrono.RealTime() <<"s \n";
    }
    //   A.Add(1.0, *CCT);
    //   CCT->Add(1.0, A);

    HYPRE_Int glob_size = A.GetGlobalNumRows();
    if (myid == 0)
    {
        cout << "Size of linear system: " << glob_size << endl;
        cout << "NNZ of linear system: " << A.NNZ() << endl;
        cout << "Number of faces: " << pmesh->GetNFaces() << endl;
    }

    SAAMGeParam* saamge_param_ = new SAAMGeParam;
    saamge_param_->first_theta = 1e-4;
    saamge_param_->first_nu_pro = 1;
    saamge_param_->nu_relax = 1;
    saamge_param_->first_coarsen_factor = 8;
    mfem::Table* elem_elem = new mfem::Table(pmesh->ElementToElementTable());

    // Set up preconditioner
    Solver* prec = NULL;
    if (hybridization && use_saamge)
    {
#if SMOOTHG_USE_SAAMGE
        saamge::proc_init(MPI_COMM_WORLD);

        auto multiplier_d_td_ = hfes->Dof_TrueDof_Matrix();
        auto& elem_dof_mat = a->GetHybridization()->GetElemDof();
        mfem::Table elem_dof_tmp = MatrixToTable(elem_dof_mat);
        mfem::Table* elem_dof = new mfem::Table;
        elem_dof->Swap(elem_dof_tmp);

        // Mark dofs that are shared by more than one processor
        saamge::SharedEntityCommunication<mfem::Vector> sec(MPI_COMM_WORLD, *multiplier_d_td_);
        std::vector<saamge::agg_dof_status_t> bdr_dofs(elem_dof->Width(), 0);
        for (unsigned int i = 0; i < bdr_dofs.size(); i++)
        {
            if (sec.NumNeighbors(i) > 1)
                SA_SET_FLAGS(bdr_dofs[i], AGG_ON_PROC_IFACE_FLAG);
        }

        int num_elems = elem_elem->Size();
        std::vector<int> sa_nparts_(saamge_param_->num_levels - 1);
        sa_nparts_[0] = (num_elems / saamge_param_->first_coarsen_factor) + 1;

        bool first_do_aggregates = (saamge_param_->num_levels <= 2 && saamge_param_->do_aggregates);
        auto sa_apr_ = saamge::agg_create_partitioning_fine(
                           A, num_elems, elem_dof, elem_elem, nullptr, bdr_dofs.data(),
                           sa_nparts_.data(), multiplier_d_td_, first_do_aggregates);

        for (int i = 1; i < saamge_param_->num_levels - 1; i++)
            sa_nparts_[i] = sa_nparts_[i - 1] / saamge_param_->coarsen_factor + 1;

        auto& elem_mat = a->GetHybridization()->GetElemMatrices();
        mfem::Array<mfem::DenseMatrix*> elmats(elem_mat.Size());
        for (int i = 0; i < elem_mat.Size(); i++)
            elmats[i] = new mfem::DenseMatrix(*elem_mat[i]);
        auto emp = new saamge::ElementMatrixDenseArray(*sa_apr_, elmats);

        int polynomial_coarse = -1; // we do not have geometric information
        saamge::MultilevelParameters mlp(
            saamge_param_->num_levels - 1, sa_nparts_.data(), saamge_param_->first_nu_pro,
            saamge_param_->nu_pro, saamge_param_->nu_relax, saamge_param_->first_theta,
            saamge_param_->theta, polynomial_coarse, saamge_param_->correct_nulspace,
            saamge_param_->use_arpack, saamge_param_->do_aggregates);
        auto sa_ml_data_ = saamge::ml_produce_data(A, sa_apr_, emp, mlp);
        auto level = saamge::levels_list_get_level(sa_ml_data_->levels_list, 0);

        prec = new saamge::VCycleSolver(level->tg_data, false);
        prec->SetOperator(A);
#else
        if (myid ==0)
        {
            cout << "SA-AMGe is not enabled!\n";
        }
        MPI_Finalize();
        std::abort();
#endif
    }
    else if (hybridization)
    {
        prec = new HypreBoomerAMG(A);
        ((HypreBoomerAMG*)prec)->SetPrintLevel(0);
    }
    else if (use_saamge)
    {
#if SMOOTHG_USE_SAAMGE
        saamge::proc_init(MPI_COMM_WORLD);

        auto multiplier_d_td_ = fespace->Dof_TrueDof_Matrix();

        mfem::Table* elem_dof = new mfem::Table;
        Array<int> dofs;
        elem_dof -> MakeI (pmesh -> GetNE());
        for (int i = 0; i < pmesh -> GetNE(); i++)
        {
           fespace->GetElementVDofs (i, dofs);
           elem_dof -> AddColumnsInRow (i, dofs.Size());
        }
        elem_dof -> MakeJ();
        for (int j = 0; j < pmesh -> GetNE(); j++)
        {
           fespace->GetElementVDofs (j, dofs);
           for (int i = 0; i < dofs.Size(); i++)
           {
              if (dofs[i] < 0)
              {
                  dofs[i] = -1-dofs[i];
              }
           }
           elem_dof -> AddConnections (j, (int *)dofs, dofs.Size());
        }
        elem_dof -> ShiftUpI();

        // Mark dofs that are shared by more than one processor
        saamge::SharedEntityCommunication<mfem::Vector> sec(MPI_COMM_WORLD, *multiplier_d_td_);
        std::vector<saamge::agg_dof_status_t> bdr_dofs(elem_dof->Width(), 0);
        for (unsigned int i = 0; i < bdr_dofs.size(); i++)
        {
            if (sec.NumNeighbors(i) > 1)
                SA_SET_FLAGS(bdr_dofs[i], AGG_ON_PROC_IFACE_FLAG);
        }

        int num_elems = elem_elem->Size();
        std::vector<int> sa_nparts_(saamge_param_->num_levels - 1);
        sa_nparts_[0] = (num_elems / saamge_param_->first_coarsen_factor) + 1;

        bool first_do_aggregates = (saamge_param_->num_levels <= 2 && saamge_param_->do_aggregates);
        auto sa_apr_ = saamge::agg_create_partitioning_fine(
                           A, num_elems, elem_dof, elem_elem, nullptr, bdr_dofs.data(),
                           sa_nparts_.data(), multiplier_d_td_, first_do_aggregates);

        for (int i = 1; i < saamge_param_->num_levels - 1; i++)
            sa_nparts_[i] = sa_nparts_[i - 1] / saamge_param_->coarsen_factor + 1;

        mfem::Array<mfem::DenseMatrix*> elmats(pmesh->GetNE());
        for (int i = 0; i < elmats.Size(); i++)
        {
            elmats[i] = new mfem::DenseMatrix;
            a->ComputeElementMatrix(i, *(elmats[i]));
        }
        auto emp = new saamge::ElementMatrixDenseArray(*sa_apr_, elmats);

        int polynomial_coarse = -1; // we do not have geometric information
        saamge::MultilevelParameters mlp(
            saamge_param_->num_levels - 1, sa_nparts_.data(), saamge_param_->first_nu_pro,
            saamge_param_->nu_pro, saamge_param_->nu_relax, saamge_param_->first_theta,
            saamge_param_->theta, polynomial_coarse, saamge_param_->correct_nulspace,
            saamge_param_->use_arpack, saamge_param_->do_aggregates);
        auto sa_ml_data_ = saamge::ml_produce_data(A, sa_apr_, emp, mlp);
        auto level = saamge::levels_list_get_level(sa_ml_data_->levels_list, 0);

        prec = new saamge::VCycleSolver(level->tg_data, false);
        prec->SetOperator(A);
#else
        if (myid ==0)
        {
            cout << "SA-AMGe is not enabled!\n";
        }
        MPI_Finalize();
        std::abort();
#endif
    }
    else
    {
        ParFiniteElementSpace* prec_fespace =
            (a->StaticCondensationIsEnabled() ? a->SCParFESpace() : fespace);
        if (dim == 2)   { prec = new HypreAMS(A, prec_fespace); }
        else            { prec = new HypreADS(A, prec_fespace); }
    }

    // 12. Define and apply a parallel PCG solver for A X = B with the 2D AMS or
    //     the 3D ADS preconditioners from hypre. If using hybridization, the
    //     system is preconditioned with hypre's BoomerAMG.
    CGSolver* pcg = new CGSolver(A.GetComm());
    pcg->SetOperator(A);
    pcg->SetRelTol(1e-12);
    pcg->SetMaxIter(50000);
    pcg->SetPrintLevel(0);
    pcg->SetPreconditioner(*prec);

    chrono.Clear(); chrono.Start();
    pcg->Mult(B, X);

    if (myid == 0)
    {
        cout << "\nSolving time = " << chrono.RealTime() <<"s \n";
        cout << "Number of iterations = " << pcg->GetNumIterations() << "\n";
    }

    // 13. Recover the parallel grid function corresponding to X. This is the
    //     local finite element solution on each processor.
    a->RecoverFEMSolution(X, *b, x);

    // 14. Compute and print the L^2 norm of the error.
    {
        double err = x.ComputeL2Error(F);
        if (myid == 0)
        {
            cout << "\n|| F_h - F ||_{L^2} = " << err << '\n' << endl;
        }
    }

    // 15. Save the refined mesh and the solution in parallel. This output can
    //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
    {
        ostringstream mesh_name, sol_name;
        mesh_name << "mesh." << setfill('0') << setw(6) << myid;
        sol_name << "sol." << setfill('0') << setw(6) << myid;

        ofstream mesh_ofs(mesh_name.str().c_str());
        mesh_ofs.precision(8);
        pmesh->Print(mesh_ofs);

        ofstream sol_ofs(sol_name.str().c_str());
        sol_ofs.precision(8);
        x.Save(sol_ofs);
    }

    // 16. Send the solution by socket to a GLVis server.
    if (visualization)
    {
        char vishost[] = "localhost";
        int  visport   = 19916;
        socketstream sol_sock(vishost, visport);
        sol_sock << "parallel " << num_procs << " " << myid << "\n";
        sol_sock.precision(8);
        sol_sock << "solution\n" << *pmesh << x << flush;
    }

    // 17. Free the used memory.
    delete pcg;
    delete prec;
    delete hfes;
    delete hfec;
    delete a;
    delete alpha;
    delete beta;
    delete b;
    delete fespace;
    delete fec;
    delete pmesh;

    MPI_Finalize();

    return 0;
}


//double fFun(const Vector& xt)
//{
//    return 0.0;
//}

//void bFun_ex(const Vector& xt, Vector& b )
//{
//    b.SetSize(xt.Size());

//    if (xt.Size() == 3)
//    {
//        b(0) = -xt(1);
//        b(1) = xt(0);
//    }
//    else
//    {
//        b(0) = sin(xt(0)*M_PI)*cos(xt(1)*M_PI)*cos(xt(2)*M_PI);
//        b(1) = -0.5*sin(xt(1)*M_PI)*cos(xt(0)*M_PI)*cos(xt(2)*M_PI);
//        b(2) = -0.5*sin(xt(2)*M_PI)*cos(xt(0)*M_PI)*cos(xt(1)*M_PI);
//    }

//    b(xt.Size()-1) = 1.;
//}

//void bfFun_ex(const Vector& xt, Vector& bf)
//{
//    bf.SetSize(xt.Size());
//    bFun_ex(xt, bf);
//    bf *= fFun(xt);
//}

void sigmaFun_ex(const Vector& xt, Vector& sigma)
{
    Vector b;
    bFun_ex(xt, b);
    sigma.SetSize(xt.Size());
    sigma(xt.Size() - 1) = uFun_ex(xt);
    for (int i = 0; i < xt.Size() - 1; i++)
        sigma(i) = b(i) * sigma(xt.Size() - 1);
}

void Ktilda_ex(const Vector& xt, DenseMatrix& Ktilda)
{
    int nDim = xt.Size();
    Vector b;
    bFun_ex(xt, b);

    double bTbInv = (-1.0 / (b * b));
    Ktilda.Diag(1.0, nDim);
    AddMult_a_VVt(bTbInv, b, Ktilda);
}

double bTb_ex(const Vector& xt)
{
    Vector b;
    bFun_ex(xt, b);
    return 1.*(b * b);
}

void bbT_ex(const Vector& xt, DenseMatrix& bbT)
{
    Vector b;
    bFun_ex(xt, b);
    MultVVt(b, bbT);
}


double uFun_ex(const Vector& xt)
{
    double t = xt(xt.Size() - 1);
    //    return t;
    return sin(t) * exp(t) * xt(0) * xt(1);
}

double fFun(const Vector& xt)
{
    //    double tmp = 0.;
    //    for (int i = 0; i < xt.Size()-1; i++)
    //        tmp += xt(i);
    //    return 1.;//+ (xt.Size()-1-2*tmp) * uFun_ex(xt);
    double t = xt(xt.Size() - 1);
    Vector b;
    bFun_ex(xt, b);
    return (cos(t) * exp(t) + sin(t) * exp(t)) * xt(0) * xt(1) +
           b(0) * sin(t) * exp(t) * xt(1) + b(1) * sin(t) * exp(t) * xt(0) * weight;
}

void bFun_ex(const Vector& xt, Vector& b)
{
    b.SetSize(xt.Size());
    b = 0.;

    b(xt.Size() - 1) = 1.;

    if (xt.Size() == 3)
    {
        b(0) = sin(xt(0) * M_PI) * cos(xt(1) * M_PI);
        b(1) = -sin(xt(1) * M_PI) * cos(xt(0) * M_PI);
    }
    else
    {
        b(0) = sin(xt(0) * M_PI) * cos(xt(1) * M_PI) * cos(xt(2) * M_PI);
        b(1) = -0.5 * sin(xt(1) * M_PI) * cos(xt(0) * M_PI) * cos(xt(2) * M_PI);
        b(2) = -0.5 * sin(xt(2) * M_PI) * cos(xt(0) * M_PI) * cos(xt(1) * M_PI);
    }
}

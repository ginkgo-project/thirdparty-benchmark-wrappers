/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "solvers_wrappers.hpp"


#include <ginkgo/core/base/executor.hpp>


// Tpetra
#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_Vector.hpp>

// Teuchos
#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include <Teuchos_Comm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_RCP.hpp>

// Belos
#include "BelosBlockCGSolMgr.hpp"
#include "BelosConfigDefs.hpp"
#include "BelosLinearProblem.hpp"
#include "BelosTpetraAdapter.hpp"

namespace gko {

template <typename ValueType>
void TrilinosCg<ValueType>::apply_impl(const gko::LinOp* b, gko::LinOp* x) const
{
    using ST = typename Tpetra::Vector<ScalarType>::scalar_type;
    using LO = typename Tpetra::Vector<>::local_ordinal_type;
    using GO = typename Tpetra::Vector<>::global_ordinal_type;
    using NT = typename Tpetra::Vector<>::node_type;
    using SCT = typename Teuchos::ScalarTraits<ST>;
    using MT = typename SCT::magnitudeType;
    using MV = typename Tpetra::MultiVector<ST, LO, GO, NT>;
    using OP = typename Tpetra::Operator<ST, LO, GO, NT>;
    using tmap_t = Tpetra::Map<LO, GO, NT>;
    using tcrsmatrix_t = Tpetra::CrsMatrix<ST, LO, GO, NT>;
    using MVT = typename Belos::MultiVecTraits<ST, MV>;
    using OPT = typename Belos::OperatorTraits<ST, MV, OP>;
    using Teuchos::ParameterList;
    using Teuchos::RCP;
    using Teuchos::rcp;

    const auto Comm = Tpetra::getDefaultComm();
    const int MyPID = Comm->getRank();

    int frequency = -1;  // frequency of status test output.
    int blockSize = 1;   // blockSize
    int numrhs = 1;      // number of right-hand sides to solve for
    int maxIters =
        -1;  // maximum number of iterations allowed per linear system
    procVerbose =
        verbose && (MyPID == 0); /* Only print on the zero processor */

    // Get the problem
    RCP<tcrsmatrix_t> A;
    Tpetra::Utils::readHBMatrix(filename, Comm, A);
    RCP<const tmap_t> Map = A->getDomainMap();
    // Create initial vectors
    RCP<MV> B, X;
    X = rcp(new MV(Map, numrhs));
    MVT::MvRandom(*X);
    B = rcp(new MV(Map, numrhs));
    OPT::Apply(*A, *X, *B);
    MVT::MvInit(*X, 0.0);

    const int numGlobalElements = B->getGlobalLength();
    if (maxIters == -1)
        maxIters = numGlobalElements / blockSize -
                   1;  // maximum number of iterations to run
    //
    ParameterList belosList;
    belosList.set("Block Size",
                  blockSize);  // BlockSize to be used by iterative solver
    belosList.set("Maximum Iterations",
                  maxIters);  // Maximum number of iterations allowed
    belosList.set("Convergence Tolerance",
                  tol);  // Relative convergence tolerance requested
    if (verbose) {
        belosList.set("Verbosity", Belos::Errors + Belos::Warnings +
                                       Belos::TimingDetails +
                                       Belos::StatusTestDetails);
        if (frequency > 0) belosList.set("Output Frequency", frequency);
    } else
        belosList.set("Verbosity", Belos::Errors + Belos::Warnings);

    // Construct an unpreconditioned linear problem instance.
    Belos::LinearProblem<ST, MV, OP> problem(A, X, B);
    bool set = problem.setProblem();
    if (set == false) {
        if (procVerbose)
            std::cout
                << std::endl
                << "ERROR:  Belos::LinearProblem failed to set up correctly!"
                << std::endl;
        return -1;
    }
    // Create an iterative solver manager.
    RCP<Belos::BlockCGSolMgr<ST, MV, OP>> newSolver =
        rcp(new Belos::BlockCGSolMgr<ST, MV, OP>(rcp(&problem, false),
                                                 rcp(&belosList, false)));

    Belos::ReturnType ret = newSolver->solve();

    int numIters = newSolver->getNumIters();
}


template class TrilinosCg<float>;
template class TrilinosCg<double>;

}  // namespace gko

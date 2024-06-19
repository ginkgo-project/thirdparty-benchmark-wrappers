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

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/solver/solver_base.hpp>

#ifndef SOLVERS_BENCHMARK_WRAPPERS_HPP_
#define SOLVERS_BENCHMARK_WRAPPERS_HPP_


namespace gko {


template <typename ValueType>
class TrilinosCg
    : public gko::EnableLinOp<TrilinosCg<ValueType>>,
      public gko::EnablePreconditionedIterativeSolver<ValueType,
                                                      TrilinosCg<ValueType>> {
    friend class gko::EnableLinOp<TrilinosCg>;
    friend class gko::EnablePolymorphicObject<TrilinosCg, gko::LinOp>;

public:
    using value_type = ValueType;

    bool apply_uses_initial_guess() const override { return true; }

    class Factory;

    struct parameters_type
        : enable_preconditioned_iterative_solver_factory_parameters<
              parameters_type, Factory> {};

    GKO_ENABLE_LIN_OP_FACTORY(TrilinosCg, parameters, Factory);
    GKO_ENABLE_BUILD_METHOD(Factory);

protected:
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override;

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta, gko::LinOp* x) const override
    {
        GKO_NOT_IMPLEMENTED;
    }

    explicit TrilinosCg(std::shared_ptr<const gko::Executor> exec)
        : EnableLinOp<TrilinosCg>(std::move(exec))
    {}

    explicit TrilinosCg(const Factory* factory,
                        std::shared_ptr<const gko::LinOp> system_matrix)
        : EnableLinOp<TrilinosCg>(factory->get_executor(),
                                  gko::transpose(system_matrix->get_size())),
          EnablePreconditionedIterativeSolver<ValueType, TrilinosCg<ValueType>>{
              std::move(system_matrix), factory->get_parameters()},
          parameters_{factory->get_parameters()}
    {}
};


template <typename ValueType>
class HypreCsr : public gko::EnableLinOp<HypreCsr<ValueType>>,
                 public gko::ReadableFromMatrixData<ValueType, int32>,
                 public gko::EnableCreateMethod<HypreCsr<ValueType>> {
public:
    using csr = gko::matrix::Csr<ValueType, int32>;
    using mat_data = gko::matrix_data<ValueType, int32>;

    void read(const mat_data& data) override { this->csr_->read(data); }

    HypreCsr(std::shared_ptr<const gko::Executor> exec,
             const gko::dim<2>& size = gko::dim<2>{});

    std::shared_ptr<csr> get_matrix() { return csr_; }

protected:
    void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override;

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* b,
                    const gko::LinOp* beta, gko::LinOp* x) const override
    {
        GKO_NOT_IMPLEMENTED;
    }

private:
    std::shared_ptr<csr> csr_;
};


}  // namespace gko

#endif  // SOLVERS_BENCHMARK_WRAPPERS_HPP_

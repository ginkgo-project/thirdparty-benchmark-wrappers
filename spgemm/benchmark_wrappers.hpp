// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/matrix/csr.hpp>

#ifndef SPGEMM_BENCHMARK_WRAPPERS_HPP_
#define SPGEMM_BENCHMARK_WRAPPERS_HPP_


namespace gko {


template <typename ValueType>
class NSparseCsr : public gko::EnableLinOp<NSparseCsr<ValueType>>,
                   public gko::ReadableFromMatrixData<ValueType, int32>,
                   public gko::EnableCreateMethod<NSparseCsr<ValueType>> {
public:
    using csr = gko::matrix::Csr<ValueType, int32>;
    using mat_data = gko::matrix_data<ValueType, int32>;

    void read(const mat_data& data) override { this->csr_->read(data); }

    NSparseCsr(std::shared_ptr<const gko::Executor> exec,
               const gko::dim<2>& size = gko::dim<2>{})
        : gko::EnableLinOp<NSparseCsr<ValueType>>(exec, size),
          csr_(gko::share(
              csr::create(exec, std::make_shared<typename csr::classical>())))
    {}

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


template <typename ValueType>
class AcCsr : public gko::EnableLinOp<AcCsr<ValueType>>,
              public gko::ReadableFromMatrixData<ValueType, gko::int32>,
              public gko::EnableCreateMethod<AcCsr<ValueType>> {
public:
    using csr = gko::matrix::Csr<ValueType, int32>;
    using mat_data = gko::matrix_data<ValueType, int32>;

    void read(const mat_data& data) override { this->csr_->read(data); }

    AcCsr(std::shared_ptr<const gko::Executor> exec,
          const gko::dim<2>& size = gko::dim<2>{})
        : gko::EnableLinOp<AcCsr<ValueType>>(exec, size),
          csr_(gko::share(
              csr::create(exec, std::make_shared<typename csr::classical>())))
    {}

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


template <typename ValueType>
class SpeckCsr : public gko::EnableLinOp<SpeckCsr<ValueType>>,
                 public gko::ReadableFromMatrixData<ValueType, gko::int32>,
                 public gko::EnableCreateMethod<SpeckCsr<ValueType>> {
public:
    using csr = gko::matrix::Csr<ValueType, int32>;
    using mat_data = gko::matrix_data<ValueType, int32>;

    void read(const mat_data& data) override { this->csr_->read(data); }

    SpeckCsr(std::shared_ptr<const gko::Executor> exec,
             const gko::dim<2>& size = gko::dim<2>{})
        : gko::EnableLinOp<SpeckCsr<ValueType>>(exec, size),
          csr_(gko::share(
              csr::create(exec, std::make_shared<typename csr::classical>())))
    {}

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


template <typename ValueType>
class KokkosCsr : public gko::EnableLinOp<KokkosCsr<ValueType>>,
                  public gko::ReadableFromMatrixData<ValueType, gko::int32>,
                  public gko::EnableCreateMethod<KokkosCsr<ValueType>> {
public:
    using csr = gko::matrix::Csr<ValueType, int32>;
    using mat_data = gko::matrix_data<ValueType, int32>;

    void read(const mat_data& data) override { this->csr_->read(data); }

    KokkosCsr(std::shared_ptr<const gko::Executor> exec,
              const gko::dim<2>& size = gko::dim<2>{});

    ~KokkosCsr();

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

#endif  // SPGEMM_BENCHMARK_WRAPPERS_HPP_

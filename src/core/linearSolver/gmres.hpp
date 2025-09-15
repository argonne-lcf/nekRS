#include "platform.hpp"
#include "linAlg.hpp"

template <typename T> class gmres : public linearSolver
{

public:
  gmres(dlong _Nlocal,
        int _Nfields,
        dlong _fieldOffset,
        const occa::memory &_o_weight,
        int _nRestartVectors,
        bool _flexible,
        bool _iR,
        bool _removeMean,
        std::function<void(const occa::memory &o_q, occa::memory &o_Aq)> _Ax,
        std::function<void(const occa::memory &o_r, occa::memory &o_z)> _preco)
  {
    this->Nlocal = _Nlocal;
    this->Nfields = _Nfields;
    this->fieldOffset = _fieldOffset;
    o_weight = _o_weight;
    nRestartVectors = _nRestartVectors;
    flexible = _flexible;
    Ax = _Ax;
    preco = _preco;
    iR = _iR;
    removeMean = _removeMean;

    weightSum = platform->linAlg->sum(this->Nlocal, o_weight, platform->comm.mpiComm());

    Nblock = (this->Nlocal + BLOCKSIZE - 1) / BLOCKSIZE;

    this->tiny = static_cast<T>(10) * std::numeric_limits<T>::min();
    this->FPfactor = (std::is_same<T, dfloat>::value) ? 1.0 : 0.5;
    this->knlPrefix = std::string("gmres::") + ((std::is_same<T, double>::value) ? "double::" : "float::") +
                      std::to_string(this->Nfields) + "::";

    residualKernel = platform->kernelRequests.load(this->knlPrefix + "fusedResidualAndNorm");
    correctionKernel = platform->kernelRequests.load(this->knlPrefix + "PGMRESSolution");
    updateSolutionKernel = platform->kernelRequests.load(this->knlPrefix + "updatePGMRESSolution");
    gsOrthoKernel = platform->kernelRequests.load(this->knlPrefix + "gramSchmidtOrthogonalization");
  };

  void solve(const dfloat tol, const int _maxIter, const occa::memory &o_rIn, occa::memory &o_xIn) override
  {
    o_r0 = o_rIn;
    this->r0Norm = this->rNorm = platform->linAlg->weightedNorm2Many<T>(this->Nlocal,
                                                                        this->Nfields,
                                                                        this->fieldOffset,
                                                                        this->o_weight,
                                                                        o_r0,
                                                                        platform->comm.mpiComm());

    nekrsCheck(!std::isfinite(this->r0Norm),
               MPI_COMM_SELF,
               EXIT_FAILURE,
               "%s unreasonable initial residual norm!\n",
               this->_name.c_str());

    maxIter = _maxIter;
    nRestartVectors = std::min(nRestartVectors, maxIter);

    H.resize((nRestartVectors + 1) * (nRestartVectors + 1));
    sn.resize(nRestartVectors);
    cs.resize(nRestartVectors);
    s.resize(nRestartVectors + 1);

    o_y = platform->deviceMemoryPool.reserve<T>(nRestartVectors);
    h_y = platform->memoryPool.reserve<T>(o_y.size());

    o_scratch = platform->deviceMemoryPool.reserve<T>(Nblock);
    h_scratch = platform->memoryPool.reserve<T>(o_scratch.size());

    {
      const auto n =
          (this->Nfields > 1) ? this->Nfields * static_cast<size_t>(this->fieldOffset) : this->Nlocal;

      o_V = platform->deviceMemoryPool.reserve<T>(alignStride<T>(n) * nRestartVectors);
      o_Z = platform->deviceMemoryPool.reserve<T>(alignStride<T>(n) * ((flexible) ? nRestartVectors : 1));

      o_tmp = platform->deviceMemoryPool.reserve<T>(n);
      o_w = platform->deviceMemoryPool.reserve<T>(n);

      o_r = platform->deviceMemoryPool.reserve<double>(n);
      if constexpr (std::is_same_v<T, float>) {
        platform->copyFloatToDoubleKernel(o_r.size(), o_r0, o_r);
      } else {
        o_r.copyFrom(o_r0);
      }

      o_xCorr = platform->deviceMemoryPool.reserve<T>(n);
      if constexpr (std::is_same_v<T, float>) {
        o_x = platform->deviceMemoryPool.reserve<double>(n);
      } else {
        o_x = o_xIn.slice(0, n);
      }
      platform->linAlg->fill<double>(o_x.size(), 0.0, o_x);
    }

    if (platform->comm.mpiRank() == 0 && platform->verbose()) {
      auto txt = (preco) ? std::string("P") : std::string("");
      txt += std::string("GMRES");
      if (flexible) {
        txt += "-flex";
      }
      if (iR) {
        txt += "-iR";
      }
      printf("%s %s: initial res norm %.15e target %e \n",
             txt.c_str(),
             this->_name.c_str(),
             this->rNorm,
             tol);
    }

    int Niter = 0;
    do {
      auto NiterInner = runInner(tol, Niter);
      Niter += NiterInner;

      if (iR) {
        updateSolution(NiterInner);
        updateResidual();
        if (platform->comm.mpiRank() == 0 && platform->verbose()) {
          std::cout << "r-norm: " << this->rNorm << std::endl;
        }
      }

      // test for exit conditions
      if (this->rNorm < tol || Niter > maxIter) {
        if (!iR) {
          updateSolution(NiterInner, false);
        }
        break;
      }

      if (!iR) {
        updateSolution(nRestartVectors);
        updateResidual();
        if (platform->comm.mpiRank() == 0 && platform->verbose()) {
          std::cout << "restarting r-norm: " << this->rNorm << std::endl;
        }
      }

    } while (this->rNorm > tol && Niter < maxIter);

    _nIter = Niter;

    if constexpr (std::is_same_v<T, float>) {
      platform->copyDoubleToFloatKernel(o_x.size(), o_x, o_xIn);
    }

    o_r.free();
    o_x.free();
    o_xCorr.free();

    o_y.free();
    h_y.free();

    o_scratch.free();
    h_scratch.free();

    o_tmp.free();
    o_r0.free();
    o_w.free();
    o_V.free();
    o_Z.free();
  };

private:
  occa::memory o_weight;
  bool flexible;
  bool iR;
  bool removeMean;

  int maxIter;
  int nRestartVectors;
  int Nblock;

  dfloat weightSum;

  occa::memory o_V;
  occa::memory o_Z;
  occa::memory o_r0;
  occa::memory o_tmp;
  occa::memory o_w;

  occa::memory o_r;
  occa::memory o_x;
  occa::memory o_xCorr;

  occa::memory o_scratch;
  occa::memory h_scratch;

  occa::memory o_y;
  occa::memory h_y;

  std::vector<dfloat> H;
  std::vector<dfloat> sn;
  std::vector<dfloat> cs;
  std::vector<dfloat> s;

  std::function<void(const occa::memory &o_q, occa::memory &o_Aq)> Ax;
  std::function<void(const occa::memory &o_r, occa::memory &o_z)> preco;

  occa::kernel residualKernel;
  occa::kernel correctionKernel;
  occa::kernel updateSolutionKernel;
  occa::kernel gsOrthoKernel;

  void updateResidual()
  {
    const auto o_Ax = [&]() {
      if (iR) {
        auto o_AxDouble = platform->deviceMemoryPool.reserve<double>(o_x.size());
        Ax(o_x, o_AxDouble);
        return o_AxDouble;
      }

      auto o_xT = o_x;

      if constexpr (std::is_same_v<T, float>) {
        platform->copyDoubleToFloatKernel(o_x.size(), o_x, o_w);
        o_xT = o_w;
      }

      Ax(o_xT, o_tmp);

      if constexpr (std::is_same_v<T, float>) {
        auto o_AxDouble = platform->deviceMemoryPool.reserve<double>(o_x.size());
        platform->copyDfloatToDoubleKernel(o_x.size(), o_tmp, o_AxDouble);
        return o_AxDouble;
      } else {
        return o_tmp;
      }
    }();

    // r += r0 - Ax
    auto o_wrk = platform->deviceMemoryPool.reserve<double>(Nblock);
    residualKernel(Nblock, this->Nlocal, this->fieldOffset, o_weight, o_r0, o_Ax, o_r, o_wrk);

    auto flopCount = this->FPfactor * 4 * this->Nfields * static_cast<double>(this->Nlocal);
    platform->flopCounter->add("gmres evaluate residual and norm", flopCount);

    double norm = 0;
    if (platform->serial()) {
      norm = o_wrk.template ptr<double>()[0];
    } else {
      auto h_wrk = platform->memoryPool.reserve<double>(o_wrk.size());
      o_wrk.copyTo(h_wrk);

      auto wrk = h_wrk.template ptr<double>();
      for (dlong n = 0; n < h_wrk.size(); ++n) {
        norm += wrk[n];
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE, MPI_SUM, platform->comm.mpiComm());

    this->rNorm = std::sqrt(norm);
  };

  void updateSolution(const int gmresUpdateSize, bool runPreco = true)
  {
    for (int k = gmresUpdateSize - 1; k >= 0; --k) {
      auto y = h_y.template ptr<T>();

      y[k] = s[k];

      for (int m = k + 1; m < gmresUpdateSize; ++m) {
        y[k] -= H[k + m * (nRestartVectors + 1)] * y[m];
      }

      y[k] /= (H[k + k * (nRestartVectors + 1)] + this->tiny);
    }

    o_y.copyFrom(h_y, gmresUpdateSize);

    // xCorr = sum_j y[j] * z[j];
    correctionKernel(this->Nlocal,
                     this->fieldOffset,
                     gmresUpdateSize,
                     o_y,
                     (flexible) ? o_Z : o_V,
                     (flexible) ? o_xCorr : o_w);

    if (!flexible) {
      if (preco && runPreco) {
        preco(o_w, o_xCorr);
      } else {
        o_xCorr.copyFrom(o_w);
      }
    }

    // x += xCorr
    updateSolutionKernel(this->Nlocal, this->fieldOffset, o_xCorr, o_x);

    auto flopCount =
        this->FPfactor * (gmresUpdateSize + 1) * this->Nfields * static_cast<double>(this->Nlocal);
    platform->flopCounter->add("gmresUpdate", flopCount);
  };

  int runInner(const dfloat tol, const int iter0)
  {
    const auto offset = o_V.size() / nRestartVectors;

    s[0] = this->rNorm;

    // init o_V0
    auto o_rT = [&]() {
      auto o_rT = o_w;
      if constexpr (std::is_same_v<T, float>) {
        platform->copyDoubleToFloatKernel(o_r.size(), o_r, o_w);
        return o_w;
      } else {
        return o_r;
      }
    }();

    platform->linAlg->axpbyMany<T>(this->Nlocal,
                                   this->Nfields,
                                   this->fieldOffset,
                                   1. / (s[0] + this->tiny),
                                   o_rT,
                                   0.0,
                                   o_V);

    int i;
    for (i = 0; i < nRestartVectors; ++i) {
      // right preconditioning: z_k = M^{-1} v_k
      auto o_zk = [&]() {
        const auto n = o_w.size();
        const auto o_vk = o_V.slice(i * offset, n);
        if (preco) {
          auto o_zk = flexible ? o_Z.slice(i * offset, n) : o_Z.slice(0, n);
          preco(o_vk, o_zk);
          return o_zk;
        } else {
          return o_vk;
        }
      }();

      if (removeMean) {
        const auto dotp = platform->linAlg->innerProd(this->Nlocal,
                                                      o_weight,
                                                      o_zk,
                                                      platform->comm.mpiComm());

        platform->linAlg->add(o_zk.size(), -dotp / weightSum, o_zk);
      }

      Ax(o_zk, o_w);

      // 1 pass classical Gram-Schmidt (project new solution vector o_w onto o_V)
      {
#if USE_WEIGHTED_INNER_PROD_MULTI_DEVICE
        platform->linAlg->weightedInnerProdMulti<T>(this->Nlocal,
                                                    (i + 1),
                                                    this->Nfields,
                                                    this->fieldOffset,
                                                    o_weight,
                                                    o_V,
                                                    o_w,
                                                    platform->comm.mpiComm(),
                                                    o_y);
        o_y.copyTo(h_y, (i + 1));
#else
        platform->linAlg->weightedInnerProdMulti<T>(this->Nlocal,
                                                    (i + 1),
                                                    this->Nfields,
                                                    this->fieldOffset,
                                                    o_weight,
                                                    o_V,
                                                    o_w,
                                                    platform->comm.mpiComm(),
                                                    h_y.template ptr<T>());
        o_y.copyFrom(h_y, (i + 1));
#endif

        // orthogonalize o_w against previous o_V
        gsOrthoKernel(Nblock, this->Nlocal, this->fieldOffset, (i + 1), o_weight, o_y, o_V, o_w, o_scratch);

        double flopCount = FPfactor * 5 * (i + 1) * this->Nfields * static_cast<double>(this->Nlocal);
        platform->flopCounter->add("gramSchmidt", flopCount);
      }

      // normalize  
      auto nw = [&]() {
        dfloat norm = 0;
        if (platform->serial()) {
          norm = o_scratch.ptr<T>()[0];
        } else {
          o_scratch.copyTo(h_scratch);
          auto scratch = h_scratch.template ptr<T>();
          for (int k = 0; k < h_scratch.size(); ++k) {
            norm += scratch[k];
          }
        }
        MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DFLOAT, MPI_SUM, platform->comm.mpiComm());
        return std::sqrt(norm);
      }();

      if (i < nRestartVectors - 1) {
        auto o_Vi = o_V.slice((i + 1) * offset);
        platform->linAlg->axpbyMany<T>(this->Nlocal,
                                       this->Nfields,
                                       this->fieldOffset,
                                       1. / (nw + this->tiny),
                                       o_w,
                                       0,
                                       o_Vi);
      }

      // apply Givens rotation
      H[i + 1 + i * (nRestartVectors + 1)] = nw;
      for (int k = 0; k <= i; ++k) {
        H[k + i * (nRestartVectors + 1)] = h_y.template ptr<T>()[k];
      }

      for (int k = 0; k < i; ++k) {
        const dfloat h1 = H[k + i * (nRestartVectors + 1)];
        const dfloat h2 = H[k + 1 + i * (nRestartVectors + 1)];

        H[k + i * (nRestartVectors + 1)] = cs[k] * h1 + sn[k] * h2;
        H[k + 1 + i * (nRestartVectors + 1)] = -sn[k] * h1 + cs[k] * h2;
      }

      // form i-th rotation matrix
      const auto h1 = H[i + i * (nRestartVectors + 1)];
      const auto h2 = H[i + 1 + i * (nRestartVectors + 1)];
#if 1
      const auto r = std::hypot(h1, h2);
      cs[i] = (r == 0) ? 1 : h1 / r;
      sn[i] = (r == 0) ? 0 : h2 / r;
      H[i + i * (nRestartVectors + 1)] = r;
      H[i + 1 + i * (nRestartVectors + 1)] = 0;
#else
      const auto hr = 1 / (std::sqrt(h1 * h1 + h2 * h2) + this->tiny);
      cs[i] = h1 * hr;
      sn[i] = h2 * hr;

      H[i + i * (nRestartVectors + 1)] = cs[i] * h1 + sn[i] * h2;
      H[i + 1 + i * (nRestartVectors + 1)] = 0;
#endif
      // approximate residual norm
      s[i + 1] = -sn[i] * s[i];
      s[i] = cs[i] * s[i];

      if (!iR) {
        this->rNorm = std::abs(s[i + 1]);

        if (platform->comm.mpiRank() == 0) {
          nekrsCheck(!std::isfinite(this->rNorm),
                     MPI_COMM_SELF,
                     EXIT_FAILURE,
                     "%s invalid resiual norm while running linear solver!\n",
                     this->_name.c_str());
        }

        const auto iter = iter0 + i + 1;
        if (platform->verbose() && platform->comm.mpiRank() == 0) {
          printf("it %d r norm %.15e\n", iter, this->rNorm);
        }

        if (this->rNorm < tol || iter == maxIter) {
          return i + 1;
        }
      }
    }

    return nRestartVectors;
  };
};

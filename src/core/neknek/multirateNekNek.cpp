#include "platform.hpp"
#include "neknek.hpp"

void neknek_t::exchangeTimes(const std::vector<dfloat>& dt, double time)
{
  if (!this->multirate()) {
    return;
  }

  const auto maxOrd = dt.size();
 
  if (!o_time_.isInitialized()) {
    o_time_ = platform->device.malloc<dfloat>(intValOffset_ * (maxOrd + 1));
  }

  if (this->globalMovingMesh) {
    platform->timer.tic("neknek updateInterpPoints");
    this->updateInterpPoints();
    platform->timer.toc("neknek updateInterpPoints");

    this->recomputePartition = true;
  }

  auto o_timeFld = platform->deviceMemoryPool.reserve<dfloat>((maxOrd + 1) * mesh->fieldOffset);
  for (int s = 0; s <= maxOrd; ++s) {
    auto o_timeSlice = o_timeFld.slice(s * mesh->fieldOffset, mesh->fieldOffset);
    platform->linAlg->fill(mesh->fieldOffset, time, o_timeSlice);
    if (s < maxOrd) {
      time -= dt[s];
    }
  }

  this->interpolator->eval(maxOrd + 1, mesh->fieldOffset, o_timeFld, this->intValOffset_, this->o_time_);
}

void neknek_t::extrapolateBoundary(int tstep, double time, bool predictor)
{
  if (!this->multirate()) {
    return;
  }

  int innerSteps = 1;
  platform->options.getArgs("NEKNEK MULTIRATE STEPS", innerSteps);

  if (!predictor && tstep < 3 * innerSteps) {
    return; // too early to provide corrected solution
  }

  int order = nEXT_;
  if (tstep <= innerSteps) {
    order = std::min(order, 1);
  }
  if (tstep <= 2 * innerSteps) {
    order = std::min(order, 2);
  }

  if (npt_ == 0) return; 

  const int predictorStep = predictor ? 1 : 0;
  for(auto&& field : fields_) {
      launchKernel("neknek::extrapolateBoundary",
                   npt_,
                   intValOffset_,
                   static_cast<int>(field.o_filter.size()),
                   order,
                   predictorStep,
                   time,
                   o_time_,
                   field.o_intVal.slice(intValOffset_ * field.o_filter.size()),
                   field.o_intVal.slice(0, npt_));
  }
}

void neknek_t::setCorrectorTime(double time)
{
  if (!multirate()) {
    return;
  }
  platform->linAlg->fill(intValOffset_, time, o_time_);

  // set t^{n,q} to interpolate on subsequent corrector steps
  for(auto&& field : fields_) {
    const auto N = (field.offsetSum / field.offset) * intValOffset_;
    field.o_intVal.copyFrom(field.o_intVal, N, N, 0);

  }
}

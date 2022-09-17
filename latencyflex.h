// Copyright 2021 Tatsuyuki Ishi
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef LATENCYFLEX_H
#define LATENCYFLEX_H

#ifdef LATENCYFLEX_HAVE_PERFETTO
#include <perfetto.h>
PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category("latencyflex").SetDescription("LatencyFleX latency and throughput metrics"));
#else
#define TRACE_COUNTER(...)
#define TRACE_EVENT_BEGIN(...)
#define TRACE_EVENT_END(...)
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace lfx {
namespace internal {
// An exponentially weighted moving average estimator.
class EwmaEstimator {
public:
  // `alpha`: Smoothing factor. Larger values means less smoothing, resulting in
  //          a bumpy but quick response.
  // `full_weight`: Set to true to disable weight correction for initial
  //                samples. The estimator will start with a value of 0 weighted
  //                at 100% instead.
  EwmaEstimator(double alpha, bool full_weight = false)
      : alpha_(alpha), current_weight_(full_weight ? 1.0 : 0.0) {}

  // Update the estimate with `value`. `value` must not be negative. If a
  // negative exponent is used, then `value` must not be too small or the
  // internal accumulator will overflow.
  void update(double value) {
    current_ = (1 - alpha_) * current_ + alpha_ * value;
    current_weight_ = (1 - alpha_) * current_weight_ + alpha_;
  }

  double get() const {
    if (current_weight_ == 0) {
      return 0;
    }
    return current_ / current_weight_;
  }

private:
  double alpha_;
  double current_ = 0;
  double current_weight_;
};
} // namespace internal

// Tracks and computes frame time, latency and the desired sleep time before
// next tick. All time is in nanoseconds. The clock domain doesn't matter as
// long as it's a single consistent clock.
//
// Access must be externally synchronized.
class LatencyFleX {
public:
  uint64_t GetWaitTarget(uint64_t frame_id, uint64_t now, uint64_t vblank_timestamp, uint64_t present_margin) {
    //*earliest_present = 0;
    if (prev_frame_end_id_ == UINT64_MAX)
      return 0;

    int64_t prediction_error = 0;
    if (prev_frame_projected_end_ts_ != 0)
      prediction_error = (int64_t)prev_frame_end_ts_ - (int64_t)prev_frame_projected_end_ts_;
    proj_correction_.update(
        std::max(INT64_C(0), prediction_error) -
        std::max(INT64_C(0), prev_prediction_error_ - prev_comp_applied_));
    prev_prediction_error_ = prediction_error;
    int64_t comp_to_apply = std::round(proj_correction_.get());
    prev_comp_applied_ = comp_to_apply;

    uint64_t latency = (uint64_t)std::max(std::round(latency_.get()), 0.0);
    uint64_t render_time = latency + present_margin + comp_to_apply + wakeup_latency_.get();
    uint64_t target = vblank_timestamp;
    for (uint64_t lat = render_time;;) {
      if (lat <= min_refresh_period)
        target += min_refresh_period;
      else if (lat <= max_refresh_period)
        target += lat;
      /*else if (earliest_present_supported) {
        target += lat;
        *earliest_present = target;
      }*/
      else {
        target += max_refresh_period;
        lat -= max_refresh_period;
        continue;
      }
      while (target < now)
        target += max_refresh_period;
      break;
    }
    target -= render_time;
    std::cerr
      << "latency: " << latency
      << "\npresent_margin: " << present_margin
      << "\ncomp_to_apply: " << comp_to_apply
      << "\nwakeup_latency_.get(): " << wakeup_latency_.get()
      << "\ntarget - now: " << (target - now)
      << "\nnow: " << now
      << "\nvblank_timestamp: " << vblank_timestamp
      << std::endl;

    if (target < now)
      target = now;
    prev_frame_projected_end_ts_ = target + latency + wakeup_latency_.get();
    return target > now ? target : 0;
  }

  // timestamp is the current time
  // target is the wait target from GetWaitTarget, or 0 if no sleep was performed
  void BeginFrame(uint64_t frame_id, uint64_t target, uint64_t timestamp) {
    TRACE_EVENT_BEGIN("latencyflex", "frame",
                      perfetto::Track(track_base_), timestamp);
    prev_frame_begin_id_ = frame_id;
    prev_frame_begin_ts_ = timestamp;
    if (target != 0) {
      int64_t forced_correction = timestamp - target;
      prev_frame_projected_end_ts_ += forced_correction;
      prev_comp_applied_ += forced_correction;
      prev_prediction_error_ += forced_correction;
      if (forced_correction >= 0)
        wakeup_latency_.update(forced_correction);
    }
  }

  // timestamp is the current time
  void EndFrame(uint64_t frame_id, uint64_t timestamp, uint64_t *latency, uint64_t *frame_time) {
    int64_t latency_val = -1;
    int64_t frame_time_val = -1;
    if (prev_frame_begin_id_ != UINT64_MAX && prev_frame_begin_id_ == frame_id) {
      prev_frame_begin_id_ = UINT64_MAX;

      if (frame_time && prev_frame_end_id_ != UINT64_MAX)
        *frame_time = timestamp - prev_frame_end_ts_;
      auto frame_start = prev_frame_begin_ts_;
      latency_val = (int64_t)timestamp - (int64_t)frame_start;
      if (latency_val < INT64_C(50000000)) {
        latency_.update(latency_val);
      }
      TRACE_COUNTER("latencyflex", "Latency", latency_val);
      TRACE_COUNTER("latencyflex", "Latency (Estimate)", latency_.get());
      if (prev_frame_end_id_ != UINT64_MAX && frame_id > prev_frame_end_id_) {
        auto frames_elapsed = frame_id - prev_frame_end_id_;
        frame_time_val =
            ((int64_t)timestamp - (int64_t)prev_frame_end_ts_) / (int64_t)frames_elapsed;
        if (frame_time_val < 0)
          frame_time_val = 0;
        if (frame_time_val < INT64_C(50000000)) {
          inv_throughtput_.update(frame_time_val);
        }
        TRACE_COUNTER("latencyflex", "Frame Time", frame_time_val);
        TRACE_COUNTER("latencyflex", "Frame Time (Estimate)", inv_throughtput_.get());
      }
      prev_frame_end_id_ = frame_id;
      prev_frame_end_ts_ = timestamp;
    }
    if (latency)
      *latency = latency_val;
    if (frame_time)
      *frame_time = frame_time_val;
    TRACE_EVENT_END("latencyflex", perfetto::Track(track_base_),
                    timestamp);
  }

  void Reset() {
    auto new_instance = LatencyFleX();
#ifdef LATENCYFLEX_HAVE_PERFETTO
    new_instance.track_base_ = track_base_ + 2;
#endif
    *this = new_instance;
  }

  uint64_t min_refresh_period = 0;
  uint64_t max_refresh_period = 0;

private:
  uint64_t prev_frame_begin_id_ = UINT64_MAX;
  uint64_t prev_frame_begin_ts_ = 0;
  uint64_t frame_end_projected_ts_ = 0;
  int64_t prev_comp_applied_ = 0;
  int64_t prev_prediction_error_ = 0;
  uint64_t prev_frame_end_id_ = UINT64_MAX;
  uint64_t prev_frame_end_ts_ = 0;
  uint64_t prev_frame_projected_end_ts_ = 0;

  internal::EwmaEstimator latency_ = internal::EwmaEstimator(0.3);
  internal::EwmaEstimator inv_throughtput_ = internal::EwmaEstimator(0.3);
  internal::EwmaEstimator proj_correction_ = internal::EwmaEstimator(0.5, true);
  internal::EwmaEstimator wakeup_latency_ = internal::EwmaEstimator(0.5, true);

#ifdef LATENCYFLEX_HAVE_PERFETTO
  uint64_t track_base_ = 0;
#endif
};
} // namespace lfx

#endif // LATENCYFLEX_H

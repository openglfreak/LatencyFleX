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

#define DEBUG_OUTPUT 0

#include "latencyflex_layer.h"
#include "version.h"

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <tuple>
#include <utility>

#include <dlfcn.h>
#include <vulkan/vk_layer.h>
#include <vulkan/vk_layer_dispatch_table.h>
#include <vulkan/vulkan.h>

#include "latencyflex.h"

#define LAYER_NAME "VK_LAYER_LFX_LatencyFleX"

namespace {
std::mutex frame_counter_lock;
std::condition_variable frame_counter_condvar;
uint64_t frame_counter_begin = 0;
uint64_t frame_counter_end = 0;
uint64_t frame_counter_queue = 0;

std::atomic_bool ticker_needs_reset = false;

uint64_t next_present_id_google = 1;

std::map<VkSwapchainKHR, VkDevice> swapchains;


struct PresentIdToFrameIdMapping {
  uint64_t present_id_google;
  uint64_t frame_id;

  struct Compare {
    bool operator()(struct PresentIdToFrameIdMapping const& first,
                    struct PresentIdToFrameIdMapping const& second) const {
      // Compare equal if present ids match
      if (first.present_id_google == second.present_id_google)
        return false;
      // Must not compare equal with just frame id matching
      if (first.frame_id == second.frame_id)
        return first.present_id_google < second.present_id_google;
      // Sort by frame id
      return first.frame_id < second.frame_id;
    }
  };
};
// Maps a present id to a frame id
std::set<PresentIdToFrameIdMapping, PresentIdToFrameIdMapping::Compare> present_id_google_to_frame_id;

lfx::LatencyFleX manager;

/* OPTIONS FROM ENVIRONMENT */

// Placebo mode. This turns off all sleeping but still retains latency and frame time tracking.
// Useful for comparison benchmarks. Note that if the game does its own sleeping between the
// syncpoint and input sampling, latency values from placebo mode might not be accurate.
bool is_placebo_mode = false;

typedef void(VKAPI_PTR *PFN_overlay_SetMetrics)(const char **, const float *, size_t);
PFN_overlay_SetMetrics overlay_SetMetrics = nullptr;

const std::chrono::milliseconds kRecalibrationSleepTime(200);

typedef std::lock_guard<std::mutex> scoped_lock;
// single global lock, for simplicity
std::mutex global_lock;

struct PresentInfo {
  VkDevice device;
  VkFence fence;
  uint64_t frame_id;
};

// use the loader's dispatch table pointer as a key for dispatch map lookups
template <typename DispatchableType> void *GetKey(DispatchableType inst) { return *(void **)inst; }

// layer book-keeping information, to store dispatch tables by key
std::map<void *, VkLayerInstanceDispatchTable> instance_dispatch;
std::map<void *, VkLayerDispatchTable> device_dispatch;
std::map<void *, VkDevice> device_map;

class FenceWaitThread {
public:
  FenceWaitThread();

  ~FenceWaitThread();

  void Push(PresentInfo &&info) {
    scoped_lock l(local_lock_);
    queue_.push_back(info);
    notify_.notify_all();
  }

private:
  void Worker();

  std::thread thread_;
  std::mutex local_lock_;
  std::condition_variable notify_;
  std::deque<PresentInfo> queue_;
  bool running_ = true;
};

FenceWaitThread::FenceWaitThread() : thread_(&FenceWaitThread::Worker, this) {}

FenceWaitThread::~FenceWaitThread() {
  running_ = false;
  notify_.notify_all();
  thread_.join();
}

void FenceWaitThread::Worker() {
  while (true) {
    PresentInfo info;
    {
      std::unique_lock<std::mutex> l(local_lock_);
      while (queue_.empty()) {
        if (!running_)
          return;
        notify_.wait(l);
      }
      info = queue_.front();
      queue_.pop_front();
    }
    VkDevice device = info.device;
    VkLayerDispatchTable &dispatch = device_dispatch[GetKey(info.device)];
    dispatch.WaitForFences(device, 1, &info.fence, VK_TRUE, -1);
    uint64_t complete = current_time_ns();
    dispatch.DestroyFence(device, info.fence, nullptr);

#if ADVANCED_MODE
    uint64_t latency;
    {
      scoped_lock l(global_lock);
      manager.EndFrame(info.frame_id, complete, &latency, nullptr);
    }

    {
      std::unique_lock<std::mutex> fl(frame_counter_lock);
      uint64_t frame_counter_local = ++frame_counter_end;
      frame_counter_condvar.notify_all();
#if 0
      if (frame_counter_begin > frame_counter_local)
        ticker_needs_reset.store(true);
#endif
    }

    float latency_f = latency / 1000000.;
    const char *name = "Latency";
    if (overlay_SetMetrics && latency != UINT64_MAX) {
      overlay_SetMetrics(&name, &latency_f, 1);
    }
#endif
  }
}

std::map<void *, std::unique_ptr<FenceWaitThread>> wait_threads;
} // namespace

///////////////////////////////////////////////////////////////////////////////////////////
// Layer init and shutdown

VkResult VKAPI_CALL lfx_CreateInstance(const VkInstanceCreateInfo *pCreateInfo,
                                       const VkAllocationCallbacks *pAllocator,
                                       VkInstance *pInstance) {
  VkLayerInstanceCreateInfo *layerCreateInfo = (VkLayerInstanceCreateInfo *)pCreateInfo->pNext;

  // step through the chain of pNext until we get to the link info
  while (layerCreateInfo &&
         (layerCreateInfo->sType != VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO ||
          layerCreateInfo->function != VK_LAYER_LINK_INFO)) {
    layerCreateInfo = (VkLayerInstanceCreateInfo *)layerCreateInfo->pNext;
  }

  if (layerCreateInfo == nullptr) {
    // No loader instance create info
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  PFN_vkGetInstanceProcAddr gpa = layerCreateInfo->u.pLayerInfo->pfnNextGetInstanceProcAddr;
  // move chain on for next layer
  layerCreateInfo->u.pLayerInfo = layerCreateInfo->u.pLayerInfo->pNext;

  PFN_vkCreateInstance createFunc = (PFN_vkCreateInstance)gpa(VK_NULL_HANDLE, "vkCreateInstance");

  VkResult ret = createFunc(pCreateInfo, pAllocator, pInstance);
  if (ret != VK_SUCCESS)
    return ret;

  // fetch our own dispatch table for the functions we need, into the next layer
  VkLayerInstanceDispatchTable dispatchTable;
  dispatchTable.GetInstanceProcAddr =
      (PFN_vkGetInstanceProcAddr)gpa(*pInstance, "vkGetInstanceProcAddr");
  dispatchTable.DestroyInstance = (PFN_vkDestroyInstance)gpa(*pInstance, "vkDestroyInstance");
  dispatchTable.EnumerateDeviceExtensionProperties = (PFN_vkEnumerateDeviceExtensionProperties)gpa(
      *pInstance, "vkEnumerateDeviceExtensionProperties");

  // store the table by key
  {
    scoped_lock l(global_lock);
    instance_dispatch[GetKey(*pInstance)] = dispatchTable;

    if (void *mod = dlopen("libMangoHud.so", RTLD_NOW | RTLD_NOLOAD)) {
      overlay_SetMetrics = (PFN_overlay_SetMetrics)dlsym(mod, "overlay_SetMetrics");
    }
  }

  return VK_SUCCESS;
}

void VKAPI_CALL lfx_DestroyInstance(VkInstance instance, const VkAllocationCallbacks *pAllocator) {
  scoped_lock l(global_lock);
  instance_dispatch[GetKey(instance)].DestroyInstance(instance, pAllocator);
  instance_dispatch.erase(GetKey(instance));
}

static void lfx_AddStringToVectorSet(std::vector<const char*> &v, const char *s) {
  for (auto iter = v.begin(); iter != v.end(); ++iter)
    if (!std::strcmp(*iter, s))
      return;
  v.push_back(s);
}

VkResult VKAPI_CALL lfx_CreateDevice(VkPhysicalDevice physicalDevice,
                                     const VkDeviceCreateInfo *pCreateInfo,
                                     const VkAllocationCallbacks *pAllocator, VkDevice *pDevice) {
  VkLayerDeviceCreateInfo *layerCreateInfo = (VkLayerDeviceCreateInfo *)pCreateInfo->pNext;

  // step through the chain of pNext until we get to the link info
  while (layerCreateInfo &&
         (layerCreateInfo->sType != VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO ||
          layerCreateInfo->function != VK_LAYER_LINK_INFO)) {
    layerCreateInfo = (VkLayerDeviceCreateInfo *)layerCreateInfo->pNext;
  }

  if (layerCreateInfo == nullptr) {
    // No loader instance create info
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  PFN_vkGetInstanceProcAddr gipa = layerCreateInfo->u.pLayerInfo->pfnNextGetInstanceProcAddr;
  PFN_vkGetDeviceProcAddr gdpa = layerCreateInfo->u.pLayerInfo->pfnNextGetDeviceProcAddr;
  // move chain on for next layer
  layerCreateInfo->u.pLayerInfo = layerCreateInfo->u.pLayerInfo->pNext;

  VkDeviceCreateInfo createInfo = *pCreateInfo;
  std::vector<const char*> enabledExtensionNames(createInfo.ppEnabledExtensionNames,
    createInfo.ppEnabledExtensionNames + createInfo.enabledExtensionCount);
  lfx_AddStringToVectorSet(enabledExtensionNames, "VK_KHR_surface");
  lfx_AddStringToVectorSet(enabledExtensionNames, "VK_KHR_swapchain");
  lfx_AddStringToVectorSet(enabledExtensionNames, "VK_KHR_present_id");
  lfx_AddStringToVectorSet(enabledExtensionNames, "VK_KHR_present_wait");
  lfx_AddStringToVectorSet(enabledExtensionNames, "VK_GOOGLE_display_timing");
  createInfo.enabledExtensionCount = enabledExtensionNames.size();
  createInfo.ppEnabledExtensionNames = &enabledExtensionNames[0];

  PFN_vkCreateDevice createFunc = (PFN_vkCreateDevice)gipa(VK_NULL_HANDLE, "vkCreateDevice");

  VkResult ret = createFunc(physicalDevice, &createInfo, pAllocator, pDevice);
  if (ret != VK_SUCCESS)
    return ret;

#if DEBUG_OUTPUT
#define ASSIGN_FUNCTION(name) dispatchTable.name = (PFN_vk##name)gdpa(*pDevice, "vk" #name); std::cerr << #name ": " << (void*)dispatchTable.name << std::endl;
#else // DEBUG_OUTPUT
#define ASSIGN_FUNCTION(name) dispatchTable.name = (PFN_vk##name)gdpa(*pDevice, "vk" #name);
#endif // DEBUG_OUTPUT
  // fetch our own dispatch table for the functions we need, into the next layer
  VkLayerDispatchTable dispatchTable;
  ASSIGN_FUNCTION(GetDeviceProcAddr);
  ASSIGN_FUNCTION(DestroyDevice);
  ASSIGN_FUNCTION(QueuePresentKHR);
  ASSIGN_FUNCTION(AcquireNextImageKHR);
  ASSIGN_FUNCTION(AcquireNextImage2KHR);
  ASSIGN_FUNCTION(CreateFence);
  ASSIGN_FUNCTION(DestroyFence);
  ASSIGN_FUNCTION(QueueSubmit);
  ASSIGN_FUNCTION(WaitForFences);
  ASSIGN_FUNCTION(WaitForPresentKHR);
  ASSIGN_FUNCTION(GetPastPresentationTimingGOOGLE);
  ASSIGN_FUNCTION(GetRefreshCycleDurationGOOGLE);
#undef ASSIGN_FUNCTION

  // store the table by key
  {
    scoped_lock l(global_lock);
    device_dispatch[GetKey(*pDevice)] = dispatchTable;
    device_map[GetKey(*pDevice)] = *pDevice;
    wait_threads[GetKey(*pDevice)] = std::make_unique<FenceWaitThread>();
  }

  return VK_SUCCESS;
}

void VKAPI_CALL lfx_DestroyDevice(VkDevice device, const VkAllocationCallbacks *pAllocator) {
  scoped_lock l(global_lock);
  wait_threads.erase(GetKey(device));
  device_dispatch[GetKey(device)].DestroyDevice(device, pAllocator);
  device_dispatch.erase(GetKey(device));
  device_map.erase(GetKey(device));
}

///////////////////////////////////////////////////////////////////////////////////////////
// Enumeration function

VkResult VKAPI_CALL lfx_EnumerateInstanceLayerProperties(uint32_t *pPropertyCount,
                                                         VkLayerProperties *pProperties) {
  if (pPropertyCount)
    *pPropertyCount = 1;

  if (pProperties) {
    strcpy(pProperties->layerName, LAYER_NAME);
    strcpy(pProperties->description, "LatencyFleX (TM) latency reduction middleware");
    pProperties->implementationVersion = 1;
    pProperties->specVersion = VK_MAKE_VERSION(1, 2, 136);
  }

  return VK_SUCCESS;
}

VkResult VKAPI_CALL lfx_EnumerateDeviceLayerProperties(VkPhysicalDevice physicalDevice,
                                                       uint32_t *pPropertyCount,
                                                       VkLayerProperties *pProperties) {
  return lfx_EnumerateInstanceLayerProperties(pPropertyCount, pProperties);
}

VkResult VKAPI_CALL lfx_EnumerateInstanceExtensionProperties(const char *pLayerName,
                                                             uint32_t *pPropertyCount,
                                                             VkExtensionProperties *pProperties) {
  if (pLayerName == nullptr || strcmp(pLayerName, LAYER_NAME))
    return VK_ERROR_LAYER_NOT_PRESENT;

  // don't expose any extensions
  if (pPropertyCount)
    *pPropertyCount = 0;
  return VK_SUCCESS;
}

VkResult VKAPI_CALL lfx_EnumerateDeviceExtensionProperties(VkPhysicalDevice physicalDevice,
                                                           const char *pLayerName,
                                                           uint32_t *pPropertyCount,
                                                           VkExtensionProperties *pProperties) {
  // pass through any queries that aren't to us
  if (pLayerName == nullptr || strcmp(pLayerName, LAYER_NAME)) {
    if (physicalDevice == VK_NULL_HANDLE)
      return VK_SUCCESS;

    scoped_lock l(global_lock);
    return instance_dispatch[GetKey(physicalDevice)].EnumerateDeviceExtensionProperties(
        physicalDevice, pLayerName, pPropertyCount, pProperties);
  }

  // don't expose any extensions
  if (pPropertyCount)
    *pPropertyCount = 0;
  return VK_SUCCESS;
}

template<typename T>
static inline bool lfx_FindInPNextChain(const void *chain, VkStructureType type, T **out) {
  for (const struct VkBaseInStructure *next = (const struct VkBaseInStructure *)chain; next; next = next->pNext)
    if (next->sType == type) {
      *out = (T*)next;
      return true;
    }
  return false;
}

static void lfx_ProjectPresentTime(uint64_t frame_id, uint64_t *projected_present_time);
static inline uint64_t lfx_GetVulkanTime();

extern "C" VK_LAYER_EXPORT void lfx_WaitAndBeginFrame();
bool wait_before_present;
bool wait_after_present;

VkResult VKAPI_CALL lfx_QueuePresentKHR(VkQueue queue, const VkPresentInfoKHR *pPresentInfo) {
  if (wait_before_present)
    lfx_WaitAndBeginFrame();

  std::unique_lock<std::mutex> l(global_lock);

  uint64_t frame_counter_local;
  {
    std::unique_lock<std::mutex> fl(frame_counter_lock);
    frame_counter_local = ++frame_counter_queue;
    frame_counter_condvar.notify_all();
#if 0
    if (frame_counter_begin > frame_counter_local)
      ticker_needs_reset.store(true);
#endif
  }

  VkDevice device = device_map[GetKey(queue)];
  VkLayerDispatchTable &dispatch = device_dispatch[GetKey(queue)];
  VkFence fence;
  VkFenceCreateInfo fenceInfo{};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  dispatch.CreateFence(device, &fenceInfo, nullptr,
                       &fence); // TODO: error check
  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  VkPipelineStageFlags stages_wait = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
  submitInfo.waitSemaphoreCount = pPresentInfo->waitSemaphoreCount;
  submitInfo.pWaitSemaphores = pPresentInfo->pWaitSemaphores;
  submitInfo.pWaitDstStageMask = &stages_wait;
  submitInfo.signalSemaphoreCount = pPresentInfo->waitSemaphoreCount;
  submitInfo.pSignalSemaphores = pPresentInfo->pWaitSemaphores;
  dispatch.QueueSubmit(queue, 1, &submitInfo, fence);
  wait_threads[GetKey(device)]->Push({device, fence, frame_counter_local});

  VkPresentInfoKHR presentInfoCopy = *pPresentInfo;

  /*VkPresentIdKHR presentId{};
  std::vector<uint64_t> presentIds;
  VkPresentIdKHR *pPresentId;
  if (!lfx_FindInPNextChain(presentInfoCopy.pNext, VK_STRUCTURE_TYPE_PRESENT_ID_KHR, &pPresentId)) {
    presentId.sType = VK_STRUCTURE_TYPE_PRESENT_ID_KHR;
    presentId.pNext = presentInfoCopy.pNext;
    presentInfoCopy.pNext = &presentId;
    presentId.swapchainCount = presentInfoCopy.swapchainCount;
    uint64_t present_id_khr = next_present_id_khr++;
    presentIds = std::vector<uint64_t>(presentId.swapchainCount);
    for (uint32_t i = 0; i < presentId.swapchainCount; ++i)
      presentIds[i] = present_id_khr;
    presentId.pPresentIds = &presentIds[0];
    pPresentId = &presentId;
  }*/

  uint64_t projected_present_time;
  lfx_ProjectPresentTime(frame_counter_local, &projected_present_time);
  uint64_t desired_present_time = projected_present_time ? (projected_present_time - 200000) : 0;
#if DEBUG_OUTPUT
  std::cerr << "queueing frame_id " << frame_counter_local << " with desiredPresentTime " << desired_present_time << std::endl;
#endif // DEBUG_OUTPUT

  VkPresentTimesInfoGOOGLE presentTimes{};
  std::vector<VkPresentTimeGOOGLE> times;
  VkPresentTimesInfoGOOGLE *pPresentTimes;
  if (!lfx_FindInPNextChain(presentInfoCopy.pNext, VK_STRUCTURE_TYPE_PRESENT_TIMES_INFO_GOOGLE, &pPresentTimes)) {
    presentTimes.sType = VK_STRUCTURE_TYPE_PRESENT_TIMES_INFO_GOOGLE;
    presentTimes.pNext = presentInfoCopy.pNext;
    presentInfoCopy.pNext = &presentTimes;
    presentTimes.swapchainCount = presentInfoCopy.swapchainCount;
    uint64_t present_id_google = next_present_id_google++;
    times = std::vector<VkPresentTimeGOOGLE>(presentTimes.swapchainCount);
    for (uint32_t i = 0; i < presentTimes.swapchainCount; ++i) {
      VkPresentTimeGOOGLE presentTime{};
      presentTime.presentID = present_id_google;
      presentTime.desiredPresentTime = desired_present_time;
      times[i] = presentTime;
    }
    presentTimes.pTimes = &times[0];
    pPresentTimes = &presentTimes;
  }

  swapchains.clear();
  for (uint32_t i = 0; i < presentInfoCopy.swapchainCount; ++i)
    swapchains[presentInfoCopy.pSwapchains[i]] = device;

  if (present_id_google_to_frame_id.size() > 16) {
    auto end = present_id_google_to_frame_id.begin();
    std::advance(end, present_id_google_to_frame_id.size() - 16);
    present_id_google_to_frame_id.erase(present_id_google_to_frame_id.begin(), end);
  }

  for (uint32_t i = 0; i < pPresentTimes->swapchainCount; ++i)
    present_id_google_to_frame_id.insert(PresentIdToFrameIdMapping{pPresentTimes->pTimes[i].presentID, frame_counter_local});

  l.unlock();
  if (desired_present_time) {
    struct timespec target;
    target.tv_sec = desired_present_time / 1000000000;
    target.tv_nsec = desired_present_time % 1000000000;
    int err = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &target, NULL);
    if (err)
      std::cerr << "clock_nanosleep error " << err << std::endl;
  }
  VkResult ret = dispatch.QueuePresentKHR(queue, &presentInfoCopy);
#if DEBUG_OUTPUT
  std::cerr << "queued frame_id " << frame_counter_local << " at time " << lfx_GetVulkanTime() << std::endl;
#endif // DEBUG_OUTPUT

  if (wait_after_present)
    lfx_WaitAndBeginFrame();

  return ret;
}

#if DEBUG_OUTPUT
static uint64_t acquire_start_ts = 0;
static uint64_t acquire_end_ts = 0;
static inline void acquire_start()
{
  acquire_start_ts = current_time_ns();
}
static inline void acquire_end()
{
  uint64_t call_to_call = acquire_start_ts - acquire_end_ts;
  acquire_end_ts = current_time_ns();
  std::cerr
    << "- acquire call-to-call: " << call_to_call
    << "\n- acquire delay: " << (acquire_end_ts - acquire_start_ts)
    << std::endl;
}
#else // DEBUG_OUTPUT
static inline void acquire_start() {}
static inline void acquire_end() {}
#endif // DEBUG_OUTPUT

VkResult VKAPI_CALL lfx_AcquireNextImageKHR(VkDevice device, VkSwapchainKHR swapchain,
                                            uint64_t timeout, VkSemaphore semaphore, VkFence fence,
                                            uint32_t *pImageIndex) {
  acquire_start();
  std::unique_lock<std::mutex> l(global_lock);
  VkLayerDispatchTable &dispatch = device_dispatch[GetKey(device)];
  l.unlock();
  VkResult res =
      dispatch.AcquireNextImageKHR(device, swapchain, timeout, semaphore, fence, pImageIndex);
  if (res < 0) {
    // An error has occurred likely due to an Alt-Tab or resize.
    // The application will likely give up presenting this frame, which means that we won't get a
    // call to QueuePresentKHR! This can cause the frame counter to desync. Schedule a recalibration
    // immediately.
    ticker_needs_reset.store(true);
  }
  acquire_end();
  return res;
}

VkResult VKAPI_CALL lfx_AcquireNextImage2KHR(VkDevice device,
                                             const VkAcquireNextImageInfoKHR *pAcquireInfo,
                                             uint32_t *pImageIndex) {
  acquire_start();
  std::unique_lock<std::mutex> l(global_lock);
  VkLayerDispatchTable &dispatch = device_dispatch[GetKey(device)];
  l.unlock();
  VkResult res = dispatch.AcquireNextImage2KHR(device, pAcquireInfo, pImageIndex);
  if (res < 0) {
    // An error has occurred likely due to an Alt-Tab or resize.
    // The application will likely give up presenting this frame, which means that we won't get a
    // call to QueuePresentKHR! This can cause the frame counter to desync. Schedule a recalibration
    // immediately.
    ticker_needs_reset.store(true);
  }
  acquire_end();
  return res;
}

///////////////////////////////////////////////////////////////////////////////////////////
// GetProcAddr functions, entry points of the layer

#define GETPROCADDR(func)                                                                          \
  if (!strcmp(pName, "vk" #func))                                                                  \
  return (PFN_vkVoidFunction)&lfx_##func

extern "C" VK_LAYER_EXPORT PFN_vkVoidFunction VKAPI_CALL lfx_GetDeviceProcAddr(VkDevice device,
                                                                               const char *pName) {
  // device chain functions we intercept
  GETPROCADDR(GetDeviceProcAddr);
  GETPROCADDR(EnumerateDeviceLayerProperties);
  GETPROCADDR(EnumerateDeviceExtensionProperties);
  GETPROCADDR(CreateDevice);
  GETPROCADDR(DestroyDevice);
  GETPROCADDR(QueuePresentKHR);
  GETPROCADDR(AcquireNextImageKHR);
  GETPROCADDR(AcquireNextImage2KHR);

  {
    scoped_lock l(global_lock);
    return device_dispatch[GetKey(device)].GetDeviceProcAddr(device, pName);
  }
}

extern "C" VK_LAYER_EXPORT PFN_vkVoidFunction VKAPI_CALL
lfx_GetInstanceProcAddr(VkInstance instance, const char *pName) {
  // instance chain functions we intercept
  GETPROCADDR(GetInstanceProcAddr);
  GETPROCADDR(EnumerateInstanceLayerProperties);
  GETPROCADDR(EnumerateInstanceExtensionProperties);
  GETPROCADDR(CreateInstance);
  GETPROCADDR(DestroyInstance);

  // device chain functions we intercept
  GETPROCADDR(GetDeviceProcAddr);
  GETPROCADDR(EnumerateDeviceLayerProperties);
  GETPROCADDR(EnumerateDeviceExtensionProperties);
  GETPROCADDR(CreateDevice);
  GETPROCADDR(DestroyDevice);
  GETPROCADDR(QueuePresentKHR);
  GETPROCADDR(AcquireNextImageKHR);
  GETPROCADDR(AcquireNextImage2KHR);

  {
    scoped_lock l(global_lock);
    return instance_dispatch[GetKey(instance)].GetInstanceProcAddr(instance, pName);
  }
}

static struct PresentTimingInfo {
  uint64_t last_present_frame_id;
  uint64_t last_present_ts;
} present_timing_info;
static uint64_t vblank_interval = 16666667;
static int64_t render_delay = 0;

static void lfx_UpdatePresentTimings() {
  for (auto it = swapchains.begin(); it != swapchains.end(); ++it) {
    VkSwapchainKHR swapchain = it->first;
    VkDevice device = it->second;

    uint32_t count = 0;
    device_dispatch[GetKey(device)].GetPastPresentationTimingGOOGLE(device, swapchain, &count, NULL);
    if (count) {
      std::vector<VkPastPresentationTimingGOOGLE> timings(count);
      device_dispatch[GetKey(device)].GetPastPresentationTimingGOOGLE(device, swapchain, &count, &timings[0]);
      for (auto it2 = timings.begin(); it2 != timings.end(); ++it2) {
        auto it3 = present_id_google_to_frame_id.begin();
        for (; it3 != present_id_google_to_frame_id.end(); ++it3)
          if (it3->present_id_google == it2->presentID)
            break;
        if (it3 == present_id_google_to_frame_id.end())
          continue;
        uint64_t frame_id = it3->frame_id;
        present_id_google_to_frame_id.erase(present_id_google_to_frame_id.begin(), ++it3);
#if DEBUG_OUTPUT
        std::cerr << "actualPresentTime: " << it2->actualPresentTime << " frame id " << frame_id << std::endl;
#endif // DEBUG_OUTPUT
        // Warning: The current RADV GOOGLE_display_timing implementation is bugged,
        // and sometimes returns the same actualPresentTime twice, even though it
        // is not accurate. So discard any actualPresentTime that is too close to
        // the saved last_present_ts.
        if (frame_id > present_timing_info.last_present_frame_id && it2->actualPresentTime > present_timing_info.last_present_ts + vblank_interval - 200000) {
          present_timing_info.last_present_frame_id = frame_id;
          present_timing_info.last_present_ts = it2->actualPresentTime;
        }
      }
    }
  }
  swapchains.clear();
}

static void lfx_ProjectPresentTime(uint64_t frame_id, uint64_t *projected_present_time) {
  if (!present_timing_info.last_present_frame_id) {
    *projected_present_time = 0;
    return;
  }
  uint64_t frame_count = frame_id - present_timing_info.last_present_frame_id;
#if DEBUG_OUTPUT
  std::cerr << "lfx_ProjectPresentTime frame_count: " << frame_count << std::endl;
#endif // DEBUG_OUTPUT
  *projected_present_time = present_timing_info.last_present_ts + vblank_interval * frame_count;
}

static inline uint64_t lfx_GetVulkanTime() {
    struct timespec tv;
    clock_gettime(CLOCK_MONOTONIC, &tv);
    return tv.tv_nsec + tv.tv_sec * UINT64_C(1000000000);
}

static uint64_t last_call;
extern "C" VK_LAYER_EXPORT void lfx_WaitAndBeginFrame() {
  uint64_t start = current_time_ns(), wait_end = 0;

  uint64_t frame_counter_local;
  {
    std::unique_lock<std::mutex> fl(frame_counter_lock);
#if 0
    while (frame_counter_begin > frame_counter_end && !ticker_needs_reset.load())
      frame_counter_condvar.wait(fl);
#endif
    wait_end = current_time_ns();
    frame_counter_local = ++frame_counter_begin;
    frame_counter_condvar.notify_all();
#if 1
    if (frame_counter_local <= frame_counter_end) {
      // Presentation has happened without going through the Tick() hook!
      // This typically happens during initialization (where graphics are redrawn
      // without ticking the platform loop).
      ticker_needs_reset.store(true);
    }
#endif
  }

  if (ticker_needs_reset.load()) {
    std::cerr << "LatencyFleX: Performing recalibration!" << std::endl;
    // Try to reset (recalibrate) the state by sleeping for a slightly long
    // period and force any work in the rendering thread or the RHI thread to be
    // flushed. The frame counter is reset after the calibration.
    std::this_thread::sleep_for(kRecalibrationSleepTime);
    // The ticker thread has already incremented the frame counter above. Start
    // from 1, or otherwise it will result in frame ID mismatch.
    {
      std::unique_lock<std::mutex> fl(frame_counter_lock);
      frame_counter_begin = 1;
      frame_counter_local = 1;
      frame_counter_end = 0;
      frame_counter_queue = 0;
    }
    ticker_needs_reset.store(false);
    scoped_lock l(global_lock);
    swapchains.clear();
    present_id_google_to_frame_id.clear();
    present_timing_info = PresentTimingInfo();
    manager.Reset();
  }

  uint64_t projected_present_time;
  {
    scoped_lock l(global_lock);
    manager.min_refresh_period = vblank_interval;
    manager.max_refresh_period = vblank_interval;
    lfx_UpdatePresentTimings();
    lfx_ProjectPresentTime(frame_counter_local, &projected_present_time);
  }

  uint64_t now;
  if (!projected_present_time)
    projected_present_time = (now = current_time_ns());
  else {
    // Translate from Vulkan timestamp to CLOCK_BOOTTIME
    projected_present_time = projected_present_time - lfx_GetVulkanTime();
    projected_present_time += (now = current_time_ns());
  }

#if !ADVANCED_MODE
  if (!is_placebo_mode) {
    uint64_t target = projected_present_time - vblank_interval + render_delay;
    if (target > now)
      std::this_thread::sleep_for(std::chrono::nanoseconds(target - now));
  }
#else
  uint64_t target;
  {
    scoped_lock l(global_lock);
    target = manager.GetWaitTarget(frame_counter_local, now, projected_present_time - vblank_interval, 200000);
  }
#if DEBUG_OUTPUT
  std::cerr << "sleep time: " << (target != 0 ? (int64_t)(target - now) : INT64_C(0)) << std::endl;
#endif // DEBUG_OUTPUT

  uint64_t wakeup;
  if (!is_placebo_mode && target > now) {
    // failsafe: if something ever goes wrong, sustain an interactive framerate
    // so the user can at least quit the application
    static uint64_t failsafe_triggered = 0;
    uint64_t failsafe = now + UINT64_C(50000000);
    if (target > failsafe) {
      wakeup = failsafe;
      failsafe_triggered++;
      if (failsafe_triggered > 0) {
        // If failsafe is triggered multiple times in a row, initiate a recalibration.
        ticker_needs_reset.store(true);
      }
    } else {
      wakeup = target;
      failsafe_triggered = 0;
    }
    std::this_thread::sleep_for(std::chrono::nanoseconds(wakeup - now));
    wakeup = current_time_ns();
  } else {
    target = 0;
    wakeup = now;
  }

  {
    scoped_lock l(global_lock);
    manager.BeginFrame(frame_counter_local, target, wakeup);
  }
#endif

  uint64_t end = current_time_ns();
#if DEBUG_OUTPUT
  std::cerr
    << "+ call-to-call delay: " << (start - last_call)
    << "\n+ actual delay = " << (end - start)
    << "\n+ wait_end - last_call = " << (wait_end - last_call)
    << "\n+ wait_end - start = " << (wait_end - start)
    << "\n+ now = " << now
    << "\n+ frame_id = " << frame_counter_local
    << std::endl;
#endif // DEBUG_OUTPUT
  last_call = end;
}

extern "C" VK_LAYER_EXPORT void lfx_SetTargetFrameTime(uint64_t target_frame_time) {
  std::cerr << "LatencyFleX: ignoring target frame of " << target_frame_time
            << std::endl;
}

namespace {
class OnLoad {
public:
  OnLoad() {
    std::streamsize orig_precision = std::cerr.precision();
    std::ios_base::fmtflags orig_flags = std::cerr.flags();
    std::cerr.precision(2);
    std::cerr.setf(std::ios::fixed, std::ios::floatfield);
    std::cerr.setf(std::ios::showpoint);
    std::cerr << "LatencyFleX: module loaded" << std::endl;
    std::cerr << "LatencyFleX: Version " LATENCYFLEX_VERSION << std::endl;
    if (getenv("LFX_PLACEBO")) {
      is_placebo_mode = true;
      std::cerr << "LatencyFleX: Running in placebo mode" << std::endl;
    }
    if (getenv("LFX_VBLANK_INTERVAL")) {
      vblank_interval = std::stoul(getenv("LFX_VBLANK_INTERVAL"));
    }
    double framerate = 1000000000.0 / vblank_interval;
    std::cerr << "LatencyFleX: Assuming vertical blanking interval of " << vblank_interval << " ns" << std::endl;
    if (getenv("LFX_RENDER_DELAY")) {
      render_delay = std::stol(getenv("LFX_RENDER_DELAY"));
      double min_framerate = 1000000000.0 / (vblank_interval - render_delay);
      std::cerr << "LatencyFleX: Delaying rendering by " << render_delay << " ns (minimum frame rate: " << min_framerate << " fps)" << std::endl;
    }
    if (getenv("LFX_WAIT_BEFORE_PRESENT"))
      wait_before_present = true;
    if (getenv("LFX_WAIT_AFTER_PRESENT")) {
      wait_before_present = false;
      wait_after_present = true;
    }
    if (wait_before_present)
      std::cerr << "LatencyFleX: Waiting before vkQueuePresentKHR" << std::endl;
    if (wait_after_present)
      std::cerr << "LatencyFleX: Waiting after vkQueuePresentKHR" << std::endl;
    std::cerr.precision(orig_precision);
    std::cerr.flags(orig_flags);
  }
};

[[maybe_unused]] OnLoad on_load;
} // namespace

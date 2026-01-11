// cuda_streams.hpp - CUDA Stream Management for BrkgaCuda 2.0 Optimizations
// Provides asynchronous operation pipelining to overlap GPU computation and reduce idle time
//
// Benefits:
// - 20-30% speedup through operation overlap
// - Better GPU utilization
// - Async memory transfers with pinned memory
//
// Based on BrkgaCuda 2.0 paper optimization strategy

#ifndef CUDA_STREAMS_HPP
#define CUDA_STREAMS_HPP

#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <string>

/**
 * @brief Manages CUDA streams for asynchronous operation pipelining
 *
 * The StreamManager class provides infrastructure for overlapping GPU operations
 * across generations and devices. It manages multiple streams and events for
 * efficient synchronization.
 *
 * Usage example:
 * ```cpp
 * StreamManager manager(3);  // 3 streams for pipelining
 *
 * // Stream 0: Evaluate generation N
 * evaluate_kernel<<<grid, block, 0, manager.get_stream(0)>>>(...);
 * manager.record_event(0);
 *
 * // Stream 1: BRKGA operations on generation N-1 (overlaps with eval)
 * brkga_kernel<<<grid, block, 0, manager.get_stream(1)>>>(...);
 *
 * // Stream 2: Async memory copy
 * cudaMemcpyAsync(..., manager.get_stream(2));
 *
 * // Synchronize when needed
 * manager.synchronize_stream(0);
 * ```
 */
class StreamManager {
private:
    std::vector<cudaStream_t> streams_;
    std::vector<cudaEvent_t> events_;
    int num_streams_;
    bool initialized_;

    /**
     * @brief Check CUDA error and throw exception if failed
     */
    void check_cuda_error(cudaError_t error, const std::string& msg) {
        if (error != cudaSuccess) {
            throw std::runtime_error(msg + ": " + cudaGetErrorString(error));
        }
    }

public:
    /**
     * @brief Construct stream manager with specified number of streams
     * @param num_streams Number of concurrent streams (default: 3)
     *
     * Recommended stream counts:
     * - 3 streams: Evaluation, BRKGA ops, Memory transfers
     * - 4+ streams: Multi-island pipelining
     */
    explicit StreamManager(int num_streams = 3)
        : num_streams_(num_streams), initialized_(false) {

        if (num_streams < 1) {
            throw std::invalid_argument("Number of streams must be at least 1");
        }

        try {
            // Allocate streams
            streams_.resize(num_streams);
            for (int i = 0; i < num_streams; i++) {
                cudaError_t err = cudaStreamCreate(&streams_[i]);
                check_cuda_error(err, "Failed to create CUDA stream " + std::to_string(i));
            }

            // Allocate events for synchronization
            events_.resize(num_streams);
            for (int i = 0; i < num_streams; i++) {
                // Use default flags for events (blocking synchronization)
                cudaError_t err = cudaEventCreate(&events_[i]);
                check_cuda_error(err, "Failed to create CUDA event " + std::to_string(i));
            }

            initialized_ = true;
        } catch (...) {
            cleanup();
            throw;
        }
    }

    /**
     * @brief Destructor - automatically cleans up streams and events
     */
    ~StreamManager() {
        cleanup();
    }

    // Disable copy constructor and assignment (streams are non-copyable)
    StreamManager(const StreamManager&) = delete;
    StreamManager& operator=(const StreamManager&) = delete;

    // Enable move constructor and assignment
    StreamManager(StreamManager&& other) noexcept
        : streams_(std::move(other.streams_)),
          events_(std::move(other.events_)),
          num_streams_(other.num_streams_),
          initialized_(other.initialized_) {
        other.initialized_ = false;
    }

    StreamManager& operator=(StreamManager&& other) noexcept {
        if (this != &other) {
            cleanup();
            streams_ = std::move(other.streams_);
            events_ = std::move(other.events_);
            num_streams_ = other.num_streams_;
            initialized_ = other.initialized_;
            other.initialized_ = false;
        }
        return *this;
    }

    /**
     * @brief Get CUDA stream by index
     * @param idx Stream index (0 to num_streams-1)
     * @return CUDA stream handle
     */
    cudaStream_t get_stream(int idx) const {
        if (idx < 0 || idx >= num_streams_) {
            throw std::out_of_range("Stream index out of range: " + std::to_string(idx));
        }
        return streams_[idx];
    }

    /**
     * @brief Get number of streams managed
     * @return Number of streams
     */
    int get_num_streams() const {
        return num_streams_;
    }

    /**
     * @brief Record event in specified stream
     * @param stream_idx Stream index
     *
     * Use this to mark completion points in a stream for later synchronization
     */
    void record_event(int stream_idx) {
        if (stream_idx < 0 || stream_idx >= num_streams_) {
            throw std::out_of_range("Stream index out of range: " + std::to_string(stream_idx));
        }
        cudaError_t err = cudaEventRecord(events_[stream_idx], streams_[stream_idx]);
        check_cuda_error(err, "Failed to record event in stream " + std::to_string(stream_idx));
    }

    /**
     * @brief Wait for event from another stream
     * @param wait_stream_idx Stream that should wait
     * @param event_stream_idx Stream whose event to wait for
     *
     * Example: Make stream 1 wait for stream 0 to finish
     * ```cpp
     * manager.record_event(0);  // Mark completion in stream 0
     * manager.wait_for_event(1, 0);  // Stream 1 waits for stream 0
     * ```
     */
    void wait_for_event(int wait_stream_idx, int event_stream_idx) {
        if (wait_stream_idx < 0 || wait_stream_idx >= num_streams_) {
            throw std::out_of_range("Wait stream index out of range: " + std::to_string(wait_stream_idx));
        }
        if (event_stream_idx < 0 || event_stream_idx >= num_streams_) {
            throw std::out_of_range("Event stream index out of range: " + std::to_string(event_stream_idx));
        }
        cudaError_t err = cudaStreamWaitEvent(streams_[wait_stream_idx], events_[event_stream_idx], 0);
        check_cuda_error(err, "Failed to wait for event");
    }

    /**
     * @brief Synchronize specific stream (block until all operations complete)
     * @param stream_idx Stream index to synchronize
     */
    void synchronize_stream(int stream_idx) {
        if (stream_idx < 0 || stream_idx >= num_streams_) {
            throw std::out_of_range("Stream index out of range: " + std::to_string(stream_idx));
        }
        cudaError_t err = cudaStreamSynchronize(streams_[stream_idx]);
        check_cuda_error(err, "Failed to synchronize stream " + std::to_string(stream_idx));
    }

    /**
     * @brief Synchronize all streams (block until all operations complete)
     *
     * Use this at critical synchronization points, e.g., before migration
     * or final result retrieval
     */
    void synchronize_all() {
        for (int i = 0; i < num_streams_; i++) {
            cudaError_t err = cudaStreamSynchronize(streams_[i]);
            check_cuda_error(err, "Failed to synchronize stream " + std::to_string(i));
        }
    }

    /**
     * @brief Query if stream has completed all operations
     * @param stream_idx Stream index to query
     * @return true if stream is idle, false if work pending
     */
    bool is_stream_idle(int stream_idx) const {
        if (stream_idx < 0 || stream_idx >= num_streams_) {
            throw std::out_of_range("Stream index out of range: " + std::to_string(stream_idx));
        }
        cudaError_t err = cudaStreamQuery(streams_[stream_idx]);
        return (err == cudaSuccess);
    }

    /**
     * @brief Get elapsed time between two recorded events
     * @param start_event_idx Index of start event
     * @param end_event_idx Index of end event
     * @return Elapsed time in milliseconds
     *
     * Useful for performance profiling
     */
    float get_elapsed_time(int start_event_idx, int end_event_idx) {
        if (start_event_idx < 0 || start_event_idx >= num_streams_) {
            throw std::out_of_range("Start event index out of range");
        }
        if (end_event_idx < 0 || end_event_idx >= num_streams_) {
            throw std::out_of_range("End event index out of range");
        }

        // Synchronize events first
        cudaEventSynchronize(events_[start_event_idx]);
        cudaEventSynchronize(events_[end_event_idx]);

        float milliseconds = 0;
        cudaError_t err = cudaEventElapsedTime(&milliseconds,
                                               events_[start_event_idx],
                                               events_[end_event_idx]);
        check_cuda_error(err, "Failed to get elapsed time");
        return milliseconds;
    }

    /**
     * @brief Reset all streams (useful for benchmarking)
     *
     * Synchronizes all streams and resets internal state
     */
    void reset() {
        synchronize_all();
    }

private:
    /**
     * @brief Clean up CUDA resources
     */
    void cleanup() {
        if (!initialized_) return;

        // Destroy events
        for (auto& event : events_) {
            cudaEventDestroy(event);
        }
        events_.clear();

        // Destroy streams
        for (auto& stream : streams_) {
            cudaStreamDestroy(stream);
        }
        streams_.clear();

        initialized_ = false;
    }
};

/**
 * @brief Helper class for pinned memory allocation
 *
 * Pinned (page-locked) memory enables faster async transfers between host and device.
 * Use this for migration buffers and result copies.
 *
 * Usage:
 * ```cpp
 * PinnedMemory<float> buffer(1000);  // Allocate 1000 floats
 * cudaMemcpyAsync(buffer.data(), d_src, size, cudaMemcpyDeviceToHost, stream);
 * ```
 */
template<typename T>
class PinnedMemory {
private:
    T* host_ptr_;
    size_t size_;
    bool allocated_;

public:
    /**
     * @brief Allocate pinned memory
     * @param size Number of elements
     */
    explicit PinnedMemory(size_t size) : size_(size), allocated_(false), host_ptr_(nullptr) {
        if (size > 0) {
            cudaError_t err = cudaMallocHost((void**)&host_ptr_, size * sizeof(T));
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to allocate pinned memory: " +
                                       std::string(cudaGetErrorString(err)));
            }
            allocated_ = true;
        }
    }

    /**
     * @brief Destructor - frees pinned memory
     */
    ~PinnedMemory() {
        if (allocated_ && host_ptr_) {
            cudaFreeHost(host_ptr_);
        }
    }

    // Disable copy
    PinnedMemory(const PinnedMemory&) = delete;
    PinnedMemory& operator=(const PinnedMemory&) = delete;

    // Enable move
    PinnedMemory(PinnedMemory&& other) noexcept
        : host_ptr_(other.host_ptr_), size_(other.size_), allocated_(other.allocated_) {
        other.host_ptr_ = nullptr;
        other.allocated_ = false;
    }

    PinnedMemory& operator=(PinnedMemory&& other) noexcept {
        if (this != &other) {
            if (allocated_ && host_ptr_) {
                cudaFreeHost(host_ptr_);
            }
            host_ptr_ = other.host_ptr_;
            size_ = other.size_;
            allocated_ = other.allocated_;
            other.host_ptr_ = nullptr;
            other.allocated_ = false;
        }
        return *this;
    }

    /**
     * @brief Get pointer to pinned memory
     * @return Host pointer
     */
    T* data() { return host_ptr_; }
    const T* data() const { return host_ptr_; }

    /**
     * @brief Get size in elements
     * @return Number of elements
     */
    size_t size() const { return size_; }

    /**
     * @brief Get size in bytes
     * @return Number of bytes
     */
    size_t size_bytes() const { return size_ * sizeof(T); }

    /**
     * @brief Array access operator
     */
    T& operator[](size_t idx) { return host_ptr_[idx]; }
    const T& operator[](size_t idx) const { return host_ptr_[idx]; }
};

#endif // CUDA_STREAMS_HPP

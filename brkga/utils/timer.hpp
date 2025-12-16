#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    bool is_running;
    
public:
    Timer() : is_running(false) {}
    
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
        is_running = true;
    }
    
    void stop() {
        end_time = std::chrono::high_resolution_clock::now();
        is_running = false;
    }
    
    double elapsed_seconds() const {
        auto current_end = is_running ? std::chrono::high_resolution_clock::now() : end_time;
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(current_end - start_time);
        return duration.count() / 1000000.0;
    }
    
    double elapsed_ms() const {
        auto current_end = is_running ? std::chrono::high_resolution_clock::now() : end_time;
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(current_end - start_time);
        return duration.count() / 1000.0;
    }
    
    long long elapsed_microseconds() const {
        auto current_end = is_running ? std::chrono::high_resolution_clock::now() : end_time;
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(current_end - start_time);
        return duration.count();
    }
    
    bool running() const {
        return is_running;
    }
    
    void reset() {
        is_running = false;
    }
};

// RAII timer for automatic timing
class ScopedTimer {
private:
    Timer& timer;
    
public:
    explicit ScopedTimer(Timer& t) : timer(t) {
        timer.start();
    }
    
    ~ScopedTimer() {
        timer.stop();
    }
};

#endif // TIMER_HPP
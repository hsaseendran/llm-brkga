#ifndef RANDOM_UTILS_HPP
#define RANDOM_UTILS_HPP

#include <random>
#include <vector>
#include <chrono>

class RandomUtils {
private:
    static std::mt19937& get_rng() {
        static std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
        return rng;
    }
    
public:
    // Seed the global random number generator
    static void seed(unsigned int seed) {
        get_rng().seed(seed);
    }
    
    // Get a random real number in [min, max]
    template<typename T>
    static T uniform_real(T min = T(0), T max = T(1)) {
        std::uniform_real_distribution<T> dist(min, max);
        return dist(get_rng());
    }
    
    // Get a random integer in [min, max]
    static int uniform_int(int min, int max) {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(get_rng());
    }
    
    // Get a random boolean with given probability of being true
    static bool bernoulli(double probability = 0.5) {
        std::bernoulli_distribution dist(probability);
        return dist(get_rng());
    }
    
    // Get a normally distributed random number
    template<typename T>
    static T normal(T mean = T(0), T stddev = T(1)) {
        std::normal_distribution<T> dist(mean, stddev);
        return dist(get_rng());
    }
    
    // Get an exponentially distributed random number
    template<typename T>
    static T exponential(T lambda = T(1)) {
        std::exponential_distribution<T> dist(lambda);
        return dist(get_rng());
    }
    
    // Shuffle a vector randomly
    template<typename T>
    static void shuffle(std::vector<T>& vec) {
        std::shuffle(vec.begin(), vec.end(), get_rng());
    }
    
    // Generate a random permutation of integers [0, n-1]
    static std::vector<int> random_permutation(int n) {
        std::vector<int> perm(n);
        std::iota(perm.begin(), perm.end(), 0);
        shuffle(perm);
        return perm;
    }
    
    // Sample k elements without replacement from [0, n-1]
    static std::vector<int> sample_without_replacement(int n, int k) {
        if (k > n) {
            throw std::invalid_argument("k cannot be greater than n");
        }
        
        std::vector<int> population(n);
        std::iota(population.begin(), population.end(), 0);
        
        std::vector<int> sample;
        sample.reserve(k);
        
        for (int i = 0; i < k; i++) {
            int idx = uniform_int(0, population.size() - 1);
            sample.push_back(population[idx]);
            population.erase(population.begin() + idx);
        }
        
        return sample;
    }
    
    // Sample k elements with replacement from [0, n-1]
    static std::vector<int> sample_with_replacement(int n, int k) {
        std::vector<int> sample;
        sample.reserve(k);
        
        for (int i = 0; i < k; i++) {
            sample.push_back(uniform_int(0, n - 1));
        }
        
        return sample;
    }
    
    // Generate random chromosome for BRKGA
    template<typename T>
    static std::vector<T> random_chromosome(int length) {
        std::vector<T> chromosome(length);
        for (auto& gene : chromosome) {
            gene = uniform_real<T>();
        }
        return chromosome;
    }
    
    // Generate multiple random chromosomes
    template<typename T>
    static std::vector<std::vector<T>> random_chromosomes(int count, int length) {
        std::vector<std::vector<T>> chromosomes;
        chromosomes.reserve(count);
        
        for (int i = 0; i < count; i++) {
            chromosomes.push_back(random_chromosome<T>(length));
        }
        
        return chromosomes;
    }
    
    // Random choice from a vector
    template<typename T>
    static const T& choice(const std::vector<T>& vec) {
        if (vec.empty()) {
            throw std::invalid_argument("Cannot choose from empty vector");
        }
        return vec[uniform_int(0, vec.size() - 1)];
    }
    
    // Random choices with weights (roulette wheel selection)
    template<typename T>
    static const T& weighted_choice(const std::vector<T>& vec, const std::vector<double>& weights) {
        if (vec.size() != weights.size()) {
            throw std::invalid_argument("Vector and weights must have same size");
        }
        
        if (vec.empty()) {
            throw std::invalid_argument("Cannot choose from empty vector");
        }
        
        // Calculate cumulative weights
        std::vector<double> cumulative_weights(weights.size());
        cumulative_weights[0] = weights[0];
        for (size_t i = 1; i < weights.size(); i++) {
            cumulative_weights[i] = cumulative_weights[i-1] + weights[i];
        }
        
        // Generate random value
        double total_weight = cumulative_weights.back();
        double random_value = uniform_real(0.0, total_weight);
        
        // Find the selected element
        for (size_t i = 0; i < cumulative_weights.size(); i++) {
            if (random_value <= cumulative_weights[i]) {
                return vec[i];
            }
        }
        
        // Fallback (should not reach here)
        return vec.back();
    }
    
    // Generate a random seed
    static unsigned int random_seed() {
        return std::chrono::steady_clock::now().time_since_epoch().count();
    }
    
    // Create a new random number generator with specified seed
    static std::mt19937 create_rng(unsigned int seed = 0) {
        if (seed == 0) {
            seed = random_seed();
        }
        return std::mt19937(seed);
    }
};

// Thread-safe random number generator (for parallel operations)
class ThreadSafeRandom {
private:
    thread_local static std::mt19937 rng;
    thread_local static bool initialized;
    
    static void ensure_initialized() {
        if (!initialized) {
            rng.seed(RandomUtils::random_seed());
            initialized = true;
        }
    }
    
public:
    template<typename T>
    static T uniform_real(T min = T(0), T max = T(1)) {
        ensure_initialized();
        std::uniform_real_distribution<T> dist(min, max);
        return dist(rng);
    }
    
    static int uniform_int(int min, int max) {
        ensure_initialized();
        std::uniform_int_distribution<int> dist(min, max);
        return dist(rng);
    }
    
    static bool bernoulli(double probability = 0.5) {
        ensure_initialized();
        std::bernoulli_distribution dist(probability);
        return dist(rng);
    }
};

// Initialize thread-local variables
thread_local std::mt19937 ThreadSafeRandom::rng;
thread_local bool ThreadSafeRandom::initialized = false;

#endif // RANDOM_UTILS_HPP
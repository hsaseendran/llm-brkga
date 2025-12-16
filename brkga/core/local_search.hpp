#ifndef LOCAL_SEARCH_HPP
#define LOCAL_SEARCH_HPP

#include "individual.hpp"
#include <map>
#include <string>
#include <random>
#include <chrono>

// Enum for different local search application strategies
enum class LocalSearchStrategy {
    DISABLED,           // No local search
    BEST_ONLY,         // Apply only to best individual
    ELITE_ONLY,        // Apply to all elite individuals
    RANDOM_SAMPLE,     // Apply to random sample of population
    ALL_INDIVIDUALS,   // Apply to entire population
    STAGNATION_ONLY,   // Apply only when population stagnates
    ADAPTIVE           // Adapt based on improvement rate
};

// Enum for when to apply local search
enum class LocalSearchTiming {
    POST_CROSSOVER,    // After generating offspring
    POST_MUTATION,     // After generating mutants
    POST_EVALUATION,   // After fitness evaluation
    END_GENERATION,    // At end of each generation
    FINAL_POLISH       // Only at the very end
};

template<typename T>
class LocalSearch {
protected:
    std::mt19937 rng;
    int max_iterations;
    double improvement_threshold;
    bool verbose;
    std::string name;
    
    // Statistics
    mutable int applications_count;
    mutable int improvements_count;
    mutable double total_improvement;
    mutable double total_time_ms;
    
public:
    LocalSearch(const std::string& search_name = "LocalSearch") 
        : rng(std::chrono::steady_clock::now().time_since_epoch().count()),
          max_iterations(100),
          improvement_threshold(0.001),
          verbose(false),
          name(search_name),
          applications_count(0),
          improvements_count(0),
          total_improvement(0.0),
          total_time_ms(0.0) {}
    
    virtual ~LocalSearch() = default;
    
    // Main interface methods
    virtual Individual<T> improve(const Individual<T>& individual) = 0;
    virtual bool should_apply(int generation, const Individual<T>& individual, 
                             const std::vector<Individual<T>>& population) = 0;
    virtual void configure(const std::map<std::string, std::string>& params) = 0;
    virtual LocalSearch<T>* clone() const = 0;
    
    // Apply local search with timing and statistics
    Individual<T> apply(const Individual<T>& individual) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        T original_fitness = individual.fitness;
        Individual<T> improved = improve(individual);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        // Update statistics
        applications_count++;
        total_time_ms += duration.count() / 1000.0;
        
        if (is_better(improved.fitness, original_fitness)) {
            improvements_count++;
            total_improvement += std::abs(improved.fitness - original_fitness);
            
            if (verbose) {
                std::cout << "[" << name << "] Improved: " 
                          << original_fitness << " -> " << improved.fitness 
                          << " (+" << (improved.fitness - original_fitness) << ")" << std::endl;
            }
        }
        
        return improved;
    }
    
    // Configuration methods
    void set_max_iterations(int max_iter) { max_iterations = max_iter; }
    void set_improvement_threshold(double threshold) { improvement_threshold = threshold; }
    void set_verbose(bool verb) { verbose = verb; }
    void set_name(const std::string& search_name) { name = search_name; }
    
    // Getters
    int get_max_iterations() const { return max_iterations; }
    double get_improvement_threshold() const { return improvement_threshold; }
    const std::string& get_name() const { return name; }
    
    // Statistics
    void reset_statistics() {
        applications_count = 0;
        improvements_count = 0;
        total_improvement = 0.0;
        total_time_ms = 0.0;
    }
    
    void print_statistics() const {
        std::cout << "\n=== " << name << " Statistics ===" << std::endl;
        std::cout << "Applications: " << applications_count << std::endl;
        std::cout << "Improvements: " << improvements_count << std::endl;
        std::cout << "Success rate: " << std::fixed << std::setprecision(2) 
                  << (applications_count > 0 ? (double)improvements_count / applications_count * 100 : 0) 
                  << "%" << std::endl;
        std::cout << "Total improvement: " << std::setprecision(4) << total_improvement << std::endl;
        std::cout << "Average improvement: " << std::setprecision(4) 
                  << (improvements_count > 0 ? total_improvement / improvements_count : 0) << std::endl;
        std::cout << "Total time: " << std::setprecision(2) << total_time_ms << " ms" << std::endl;
        std::cout << "Average time per application: " << std::setprecision(2) 
                  << (applications_count > 0 ? total_time_ms / applications_count : 0) << " ms" << std::endl;
        std::cout << "===============================" << std::endl;
    }
    
    double get_success_rate() const {
        return applications_count > 0 ? (double)improvements_count / applications_count : 0.0;
    }
    
    double get_average_improvement() const {
        return improvements_count > 0 ? total_improvement / improvements_count : 0.0;
    }
    
    double get_average_time_ms() const {
        return applications_count > 0 ? total_time_ms / applications_count : 0.0;
    }

protected:
    // Helper method to determine if one fitness is better than another
    virtual bool is_better(T fitness1, T fitness2) const {
        // Default: minimization problem
        return fitness1 < fitness2;
    }
    
    // Helper method to generate random neighbor
    virtual std::vector<int> get_random_neighbors(int size, int count) {
        std::vector<int> neighbors;
        std::uniform_int_distribution<int> dist(0, size - 1);
        
        while (neighbors.size() < static_cast<size_t>(count) && neighbors.size() < static_cast<size_t>(size)) {
            int neighbor = dist(rng);
            if (std::find(neighbors.begin(), neighbors.end(), neighbor) == neighbors.end()) {
                neighbors.push_back(neighbor);
            }
        }
        
        return neighbors;
    }
    
    // Helper method for basic configuration parsing
    void parse_basic_config(const std::map<std::string, std::string>& params) {
        auto it = params.find("max_iterations");
        if (it != params.end()) {
            max_iterations = std::stoi(it->second);
        }
        
        it = params.find("improvement_threshold");
        if (it != params.end()) {
            improvement_threshold = std::stod(it->second);
        }
        
        it = params.find("verbose");
        if (it != params.end()) {
            verbose = (it->second == "true" || it->second == "1");
        }
    }
};

// Local search configuration container
template<typename T>
struct LocalSearchConfig {
    LocalSearchStrategy strategy = LocalSearchStrategy::DISABLED;
    LocalSearchTiming timing = LocalSearchTiming::POST_EVALUATION;
    int frequency = 10;                    // Apply every N generations
    double probability = 1.0;              // Probability of applying to selected individuals
    int max_applications_per_gen = -1;     // -1 means unlimited
    bool apply_to_best = true;             // Always apply to best solution
    bool enable_parallel = false;          // Enable parallel local search
    
    // Adaptive parameters
    int stagnation_threshold = 50;         // Generations without improvement
    double min_improvement_rate = 0.001;   // Minimum improvement to continue
    
    void print_config() const {
        std::cout << "Local Search Configuration:" << std::endl;
        std::cout << "  Strategy: ";
        switch (strategy) {
            case LocalSearchStrategy::DISABLED: std::cout << "DISABLED"; break;
            case LocalSearchStrategy::BEST_ONLY: std::cout << "BEST_ONLY"; break;
            case LocalSearchStrategy::ELITE_ONLY: std::cout << "ELITE_ONLY"; break;
            case LocalSearchStrategy::RANDOM_SAMPLE: std::cout << "RANDOM_SAMPLE"; break;
            case LocalSearchStrategy::ALL_INDIVIDUALS: std::cout << "ALL_INDIVIDUALS"; break;
            case LocalSearchStrategy::STAGNATION_ONLY: std::cout << "STAGNATION_ONLY"; break;
            case LocalSearchStrategy::ADAPTIVE: std::cout << "ADAPTIVE"; break;
        }
        std::cout << std::endl;
        
        std::cout << "  Timing: ";
        switch (timing) {
            case LocalSearchTiming::POST_CROSSOVER: std::cout << "POST_CROSSOVER"; break;
            case LocalSearchTiming::POST_MUTATION: std::cout << "POST_MUTATION"; break;
            case LocalSearchTiming::POST_EVALUATION: std::cout << "POST_EVALUATION"; break;
            case LocalSearchTiming::END_GENERATION: std::cout << "END_GENERATION"; break;
            case LocalSearchTiming::FINAL_POLISH: std::cout << "FINAL_POLISH"; break;
        }
        std::cout << std::endl;
        
        std::cout << "  Frequency: " << frequency << " generations" << std::endl;
        std::cout << "  Probability: " << std::fixed << std::setprecision(2) << probability * 100 << "%" << std::endl;
        std::cout << "  Max applications per gen: " << (max_applications_per_gen < 0 ? "unlimited" : std::to_string(max_applications_per_gen)) << std::endl;
        std::cout << "  Apply to best: " << (apply_to_best ? "yes" : "no") << std::endl;
        std::cout << "  Parallel: " << (enable_parallel ? "yes" : "no") << std::endl;
    }
};

#endif // LOCAL_SEARCH_HPP
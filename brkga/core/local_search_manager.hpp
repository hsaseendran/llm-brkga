#ifndef LOCAL_SEARCH_MANAGER_HPP
#define LOCAL_SEARCH_MANAGER_HPP

#include "local_search.hpp"
#include "individual.hpp"
#include <vector>
#include <memory>
#include <algorithm>
#include <random>
#include <functional>

template<typename T>
class LocalSearchManager {
private:
    std::vector<std::unique_ptr<LocalSearch<T>>> local_searches;
    LocalSearchConfig<T> config;
    std::mt19937 rng;
    
    // Tracking variables for adaptive behavior
    int generations_without_improvement;
    T last_best_fitness;
    bool first_generation;
    
    // Comparator function for fitness comparison
    std::function<bool(T, T)> comparator;
    
public:
    LocalSearchManager() 
        : rng(std::chrono::steady_clock::now().time_since_epoch().count()),
          generations_without_improvement(0),
          last_best_fitness(T(0)),
          first_generation(true),
          comparator([](T a, T b) { return a < b; }) {}  // Default: minimization
    
    // Set the fitness comparator (for maximization vs minimization problems)
    void set_comparator(std::function<bool(T, T)> comp) {
        comparator = comp;
    }
    
    // Add a local search algorithm
    void add_local_search(std::unique_ptr<LocalSearch<T>> search) {
        local_searches.push_back(std::move(search));
    }
    
    // Configure the local search behavior
    void set_config(const LocalSearchConfig<T>& ls_config) {
        config = ls_config;
    }
    
    LocalSearchConfig<T>& get_config() { return config; }
    const LocalSearchConfig<T>& get_config() const { return config; }
    
    // Check if local search should be applied this generation
    bool should_apply_this_generation(int generation, T current_best_fitness) {
        if (config.strategy == LocalSearchStrategy::DISABLED || local_searches.empty()) {
            return false;
        }
        
        // Update stagnation tracking
        if (first_generation) {
            last_best_fitness = current_best_fitness;
            first_generation = false;
            generations_without_improvement = 0;
        } else {
            if (comparator(current_best_fitness, last_best_fitness)) {
                // Improvement found
                last_best_fitness = current_best_fitness;
                generations_without_improvement = 0;
            } else {
                generations_without_improvement++;
            }
        }
        
        switch (config.strategy) {
            case LocalSearchStrategy::DISABLED:
                return false;
                
            case LocalSearchStrategy::STAGNATION_ONLY:
                return generations_without_improvement >= config.stagnation_threshold;
                
            case LocalSearchStrategy::ADAPTIVE:
                // Apply more frequently when stagnating
                if (generations_without_improvement >= config.stagnation_threshold) {
                    return true;
                }
                // Otherwise apply with normal frequency
                return generation % config.frequency == 0;
                
            default:
                // Regular frequency-based application
                return generation % config.frequency == 0;
        }
    }
    
    // Apply local search to a population based on strategy
    std::vector<Individual<T>> apply_to_population(const std::vector<Individual<T>>& population, 
                                                  int generation) {
        if (!should_apply_this_generation(generation, get_best_fitness(population))) {
            return population;
        }
        
        std::vector<Individual<T>> improved_population = population;
        std::vector<int> target_indices = select_targets(improved_population, generation);
        
        int applications_count = 0;
        for (int index : target_indices) {
            if (config.max_applications_per_gen > 0 && 
                applications_count >= config.max_applications_per_gen) {
                break;
            }
            
            // Check probability
            std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
            if (prob_dist(rng) > config.probability) {
                continue;
            }
            
            // Apply local search
            improved_population[index] = apply_to_individual(improved_population[index], generation);
            applications_count++;
        }
        
        return improved_population;
    }
    
    // Apply local search to a single individual
    Individual<T> apply_to_individual(const Individual<T>& individual, int generation) {
        if (local_searches.empty()) {
            return individual;
        }
        
        Individual<T> current = individual;
        
        // Apply each local search algorithm
        for (auto& search : local_searches) {
            if (search->should_apply(generation, current, {})) {
                current = search->apply(current);
            }
        }
        
        return current;
    }
    
    // Apply intensive local search (for final polishing)
    Individual<T> intensive_search(const Individual<T>& individual) {
        Individual<T> current = individual;
        Individual<T> best = individual;
        
        bool improved = true;
        int iteration = 0;
        const int max_intensive_iterations = 10;
        
        while (improved && iteration < max_intensive_iterations) {
            improved = false;
            
            for (auto& search : local_searches) {
                Individual<T> candidate = search->apply(current);
                if (comparator(candidate.fitness, best.fitness)) {
                    best = candidate;
                    current = candidate;
                    improved = true;
                }
            }
            
            iteration++;
        }
        
        return best;
    }
    
    // Select target individuals based on strategy
    std::vector<int> select_targets(const std::vector<Individual<T>>& population, int generation) {
        std::vector<int> targets;
        
        switch (config.strategy) {
            case LocalSearchStrategy::BEST_ONLY: {
                if (!population.empty()) {
                    targets.push_back(0); // Assuming population is sorted
                }
                break;
            }
            
            case LocalSearchStrategy::ELITE_ONLY: {
                // Assume first 20% are elite (this could be parameterized)
                int elite_count = std::max(1, static_cast<int>(population.size() * 0.2));
                for (int i = 0; i < elite_count && i < static_cast<int>(population.size()); i++) {
                    targets.push_back(i);
                }
                break;
            }
            
            case LocalSearchStrategy::RANDOM_SAMPLE: {
                int sample_size = std::max(1, static_cast<int>(population.size() * 0.1));
                std::uniform_int_distribution<int> dist(0, population.size() - 1);
                
                for (int i = 0; i < sample_size; i++) {
                    int index = dist(rng);
                    if (std::find(targets.begin(), targets.end(), index) == targets.end()) {
                        targets.push_back(index);
                    }
                }
                break;
            }
            
            case LocalSearchStrategy::ALL_INDIVIDUALS: {
                for (int i = 0; i < static_cast<int>(population.size()); i++) {
                    targets.push_back(i);
                }
                break;
            }
            
            case LocalSearchStrategy::ADAPTIVE:
            case LocalSearchStrategy::STAGNATION_ONLY: {
                // Use elite + random sample when stagnating
                int elite_count = std::max(1, static_cast<int>(population.size() * 0.1));
                for (int i = 0; i < elite_count; i++) {
                    targets.push_back(i);
                }
                
                // Add some random individuals
                int random_count = std::max(1, static_cast<int>(population.size() * 0.05));
                std::uniform_int_distribution<int> dist(elite_count, population.size() - 1);
                
                for (int i = 0; i < random_count; i++) {
                    int index = dist(rng);
                    if (std::find(targets.begin(), targets.end(), index) == targets.end()) {
                        targets.push_back(index);
                    }
                }
                break;
            }
            
            default:
                break;
        }
        
        // Always include best if configured
        if (config.apply_to_best && !population.empty()) {
            if (std::find(targets.begin(), targets.end(), 0) == targets.end()) {
                targets.insert(targets.begin(), 0);
            }
        }
        
        return targets;
    }
    
    // Get number of local search algorithms
    size_t get_search_count() const {
        return local_searches.size();
    }
    
    // Check if any local searches are configured
    bool has_local_searches() const {
        return !local_searches.empty() && config.strategy != LocalSearchStrategy::DISABLED;
    }
    
    // Print statistics for all local searches
    void print_all_statistics() const {
        if (local_searches.empty()) {
            std::cout << "No local search algorithms configured." << std::endl;
            return;
        }
        
        std::cout << "\n=========================================" << std::endl;
        std::cout << "    Local Search Statistics Summary" << std::endl;
        std::cout << "=========================================" << std::endl;
        
        for (const auto& search : local_searches) {
            search->print_statistics();
        }
        
        std::cout << "Generations without improvement: " << generations_without_improvement << std::endl;
        std::cout << "=========================================" << std::endl;
    }
    
    // Reset statistics for all local searches
    void reset_all_statistics() {
        for (auto& search : local_searches) {
            search->reset_statistics();
        }
        generations_without_improvement = 0;
        first_generation = true;
    }
    
    // Configure all local searches with same parameters
    void configure_all(const std::map<std::string, std::string>& params) {
        for (auto& search : local_searches) {
            search->configure(params);
        }
    }
    
    // Get local search by name
    LocalSearch<T>* get_local_search(const std::string& name) {
        for (auto& search : local_searches) {
            if (search->get_name() == name) {
                return search.get();
            }
        }
        return nullptr;
    }
    
private:
    T get_best_fitness(const std::vector<Individual<T>>& population) {
        if (population.empty()) {
            return T(0);
        }
        
        T best = population[0].fitness;
        for (const auto& individual : population) {
            if (comparator(individual.fitness, best)) {
                best = individual.fitness;
            }
        }
        return best;
    }
};

#endif // LOCAL_SEARCH_MANAGER_HPP
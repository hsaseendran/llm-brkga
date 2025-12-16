// core/individual.hpp - EXTENDED for multi-objective support
#ifndef INDIVIDUAL_HPP
#define INDIVIDUAL_HPP

#include <vector>
#include <random>
#include <algorithm>
#include <limits>

template<typename T>
class Individual {
public:
    std::vector<std::vector<T>> chromosomes;  // Multiple chromosomes for multi-component problems
    
    // Single-objective support (backward compatible)
    T fitness;
    bool evaluated;
    
    // Multi-objective support (NEW)
    std::vector<T> objectives;           // Multiple objective values
    int rank;                            // Non-domination rank (0 = Pareto front)
    T crowding_distance;                 // Diversity measure
    
    // Internal NSGA-II data
    std::vector<int> dominated_solutions;  // Indices of solutions this dominates
    int domination_count;                  // How many solutions dominate this one
    
    // Single-component constructor (backward compatibility)
    Individual(int length) 
        : chromosomes(1, std::vector<T>(length)), 
          fitness(T(0)), evaluated(false),
          rank(-1), crowding_distance(0), domination_count(0) {}
    
    // Multi-component constructor
    Individual(const std::vector<int>& component_lengths) 
        : fitness(T(0)), evaluated(false),
          rank(-1), crowding_distance(0), domination_count(0) {
        chromosomes.resize(component_lengths.size());
        for (size_t i = 0; i < component_lengths.size(); i++) {
            chromosomes[i].resize(component_lengths[i]);
        }
    }
    
    // Multi-objective constructor
    Individual(const std::vector<int>& component_lengths, int num_objectives)
        : fitness(T(0)), evaluated(false),
          rank(-1), crowding_distance(0), domination_count(0) {
        chromosomes.resize(component_lengths.size());
        for (size_t i = 0; i < component_lengths.size(); i++) {
            chromosomes[i].resize(component_lengths[i]);
        }
        objectives.resize(num_objectives, T(0));
    }
    
    Individual(const Individual& other) 
        : chromosomes(other.chromosomes), 
          fitness(other.fitness), 
          evaluated(other.evaluated),
          objectives(other.objectives),
          rank(other.rank),
          crowding_distance(other.crowding_distance),
          dominated_solutions(other.dominated_solutions),
          domination_count(other.domination_count) {}
    
    Individual& operator=(const Individual& other) {
        if (this != &other) {
            chromosomes = other.chromosomes;
            fitness = other.fitness;
            evaluated = other.evaluated;
            objectives = other.objectives;
            rank = other.rank;
            crowding_distance = other.crowding_distance;
            dominated_solutions = other.dominated_solutions;
            domination_count = other.domination_count;
        }
        return *this;
    }
    
    void randomize(std::mt19937& rng) {
        std::uniform_real_distribution<T> dist(0.0, 1.0);
        for (auto& chromosome : chromosomes) {
            for (auto& gene : chromosome) {
                gene = dist(rng);
            }
        }
        evaluated = false;
    }
    
    void reset_evaluation() {
        evaluated = false;
    }
    
    // Multi-objective methods
    bool is_multi_objective() const {
        return !objectives.empty();
    }
    
    // Dominance check (for minimization)
    bool dominates(const Individual<T>& other) const {
        if (!is_multi_objective() || objectives.empty() || other.objectives.empty()) {
            return false;
        }
        
        if (objectives.size() != other.objectives.size()) {
            return false;
        }
        
        bool at_least_one_better = false;
        for (size_t i = 0; i < objectives.size(); i++) {
            if (objectives[i] > other.objectives[i]) return false;
            if (objectives[i] < other.objectives[i]) at_least_one_better = true;
        }
        return at_least_one_better;
    }
    
    // For single-objective compatibility
    void set_fitness(T new_fitness) {
        fitness = new_fitness;
        if (is_multi_objective() && objectives.empty()) {
            objectives.resize(1);
            objectives[0] = new_fitness;
        }
        evaluated = true;
    }
    
    // For multi-objective
    void set_objectives(const std::vector<T>& objs) {
        objectives = objs;
        if (!objs.empty()) {
            fitness = objs[0];  // Set fitness to first objective for compatibility
        }
        evaluated = true;
    }
    
    int total_genes() const {
        int total = 0;
        for (const auto& chromosome : chromosomes) {
            total += chromosome.size();
        }
        return total;
    }
    
    int get_total_chromosome_length() const {
        return total_genes();
    }
    
    int num_components() const {
        return chromosomes.size();
    }
    
    bool is_evaluated() const {
        return evaluated;
    }
    
    T get_fitness() const {
        return fitness;
    }
    
    // Backward compatibility
    std::vector<T>& get_chromosome() {
        return chromosomes[0];
    }
    
    const std::vector<T>& get_chromosome() const {
        return chromosomes[0];
    }
    
    std::vector<T>& chromosome() {
        return chromosomes[0];
    }
    
    const std::vector<T>& chromosome() const {
        return chromosomes[0];
    }
    
    // Component access
    std::vector<T>& get_component(int index) {
        return chromosomes[index];
    }
    
    const std::vector<T>& get_component(int index) const {
        return chromosomes[index];
    }
    
    int size() const {
        return chromosomes.empty() ? 0 : chromosomes[0].size();
    }
    
    T& operator[](int index) {
        return chromosomes[0][index];
    }
    
    const T& operator[](int index) const {
        return chromosomes[0][index];
    }
    
    // Flatten/unflatten for GPU operations
    std::vector<T> flatten() const {
        std::vector<T> flat;
        flat.reserve(total_genes());
        for (const auto& chromosome : chromosomes) {
            flat.insert(flat.end(), chromosome.begin(), chromosome.end());
        }
        return flat;
    }
    
    void unflatten(const std::vector<T>& flat_data) {
        int pos = 0;
        for (auto& chromosome : chromosomes) {
            for (auto& gene : chromosome) {
                if (pos < flat_data.size()) {
                    gene = flat_data[pos++];
                }
            }
        }
        evaluated = false;
    }
    
    // Reset NSGA-II specific data
    void reset_nsga2_data() {
        rank = -1;
        crowding_distance = 0;
        dominated_solutions.clear();
        domination_count = 0;
    }
};

#endif // INDIVIDUAL_HPP
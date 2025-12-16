#ifndef KNAPSACK_CONFIG_HPP
#define KNAPSACK_CONFIG_HPP

#include "../core/config.hpp"
#include "../utils/file_utils.hpp"
#include "../utils/random_utils.hpp"
#include <string>
#include <vector>
#include <memory>

template<typename T>
class KnapsackConfig : public BRKGAConfig<T> {
private:
    std::vector<T> weights;
    std::vector<T> values;
    T capacity;
    std::string instance_name;
    
public:
    KnapsackConfig(const std::vector<T>& w, const std::vector<T>& v, T cap, const std::string& name = "Knapsack")
        : BRKGAConfig<T>({static_cast<int>(w.size())}), // Single component
          weights(w), values(v), capacity(cap), instance_name(name) {
        
        if (weights.size() != values.size()) {
            throw std::invalid_argument("Weights and values must have same size");
        }
        
        // FIXED: Set up the problem-specific functions with correct signatures
        this->fitness_function = [this](const Individual<T>& individual) {
            return calculate_knapsack_fitness(individual);
        };
        
        this->decoder = [this](const Individual<T>& individual) {
            return decode_to_solution(individual);
        };
        
        this->comparator = [](T a, T b) { return a > b; }; // Maximization
        
        this->threads_per_block = 256;
        this->update_cuda_grid_size();
    }
    
    T calculate_knapsack_fitness(const Individual<T>& individual) {
        const auto& chromosome = individual.get_chromosome(); // FIXED: Use get_chromosome()
        T total_weight = 0;
        T total_value = 0;
        
        for (size_t i = 0; i < chromosome.size(); i++) {
            if (chromosome[i] > 0.5) { // Item is selected
                total_weight += weights[i];
                total_value += values[i];
            }
        }
        
        // Penalty for exceeding capacity
        if (total_weight > capacity) {
            T penalty = (total_weight - capacity) * 1000;
            return total_value - penalty;
        }
        
        return total_value;
    }
    
    std::vector<std::vector<T>> decode_to_solution(const Individual<T>& individual) { // FIXED: Return type
        const auto& chromosome = individual.get_chromosome();
        
        std::vector<std::vector<T>> result(1); // Single component result
        result[0].reserve(chromosome.size());
        
        for (T gene : chromosome) {
            result[0].push_back(gene > 0.5 ? T(1) : T(0));
        }
        
        return result;
    }
    
    // Helper method for backward compatibility
    std::vector<T> decode_to_selection(const Individual<T>& individual) {
        const auto& chromosome = individual.get_chromosome();
        std::vector<T> selection;
        selection.reserve(chromosome.size());
        
        for (T gene : chromosome) {
            selection.push_back(gene > 0.5 ? T(1) : T(0));
        }
        
        return selection;
    }
    
    void print_solution(const Individual<T>& individual) override {
        auto selection = decode_to_selection(individual);
        T total_weight = 0;
        T total_value = 0;
        int selected_items = 0;
        
        std::cout << "\n=== Knapsack Solution ===" << std::endl;
        std::cout << "Instance: " << instance_name << std::endl;
        std::cout << "Selected items: ";
        for (size_t i = 0; i < selection.size(); i++) {
            if (selection[i] > 0.5) {
                std::cout << i << " ";
                total_weight += weights[i];
                total_value += values[i];
                selected_items++;
            }
        }
        std::cout << std::endl;
        std::cout << "Total weight: " << total_weight << "/" << capacity << std::endl;
        std::cout << "Total value: " << total_value << std::endl;
        std::cout << "Items selected: " << selected_items << "/" << weights.size() << std::endl;
        std::cout << "Capacity utilization: " << std::fixed << std::setprecision(2) 
                  << (total_weight / capacity) * 100 << "%" << std::endl;
        std::cout << "=========================" << std::endl;
    }
    
    bool validate_solution(const Individual<T>& individual) override {
        const auto& chromosome = individual.get_chromosome(); // FIXED: Use get_chromosome()
        T total_weight = 0;
        
        for (size_t i = 0; i < chromosome.size(); i++) {
            if (chromosome[i] > 0.5) {
                total_weight += weights[i];
            }
        }
        
        return total_weight <= capacity;
    }
    
    void export_solution(const Individual<T>& individual, const std::string& filename) override {
        auto selection = decode_to_selection(individual);
        T total_weight = 0;
        T total_value = 0;
        
        FileUtils::ensure_directory(FileUtils::get_directory(filename));
        
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot create solution file: " + filename);
        }
        
        file << "Knapsack Solution for " << instance_name << std::endl;
        file << "Value: " << individual.fitness << std::endl;
        file << "Selected items: ";
        for (size_t i = 0; i < selection.size(); i++) {
            if (selection[i] > 0.5) {
                file << i << " ";
                total_weight += weights[i];
                total_value += values[i];
            }
        }
        file << std::endl;
        file << "Total weight: " << total_weight << "/" << capacity << std::endl;
        file << "Total value: " << total_value << std::endl;
        file << "Capacity utilization: " << std::fixed << std::setprecision(2) 
              << (total_weight / capacity) * 100 << "%" << std::endl;
        file.close();
    }
    
    static std::unique_ptr<KnapsackConfig<T>> create_random(int num_items, T max_weight = 50, 
                                                           T max_value = 100, T capacity_ratio = 0.5,
                                                           const std::string& name = "Random_Knapsack") {
        std::vector<T> weights(num_items);
        std::vector<T> values(num_items);
        T total_weight = 0;
        
        for (int i = 0; i < num_items; i++) {
            weights[i] = RandomUtils::uniform_real<T>(1, max_weight);
            values[i] = RandomUtils::uniform_real<T>(1, max_value);
            total_weight += weights[i];
        }
        
        T capacity = total_weight * capacity_ratio;
        
        std::cout << "Generated random Knapsack with " << num_items << " items, capacity: " 
                  << capacity << " (ratio: " << capacity_ratio << ")" << std::endl;
        
        return std::make_unique<KnapsackConfig<T>>(weights, values, capacity, name + "_" + std::to_string(num_items));
    }
    
    static std::unique_ptr<KnapsackConfig<T>> load_from_file(const std::string& filename) {
        if (!FileUtils::file_exists(filename)) {
            throw std::runtime_error("File does not exist: " + filename);
        }
        
        std::string extension = FileUtils::get_extension(filename);
        
        if (extension == "csv") {
            return load_from_csv(filename);
        } else {
            return load_from_text_file(filename);
        }
    }
    
private:
    static std::unique_ptr<KnapsackConfig<T>> load_from_csv(const std::string& filename) {
        auto lines = FileUtils::read_lines(filename);
        std::vector<T> weights;
        std::vector<T> values;
        T capacity = 0;
        bool found_capacity = false;
        
        std::cout << "Loading Knapsack CSV file: " << filename << std::endl;
        
        for (const auto& line : lines) {
            if (line.empty() || line[0] == '#') continue;
            
            std::vector<std::string> fields = FileUtils::parse_csv_line(line);
            
            if (fields.size() >= 2) {
                if (!found_capacity && fields.size() >= 3) {
                    // First line might contain capacity
                    try {
                        capacity = static_cast<T>(std::stod(fields[2]));
                        found_capacity = true;
                        std::cout << "Found capacity: " << capacity << std::endl;
                    } catch (...) {
                        // If not a number, treat as regular item
                    }
                }
                
                try {
                    T weight = static_cast<T>(std::stod(fields[0]));
                    T value = static_cast<T>(std::stod(fields[1]));
                    weights.push_back(weight);
                    values.push_back(value);
                } catch (const std::exception& e) {
                    std::cout << "Warning: Could not parse line: " << line << std::endl;
                }
            }
        }
        
        if (!found_capacity && !weights.empty()) {
            // Estimate capacity as 50% of total weight
            T total_weight = 0;
            for (T w : weights) total_weight += w;
            capacity = total_weight * 0.5;
            std::cout << "Estimated capacity (50% of total): " << capacity << std::endl;
        }
        
        if (weights.empty()) {
            throw std::runtime_error("No valid items found in CSV file");
        }
        
        std::cout << "Loaded " << weights.size() << " items" << std::endl;
        
        std::string instance_name = FileUtils::get_basename(filename);
        return std::make_unique<KnapsackConfig<T>>(weights, values, capacity, instance_name);
    }
    
    static std::unique_ptr<KnapsackConfig<T>> load_from_text_file(const std::string& filename) {
        auto lines = FileUtils::read_lines(filename);
        std::vector<T> weights;
        std::vector<T> values;
        T capacity;
        int num_items = 0;
        bool reading_items = false;
        
        std::cout << "Loading Knapsack file: " << filename << std::endl;
        
        for (const auto& line : lines) {
            // Skip comments and empty lines
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            
            if (!reading_items) {
                // First line should contain num_items and capacity
                if (iss >> num_items >> capacity) {
                    std::cout << "Items: " << num_items << ", Capacity: " << capacity << std::endl;
                    weights.reserve(num_items);
                    values.reserve(num_items);
                    reading_items = true;
                    continue;
                }
            } else {
                // Read weight and value pairs
                T weight, value;
                if (iss >> weight >> value) {
                    weights.push_back(weight);
                    values.push_back(value);
                    if (weights.size() <= 5) { // Debug first few items
                        std::cout << "Item " << weights.size() << ": weight=" << weight 
                                  << ", value=" << value << std::endl;
                    }
                }
            }
        }
        
        std::cout << "Read " << weights.size() << " items, expected " << num_items << std::endl;
        
        if (weights.size() != static_cast<size_t>(num_items)) {
            std::cout << "Warning: Item count (" << weights.size() 
                      << ") doesn't match expected (" << num_items << ")" << std::endl;
        }
        
        if (weights.empty()) {
            throw std::runtime_error("No items found in file");
        }
        
        // Extract instance name from filename
        std::string instance_name = FileUtils::get_basename(filename);
        
        return std::make_unique<KnapsackConfig<T>>(weights, values, capacity, instance_name);
    }

public:
    static void configure_for_size(KnapsackConfig<T>* config, int num_items) {
        if (num_items <= 50) {
            config->population_size = 200;
            config->elite_size = 40;
            config->mutant_size = 20;
            config->max_generations = 300;
        } else if (num_items <= 100) {
            config->population_size = 400;
            config->elite_size = 80;
            config->mutant_size = 40;
            config->max_generations = 500;
        } else if (num_items <= 500) {
            config->population_size = 800;
            config->elite_size = 160;
            config->mutant_size = 80;
            config->max_generations = 1000;
        } else {
            config->population_size = 1200;
            config->elite_size = 240;
            config->mutant_size = 120;
            config->max_generations = 1500;
        }
        config->elite_prob = 0.7;
        config->update_cuda_grid_size();
    }
    
    // Getters
    const std::vector<T>& get_weights() const { return weights; }
    const std::vector<T>& get_values() const { return values; }
    T get_capacity() const { return capacity; }
    const std::string& get_instance_name() const { return instance_name; }
    
    // Analysis methods
    T get_optimal_upper_bound() const {
        // Calculate linear relaxation upper bound
        std::vector<std::pair<T, int>> value_weight_ratio;
        for (size_t i = 0; i < weights.size(); i++) {
            value_weight_ratio.emplace_back(values[i] / weights[i], i);
        }
        
        std::sort(value_weight_ratio.rbegin(), value_weight_ratio.rend());
        
        T total_value = 0;
        T total_weight = 0;
        
        for (const auto& pair : value_weight_ratio) {
            int item = pair.second;
            if (total_weight + weights[item] <= capacity) {
                total_weight += weights[item];
                total_value += values[item];
            } else {
                // Fractional item for upper bound
                T remaining_capacity = capacity - total_weight;
                total_value += values[item] * (remaining_capacity / weights[item]);
                break;
            }
        }
        
        return total_value;
    }
    
    void print_instance_info() const {
        std::cout << "\n=== Knapsack Instance Info ===" << std::endl;
        std::cout << "Name: " << instance_name << std::endl;
        std::cout << "Items: " << weights.size() << std::endl;
        std::cout << "Capacity: " << capacity << std::endl;
        std::cout << "Linear relaxation upper bound: " << get_optimal_upper_bound() << std::endl;
        
        T total_weight = 0, total_value = 0;
        for (size_t i = 0; i < weights.size(); i++) {
            total_weight += weights[i];
            total_value += values[i];
        }
        
        std::cout << "Total weight of all items: " << total_weight << std::endl;
        std::cout << "Total value of all items: " << total_value << std::endl;
        std::cout << "Capacity ratio: " << std::fixed << std::setprecision(2) 
                  << (capacity / total_weight) * 100 << "%" << std::endl;
        std::cout << "===============================" << std::endl;
    }
};

#endif // KNAPSACK_CONFIG_HPP
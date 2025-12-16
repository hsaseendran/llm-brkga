// configs/multi_tsp_config.hpp - Multi-objective TSP
#ifndef MULTI_TSP_CONFIG_HPP
#define MULTI_TSP_CONFIG_HPP

#include "../core/config.hpp"
#include "../utils/file_utils.hpp"
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <algorithm>

// Multi-objective TSP: Minimize both distance AND time
// Useful when routes have different speeds/traffic conditions

template<typename T>
class MultiTSPConfig : public BRKGAConfig<T> {
private:
    std::vector<std::vector<T>> distance_matrix;
    std::vector<std::vector<T>> time_matrix;
    int num_cities;
    std::string instance_name;
    
public:
    MultiTSPConfig(const std::vector<std::vector<T>>& distances,
                   const std::vector<std::vector<T>>& times,
                   const std::string& name = "MultiTSP")
        : BRKGAConfig<T>({static_cast<int>(distances.size() - 1)}, 2),  // 2 objectives
          distance_matrix(distances),
          time_matrix(times),
          num_cities(distances.size() - 1),
          instance_name(name) {
        
        // Configure for NSGA-II
        this->population_size = 100;
        this->elite_size = 0;      // Not used in NSGA-II
        this->mutant_size = 0;     // Not used in NSGA-II
        this->max_generations = 500;
        this->elite_prob = 0.7;    // Crossover probability
        
        // Objective 1: Minimize total distance
        this->objective_functions[0] = [this](const Individual<T>& ind) {
            return calculate_tour_distance(ind);
        };
        
        // Objective 2: Minimize total time
        this->objective_functions[1] = [this](const Individual<T>& ind) {
            return calculate_tour_time(ind);
        };
        
        this->update_cuda_grid_size();
    }
    
    T calculate_tour_distance(const Individual<T>& individual) const {
        auto tour = decode_tour(individual);
        
        T total_distance = 0;
        int current_city = 0;  // Start at depot
        
        for (int city : tour) {
            total_distance += distance_matrix[current_city][city + 1];
            current_city = city + 1;
        }
        total_distance += distance_matrix[current_city][0];  // Return to depot
        
        return total_distance;
    }
    
    T calculate_tour_time(const Individual<T>& individual) const {
        auto tour = decode_tour(individual);
        
        T total_time = 0;
        int current_city = 0;  // Start at depot
        
        for (int city : tour) {
            total_time += time_matrix[current_city][city + 1];
            current_city = city + 1;
        }
        total_time += time_matrix[current_city][0];  // Return to depot
        
        return total_time;
    }
    
    std::vector<int> decode_tour(const Individual<T>& individual) const {
        const auto& chromosome = individual.get_chromosome();
        
        std::vector<std::pair<T, int>> keyed_cities;
        keyed_cities.reserve(num_cities);
        
        for (int i = 0; i < num_cities; i++) {
            keyed_cities.emplace_back(chromosome[i], i);
        }
        
        std::sort(keyed_cities.begin(), keyed_cities.end());
        
        std::vector<int> tour;
        tour.reserve(num_cities);
        for (const auto& pair : keyed_cities) {
            tour.push_back(pair.second);
        }
        
        return tour;
    }
    
    void print_solution(const Individual<T>& individual) override {
        auto tour = decode_tour(individual);
        
        std::cout << "\n=== Multi-Objective TSP Solution ===" << std::endl;
        std::cout << "Instance: " << instance_name << std::endl;
        std::cout << "Cities: " << num_cities << " (plus depot)" << std::endl;
        
        std::cout << "Tour: 0";
        for (int city : tour) {
            std::cout << " -> " << (city + 1);
        }
        std::cout << " -> 0" << std::endl;
        
        std::cout << "\nObjectives:" << std::endl;
        std::cout << "  Distance: " << std::fixed << std::setprecision(2) 
                  << individual.objectives[0] << std::endl;
        std::cout << "  Time:     " << std::setprecision(2) 
                  << individual.objectives[1] << std::endl;
        std::cout << "====================================" << std::endl;
    }
    
    bool validate_solution(const Individual<T>& individual) override {
        auto tour = decode_tour(individual);
        
        std::vector<bool> visited(num_cities, false);
        for (int city : tour) {
            if (city < 0 || city >= num_cities || visited[city]) {
                return false;
            }
            visited[city] = true;
        }
        return true;
    }
    
    void export_solution(const Individual<T>& individual, const std::string& filename) override {
        auto tour = decode_tour(individual);
        
        FileUtils::ensure_directory(FileUtils::get_directory(filename));
        
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot create solution file: " + filename);
        }
        
        file << "Multi-Objective TSP Solution for " << instance_name << std::endl;
        file << "Distance: " << std::fixed << std::setprecision(6) 
             << individual.objectives[0] << std::endl;
        file << "Time: " << std::setprecision(6) 
             << individual.objectives[1] << std::endl;
        
        file << "Tour: 0";
        for (int city : tour) {
            file << " -> " << (city + 1);
        }
        file << " -> 0" << std::endl;
        
        file.close();
    }
    
    // Generate time matrix from distances with variation
    static std::vector<std::vector<T>> generate_time_matrix(
        const std::vector<std::vector<T>>& distances,
        T speed_min = 0.8,
        T speed_max = 1.5) {
        
        int n = distances.size();
        std::vector<std::vector<T>> times(n, std::vector<T>(n));
        
        std::mt19937 rng(42);  // Fixed seed for reproducibility
        std::uniform_real_distribution<T> speed_dist(speed_min, speed_max);
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j) {
                    times[i][j] = 0;
                } else {
                    T speed_factor = speed_dist(rng);
                    times[i][j] = distances[i][j] * speed_factor;
                }
            }
        }
        
        return times;
    }
    
    static std::unique_ptr<MultiTSPConfig<T>> load_from_file(const std::string& filename) {
        // Load distance matrix from TSP file
        auto tsp_config = load_tsplib(filename);
        
        // Generate time matrix with variation
        auto time_matrix = generate_time_matrix(tsp_config.first, 0.7, 1.8);
        
        std::string instance_name = FileUtils::get_basename(filename);
        
        return std::make_unique<MultiTSPConfig<T>>(
            tsp_config.first, time_matrix, instance_name);
    }
    
private:
    static std::pair<std::vector<std::vector<T>>, std::string> 
    load_tsplib(const std::string& filename) {
        auto lines = FileUtils::read_lines(filename);
        std::vector<std::pair<T, T>> coordinates;
        std::string instance_name;
        int num_cities = 0;
        bool reading_coords = false;
        
        for (const auto& line : lines) {
            if (line.empty()) continue;
            
            if (line.find("NAME") != std::string::npos) {
                size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    instance_name = line.substr(colon + 1);
                    instance_name.erase(0, instance_name.find_first_not_of(" \t"));
                    instance_name.erase(instance_name.find_last_not_of(" \t") + 1);
                }
            }
            else if (line.find("DIMENSION") != std::string::npos) {
                std::istringstream iss(line);
                std::string token;
                while (iss >> token) {
                    if (std::isdigit(token[0])) {
                        num_cities = std::stoi(token);
                        break;
                    }
                }
                coordinates.resize(num_cities);
            }
            else if (line.find("NODE_COORD_SECTION") != std::string::npos) {
                reading_coords = true;
                continue;
            }
            else if (line.find("EOF") != std::string::npos) {
                break;
            }
            else if (reading_coords) {
                std::istringstream iss(line);
                int id;
                T x, y;
                if (iss >> id >> x >> y) {
                    if (id > 0 && id <= num_cities) {
                        coordinates[id - 1] = {x, y};
                    }
                }
            }
        }
        
        // Calculate distance matrix
        std::vector<std::vector<T>> distances(num_cities, std::vector<T>(num_cities));
        
        for (int i = 0; i < num_cities; i++) {
            for (int j = 0; j < num_cities; j++) {
                if (i == j) {
                    distances[i][j] = 0;
                } else {
                    T dx = coordinates[i].first - coordinates[j].first;
                    T dy = coordinates[i].second - coordinates[j].second;
                    distances[i][j] = std::sqrt(dx * dx + dy * dy);
                }
            }
        }
        
        return {distances, instance_name};
    }

public:
    static void configure_for_size(MultiTSPConfig<T>* config, int num_cities) {
        if (num_cities <= 20) {
            config->population_size = 100;
            config->max_generations = 250;
        } else if (num_cities <= 50) {
            config->population_size = 200;
            config->max_generations = 500;
        } else {
            config->population_size = 300;
            config->max_generations = 1000;
        }
        config->update_cuda_grid_size();
    }
    
    int get_num_cities() const { return num_cities; }
    const std::string& get_instance_name() const { return instance_name; }
    
    void print_instance_info() const {
        std::cout << "=== Multi-Objective TSP Instance ===" << std::endl;
        std::cout << "Name: " << instance_name << std::endl;
        std::cout << "Cities: " << num_cities << " (plus depot)" << std::endl;
        std::cout << "Objectives: 2 (distance, time)" << std::endl;
        std::cout << "====================================" << std::endl;
    }
};

#endif // MULTI_TSP_CONFIG_HPP
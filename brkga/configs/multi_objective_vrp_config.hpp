// configs/multi_objective_vrp_config.hpp
// Multi-Objective CVRP: Minimize total distance AND minimize longest route (minimax)
// Objective 1: Total route distance (minimize)
// Objective 2: Maximum single route length (minimize) - ensures fair workload distribution
// This creates a trade-off: balanced routes vs. shortest total distance

#ifndef MULTI_OBJECTIVE_VRP_CONFIG_HPP
#define MULTI_OBJECTIVE_VRP_CONFIG_HPP

#include "../core/config.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <iomanip>

template<typename Float>
class MultiObjectiveVRPConfig : public BRKGAConfig<Float> {
private:
    struct Customer {
        int id;
        Float x, y;
        int demand;
    };
    
    struct Route {
        std::vector<int> customers;
        Float distance;
        int load;
    };
    
    std::vector<Customer> customers;
    int num_customers;
    int num_vehicles;
    int vehicle_capacity;
    std::string instance_name;
    
    // Distance matrix
    std::vector<std::vector<Float>> distances;
    
    // Calculate Euclidean distance
    Float calculate_distance(const Customer& a, const Customer& b) const {
        Float dx = a.x - b.x;
        Float dy = a.y - b.y;
        return std::sqrt(dx * dx + dy * dy);
    }
    
    // Build distance matrix
    void build_distance_matrix() {
        distances.resize(num_customers + 1, std::vector<Float>(num_customers + 1));
        for (int i = 0; i <= num_customers; i++) {
            for (int j = 0; j <= num_customers; j++) {
                if (i == j) {
                    distances[i][j] = 0.0;
                } else {
                    distances[i][j] = calculate_distance(customers[i], customers[j]);
                }
            }
        }
    }

public:
    MultiObjectiveVRPConfig() 
        : BRKGAConfig<Float>({31}, 2) {  // 31 customers, 2 objectives
        // Will be updated after loading file
    }
    
    int get_num_customers() const { return num_customers; }
    int get_num_vehicles() const { return num_vehicles; }
    int get_vehicle_capacity() const { return vehicle_capacity; }
    const std::vector<Customer>& get_customers() const { return customers; }
    const std::vector<std::vector<Float>>& get_distances() const { return distances; }
    
    static std::unique_ptr<MultiObjectiveVRPConfig<Float>> load_from_file(const std::string& filename) {
        auto config = std::make_unique<MultiObjectiveVRPConfig<Float>>();
        std::ifstream file(filename);
        
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open VRP file: " + filename);
        }
        
        std::string line;
        bool in_coord_section = false;
        bool in_demand_section = false;
        int dimension = 0;
        
        // Parse header
        while (std::getline(file, line)) {
            if (line.find("NAME") != std::string::npos) {
                size_t pos = line.find(":");
                if (pos != std::string::npos) {
                    config->instance_name = line.substr(pos + 1);
                    // Trim whitespace
                    config->instance_name.erase(0, config->instance_name.find_first_not_of(" \t"));
                }
            }
            else if (line.find("DIMENSION") != std::string::npos) {
                std::istringstream iss(line);
                std::string dummy;
                char colon;
                iss >> dummy >> colon >> dimension;
                config->num_customers = dimension - 1; // Subtract depot
            }
            else if (line.find("CAPACITY") != std::string::npos) {
                std::istringstream iss(line);
                std::string dummy;
                char colon;
                iss >> dummy >> colon >> config->vehicle_capacity;
            }
            else if (line.find("NODE_COORD_SECTION") != std::string::npos) {
                in_coord_section = true;
                config->customers.resize(dimension);
                break;
            }
        }
        
        // Read coordinates
        if (in_coord_section) {
            for (int i = 0; i < dimension; i++) {
                int id;
                Float x, y;
                file >> id >> x >> y;
                config->customers[id - 1] = {id - 1, x, y, 0};
            }
        }
        
        // Read demands
        while (std::getline(file, line)) {
            if (line.find("DEMAND_SECTION") != std::string::npos) {
                in_demand_section = true;
                break;
            }
        }
        
        if (in_demand_section) {
            for (int i = 0; i < dimension; i++) {
                int id, demand;
                file >> id >> demand;
                config->customers[id - 1].demand = demand;
            }
        }
        
        file.close();
        
        // Extract number of vehicles from instance name (e.g., A-n32-k5 -> 5 vehicles)
        size_t k_pos = config->instance_name.find("-k");
        if (k_pos != std::string::npos) {
            config->num_vehicles = std::stoi(config->instance_name.substr(k_pos + 2));
        } else {
            // Default estimate: one vehicle per 6 customers
            config->num_vehicles = std::max(3, config->num_customers / 6);
        }
        
        // Build distance matrix
        config->build_distance_matrix();
        
        // Update component lengths with actual customer count
        config->component_lengths = {config->num_customers};
        config->num_components = 1;
        
        // Set up objective functions
        config->objective_functions.resize(2);
        config->objective_functions[0] = [config_ptr = config.get()](const Individual<Float>& ind) {
            return config_ptr->calculate_total_distance(ind);
        };
        config->objective_functions[1] = [config_ptr = config.get()](const Individual<Float>& ind) {
            return config_ptr->calculate_max_route_length(ind);
        };
        
        return config;
    }
    
    void print_instance_info() const {
        std::cout << "\n=== VRP Instance Information ===" << std::endl;
        std::cout << "Instance: " << instance_name << std::endl;
        std::cout << "Customers: " << num_customers << std::endl;
        std::cout << "Vehicles: " << num_vehicles << std::endl;
        std::cout << "Vehicle capacity: " << vehicle_capacity << std::endl;
        std::cout << "Chromosome size: " << num_customers << std::endl;
        std::cout << "Optimization: Multi-Objective (Distance vs. Max Route)" << std::endl;
        std::cout << "  Objective 1: Total distance (minimize)" << std::endl;
        std::cout << "  Objective 2: Longest single route (minimize - minimax)" << std::endl;
        std::cout << "\nTrade-off explanation:" << std::endl;
        std::cout << "  - Minimize total: May create very long routes for some vehicles" << std::endl;
        std::cout << "  - Minimize longest: Ensures no vehicle is overworked (fairness)" << std::endl;
    }
    
    static void configure_for_size(MultiObjectiveVRPConfig<Float>* config, int num_customers) {
        // Multi-objective typically needs larger populations
        // INCREASED further to maintain better diversity
        if (num_customers <= 50) {
            config->population_size = 1000;  // Doubled from 500
            config->max_generations = 1500;  // Increased from 1000
        } else if (num_customers <= 100) {
            config->population_size = 1500;  // Increased
            config->max_generations = 2000;
        } else {
            config->population_size = 2000;
            config->max_generations = 2500;
        }
        
        // NSGA-II specific parameters - tuned for better exploration
        config->elite_size = config->population_size / 5;
        config->mutant_size = config->population_size / 6;  // Even more mutants
        config->elite_prob = 0.10f;  // Very low crossover for maximum diversity
        
        config->update_cuda_grid_size();
    }
    
    // Decode chromosome into routes
    std::vector<Route> decode_routes(const Float* chromosome) const {
        std::vector<Route> routes;
        
        // Create a sorted list of customers based on chromosome values
        std::vector<std::pair<Float, int>> chr_customers;
        
        // Copy chromosome values with hash-based tie-breaker
        Float chromosome_hash = 0.0f;
        for (int i = 0; i < num_customers; i++) {
            chromosome_hash += chromosome[i] * (i + 1);
        }
        
        for (int i = 0; i < num_customers; i++) {
            Float position_bias = (i * 0.00001f) + (chromosome_hash * 0.00001f);
            Float value = chromosome[i] + position_bias;
            chr_customers.push_back({value, i + 1}); // Customer IDs (1-based, 0 is depot)
        }
        
        // Sort customers by chromosome values
        std::sort(chr_customers.begin(), chr_customers.end());
        
        // Build routes sequentially with capacity constraints
        Route current_route;
        current_route.load = 0;
        current_route.distance = 0.0f;
        int last_customer = 0; // Start from depot
        
        for (const auto& [chr_val, customer] : chr_customers) {
            int demand = customers[customer].demand;
            
            // Check if adding this customer exceeds capacity
            if (current_route.load + demand > vehicle_capacity && !current_route.customers.empty()) {
                // Close current route (return to depot)
                current_route.distance += distances[last_customer][0];
                routes.push_back(current_route);
                
                // Start new route
                current_route.customers.clear();
                current_route.load = 0;
                current_route.distance = 0.0f;
                last_customer = 0;
            }
            
            // Add customer to current route
            current_route.distance += distances[last_customer][customer];
            current_route.load += demand;
            current_route.customers.push_back(customer);
            last_customer = customer;
        }
        
        // Close final route
        if (!current_route.customers.empty()) {
            current_route.distance += distances[last_customer][0];
            routes.push_back(current_route);
        }
        
        return routes;
    }
    
    // Calculate objectives
    Float calculate_total_distance(const Individual<Float>& individual) const {
        Float objectives[2];
        decode_solution(individual.get_chromosome().data(), objectives);
        return objectives[0];
    }
    
    Float calculate_max_route_length(const Individual<Float>& individual) const {
        Float objectives[2];
        decode_solution(individual.get_chromosome().data(), objectives);
        return objectives[1];
    }
    
    void decode_solution(const Float* chromosome, Float* objectives) const {
        auto routes = decode_routes(chromosome);
        
        // Calculate objectives
        Float total_distance = 0.0f;
        Float max_route_length = 0.0f;
        Float capacity_penalty = 0.0f;
        
        for (const auto& route : routes) {
            total_distance += route.distance;
            if (route.distance > max_route_length) {
                max_route_length = route.distance;
            }
            if (route.load > vehicle_capacity) {
                capacity_penalty += (route.load - vehicle_capacity) * 100.0f;
            }
        }
        
        // Set objectives
        objectives[0] = total_distance + capacity_penalty;
        objectives[1] = max_route_length;
        
        // Penalty for using too many vehicles
        if (routes.size() > (size_t)num_vehicles) {
            Float vehicle_penalty = (routes.size() - num_vehicles) * 500.0f;
            objectives[0] += vehicle_penalty;
            objectives[1] += vehicle_penalty;
        }
    }
    
    void export_solution(const Individual<Float>& individual, 
                        const std::string& filename) const {
        std::ofstream out(filename);
        if (!out.is_open()) {
            throw std::runtime_error("Cannot open output file: " + filename);
        }
        
        const auto& chromosome_data = individual.chromosome();
        auto routes = decode_routes(chromosome_data.data());
        
        out << "Multi-Objective VRP Solution for: " << instance_name << "\n";
        out << "Objective 1 (Total Distance): " << std::fixed << std::setprecision(2) 
            << individual.objectives[0] << "\n";
        out << "Objective 2 (Max Route Length): " << std::fixed << std::setprecision(2)
            << individual.objectives[1] << "\n";
        out << "Number of routes: " << routes.size() << "\n";
        out << "\n";
        
        // Export routes
        Float total_distance = 0.0f;
        Float max_route = 0.0f;
        
        for (size_t r = 0; r < routes.size(); r++) {
            const auto& route = routes[r];
            out << "Route " << (r + 1) << ": 0";
            for (int customer : route.customers) {
                out << " -> " << customer;
            }
            out << " -> 0\n";
            out << "  Distance: " << std::fixed << std::setprecision(2) << route.distance << "\n";
            out << "  Load: " << route.load << " / " << vehicle_capacity << "\n";
            out << "  Utilization: " << std::fixed << std::setprecision(1) 
                << (100.0 * route.load / vehicle_capacity) << "%\n";
            out << "\n";
            
            total_distance += route.distance;
            if (route.distance > max_route) {
                max_route = route.distance;
            }
        }
        
        out << "=== Summary ===\n";
        out << "Total distance: " << std::fixed << std::setprecision(2) << total_distance << "\n";
        out << "Longest route: " << std::fixed << std::setprecision(2) << max_route << "\n";
        out << "Average route: " << std::fixed << std::setprecision(2) 
            << (total_distance / routes.size()) << "\n";
        out << "Balance ratio (max/avg): " << std::fixed << std::setprecision(2)
            << (max_route / (total_distance / routes.size())) << "\n";
        
        out.close();
    }
    
    // Export all Pareto front solutions
    void export_all_pareto_solutions(const std::vector<Individual<Float>>& pareto_front,
                                     const std::string& directory) {
        // Create directory
        std::string cmd = "mkdir -p " + directory;
        system(cmd.c_str());
        
        std::cout << "\nExporting " << pareto_front.size() << " Pareto solutions..." << std::endl;
        
        for (size_t i = 0; i < pareto_front.size(); i++) {
            std::stringstream filename;
            filename << directory << "/solution_" << std::setw(3) << std::setfill('0') << i << ".txt";
            export_solution(pareto_front[i], filename.str());
        }
        
        // Export route coordinates for visualization
        export_pareto_routes_data(pareto_front, directory + "/routes_data.txt");
        
        std::cout << "✓ All solutions exported to: " << directory << "/" << std::endl;
    }
    
    // Export route data for visualization
    void export_pareto_routes_data(const std::vector<Individual<Float>>& pareto_front,
                                   const std::string& filename) const {
        std::ofstream out(filename);
        if (!out.is_open()) {
            throw std::runtime_error("Cannot open output file: " + filename);
        }
        
        out << "# VRP Routes Data for Visualization\n";
        out << "# Instance: " << instance_name << "\n";
        out << "# Customers: " << num_customers << "\n";
        out << "# Vehicles: " << num_vehicles << "\n";
        out << "#\n";
        out << "# Format:\n";
        out << "# SOLUTION <idx> <obj1> <obj2>\n";
        out << "# ROUTE <route_idx> <distance> <load>\n";
        out << "# CUSTOMER <customer_id> <x> <y> <demand>\n";
        out << "#\n\n";
        
        for (size_t sol_idx = 0; sol_idx < pareto_front.size(); sol_idx++) {
            const auto& individual = pareto_front[sol_idx];
            const auto& chromosome = individual.chromosome();
            auto routes = decode_routes(chromosome.data());
            
            out << "SOLUTION " << sol_idx << " " 
                << std::fixed << std::setprecision(2)
                << individual.objectives[0] << " "
                << individual.objectives[1] << "\n";
            
            for (size_t r = 0; r < routes.size(); r++) {
                const auto& route = routes[r];
                out << "ROUTE " << r << " " 
                    << std::fixed << std::setprecision(2)
                    << route.distance << " " << route.load << "\n";
                
                // Output depot
                out << "CUSTOMER 0 " 
                    << customers[0].x << " " << customers[0].y << " " << customers[0].demand << "\n";
                
                // Output customers in route
                for (int cust_id : route.customers) {
                    const auto& cust = customers[cust_id];
                    out << "CUSTOMER " << cust_id << " " 
                        << cust.x << " " << cust.y << " " << cust.demand << "\n";
                }
                
                // Return to depot
                out << "CUSTOMER 0 " 
                    << customers[0].x << " " << customers[0].y << " " << customers[0].demand << "\n";
                
                out << "END_ROUTE\n";
            }
            
            out << "END_SOLUTION\n\n";
        }
        
        out.close();
    }
};

#endif // MULTI_OBJECTIVE_VRP_CONFIG_HPP
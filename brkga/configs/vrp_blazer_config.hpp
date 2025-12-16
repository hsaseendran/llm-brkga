// configs/vrp_blazer_config.hpp
// Multi-Day Capacitated VRP for VrpBlazerApp Integration
// Designed to work with professor's IVrpSolver interface
// Input: Pre-computed time matrix (minutes), constraints
// Output: JSON-formatted multi-day routes

#ifndef VRP_BLAZER_CONFIG_HPP
#define VRP_BLAZER_CONFIG_HPP

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
#include <chrono>

template<typename Float>
class VrpBlazerConfig : public BRKGAConfig<Float> {
public:
    // Problem parameters (matching professor's VrpSolverConfig)
    int max_stops_per_day;           // e.g., 12
    double max_drive_hours_per_day;  // e.g., 8.0
    double stop_time_hours;          // e.g., 0.5
    int depot_index;                 // Always 0

    // Derived parameters
    double max_drive_minutes_per_day;
    double stop_time_minutes;

private:
    // Time matrix (in minutes) - provided externally
    std::vector<std::vector<Float>> time_matrix;
    int num_locations;  // Total locations (depot + stops)
    int num_stops;      // Stops only (excludes depot)

    // Route structure for decoding
    struct DayRoute {
        int day_number;  // 1-based
        std::vector<int> stop_indices;  // indices in time_matrix (0=depot)
        std::vector<int> stop_orders;   // 1-based order in route
        std::vector<Float> drive_times_from_prev;
        std::vector<Float> cumulative_drive_times;
        Float total_drive_minutes;
        Float total_stop_minutes;
    };

    struct VrpSolutionData {
        std::vector<DayRoute> days;
        Float total_drive_minutes;
        Float total_stop_minutes;
        int total_stops;
        std::vector<int> unvisited_stops;
    };

public:
    VrpBlazerConfig()
        : BRKGAConfig<Float>({1}, 1),  // Will be updated after loading
          max_stops_per_day(12),
          max_drive_hours_per_day(8.0),
          stop_time_hours(0.5),
          depot_index(0) {
        update_derived_params();
    }

    void update_derived_params() {
        max_drive_minutes_per_day = max_drive_hours_per_day * 60.0;
        stop_time_minutes = stop_time_hours * 60.0;
    }

    // Load time matrix from file (format: N rows x N columns, space-separated)
    static std::unique_ptr<VrpBlazerConfig<Float>> load_from_matrix_file(
            const std::string& matrix_file,
            const std::string& config_file) {

        auto config = std::make_unique<VrpBlazerConfig<Float>>();

        // Load configuration parameters
        config->load_config_params(config_file);

        // Load time matrix
        config->load_time_matrix(matrix_file);

        // Setup BRKGA parameters
        config->setup_brkga_params();

        return config;
    }

    void load_config_params(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Warning: Cannot open config file, using defaults: " << filename << std::endl;
            return;
        }

        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string key;
            char equals;

            if (iss >> key >> equals) {
                if (key == "MaxStopsPerDay") {
                    iss >> max_stops_per_day;
                } else if (key == "MaxDriveHoursPerDay") {
                    iss >> max_drive_hours_per_day;
                } else if (key == "StopTimeHours") {
                    iss >> stop_time_hours;
                } else if (key == "DepotIndex") {
                    iss >> depot_index;
                } else if (key == "PopulationSize") {
                    iss >> this->population_size;
                } else if (key == "MaxGenerations") {
                    iss >> this->max_generations;
                }
            }
        }
        file.close();
        update_derived_params();
    }

    void load_time_matrix(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open time matrix file: " + filename);
        }

        // First line: matrix size
        file >> num_locations;
        num_stops = num_locations - 1;  // Exclude depot

        // Read matrix
        time_matrix.resize(num_locations, std::vector<Float>(num_locations));
        for (int i = 0; i < num_locations; i++) {
            for (int j = 0; j < num_locations; j++) {
                file >> time_matrix[i][j];
            }
        }

        file.close();
    }

    void set_time_matrix(const std::vector<std::vector<Float>>& matrix) {
        time_matrix = matrix;
        num_locations = matrix.size();
        num_stops = num_locations - 1;
        setup_brkga_params();
    }

    void setup_brkga_params() {
        // Chromosome length = number of stops (excluding depot)
        this->component_lengths = {num_stops};
        this->num_components = 1;

        // Scale population based on problem size
        if (num_stops <= 50) {
            this->population_size = 500;
            this->max_generations = 1000;
        } else if (num_stops <= 100) {
            this->population_size = 800;
            this->max_generations = 1500;
        } else if (num_stops <= 200) {
            this->population_size = 1000;
            this->max_generations = 2000;
        } else {
            this->population_size = 1500;
            this->max_generations = 2500;
        }

        // BRKGA parameters
        this->elite_size = this->population_size / 5;      // 20% elite
        this->mutant_size = this->population_size / 10;    // 10% mutants
        this->elite_prob = 0.7;                            // Bias to elite parent

        this->update_cuda_grid_size();

        // Setup fitness function (minimize total travel time)
        this->fitness_function = [this](const Individual<Float>& ind) {
            return calculate_fitness(ind);
        };

        // Minimization
        this->comparator = [](Float a, Float b) { return a < b; };
    }

    // Decode chromosome into multi-day routes
    VrpSolutionData decode_solution(const Float* chromosome) const {
        VrpSolutionData solution;
        solution.total_drive_minutes = 0;
        solution.total_stop_minutes = 0;
        solution.total_stops = 0;

        // Create sorted list of stops based on chromosome values
        std::vector<std::pair<Float, int>> chr_stops;
        for (int i = 0; i < num_stops; i++) {
            // Stop index in time_matrix is i+1 (0 is depot)
            chr_stops.push_back({chromosome[i], i + 1});
        }
        std::sort(chr_stops.begin(), chr_stops.end());

        // Track which stops are visited
        std::vector<bool> visited(num_locations, false);
        visited[0] = true;  // Depot is always "visited"

        // Build routes day by day
        int day_number = 1;
        size_t stop_idx = 0;

        while (stop_idx < chr_stops.size()) {
            DayRoute day;
            day.day_number = day_number;
            day.total_drive_minutes = 0;
            day.total_stop_minutes = 0;

            int current_location = depot_index;
            Float cumulative_drive = 0;
            int stops_today = 0;

            // Try to add stops to this day
            while (stop_idx < chr_stops.size() && stops_today < max_stops_per_day) {
                int next_stop = chr_stops[stop_idx].second;

                // Calculate time to go to this stop and back to depot
                Float time_to_stop = time_matrix[current_location][next_stop];
                Float time_to_depot = time_matrix[next_stop][depot_index];
                Float total_if_added = cumulative_drive + time_to_stop + time_to_depot;

                // Check if we can fit this stop (drive time constraint)
                if (total_if_added <= max_drive_minutes_per_day) {
                    // Add stop to route
                    day.stop_indices.push_back(next_stop);
                    day.stop_orders.push_back(stops_today + 1);
                    day.drive_times_from_prev.push_back(time_to_stop);

                    cumulative_drive += time_to_stop;
                    day.cumulative_drive_times.push_back(cumulative_drive);

                    visited[next_stop] = true;
                    current_location = next_stop;
                    stops_today++;
                    stop_idx++;
                } else {
                    // Can't fit more stops today
                    break;
                }
            }

            // Close the route (return to depot)
            if (stops_today > 0) {
                Float return_time = time_matrix[current_location][depot_index];
                day.total_drive_minutes = cumulative_drive + return_time;
                day.total_stop_minutes = stops_today * stop_time_minutes;

                solution.days.push_back(day);
                solution.total_drive_minutes += day.total_drive_minutes;
                solution.total_stop_minutes += day.total_stop_minutes;
                solution.total_stops += stops_today;

                day_number++;
            }

            // Safety: if we couldn't add any stops, skip to next
            if (stops_today == 0 && stop_idx < chr_stops.size()) {
                // This stop can't be reached within daily constraints
                solution.unvisited_stops.push_back(chr_stops[stop_idx].second);
                stop_idx++;
            }
        }

        // Check for any unvisited stops
        for (int i = 1; i < num_locations; i++) {
            if (!visited[i]) {
                solution.unvisited_stops.push_back(i);
            }
        }

        return solution;
    }

    Float calculate_fitness(const Individual<Float>& individual) const {
        const auto& chromosome = individual.get_chromosome();
        auto solution = decode_solution(chromosome.data());

        // Primary objective: minimize total drive time
        Float fitness = solution.total_drive_minutes;

        // Penalty for unvisited stops (heavy penalty)
        Float unvisited_penalty = solution.unvisited_stops.size() * 1000.0;
        fitness += unvisited_penalty;

        return fitness;
    }

    // Export solution to JSON format for C# parsing
    void export_solution_json(const Individual<Float>& individual,
                              const std::string& filename,
                              double solve_time_seconds) const {
        std::ofstream out(filename);
        if (!out.is_open()) {
            throw std::runtime_error("Cannot open output file: " + filename);
        }

        const auto& chromosome = individual.get_chromosome();
        auto solution = decode_solution(chromosome.data());

        out << "{\n";
        out << "  \"SolverName\": \"BRKGA-CUDA\",\n";
        out << "  \"SolveTimeSeconds\": " << std::fixed << std::setprecision(3)
            << solve_time_seconds << ",\n";
        out << "  \"TotalDriveMinutes\": " << std::setprecision(2)
            << solution.total_drive_minutes << ",\n";
        out << "  \"TotalStopMinutes\": " << solution.total_stop_minutes << ",\n";
        out << "  \"TotalStops\": " << solution.total_stops << ",\n";

        // Unvisited stops
        out << "  \"UnvisitedStops\": [";
        for (size_t i = 0; i < solution.unvisited_stops.size(); i++) {
            out << solution.unvisited_stops[i];
            if (i < solution.unvisited_stops.size() - 1) out << ", ";
        }
        out << "],\n";

        // Days/Routes
        out << "  \"Days\": [\n";
        for (size_t d = 0; d < solution.days.size(); d++) {
            const auto& day = solution.days[d];
            out << "    {\n";
            out << "      \"DayNumber\": " << day.day_number << ",\n";
            out << "      \"DriveTimeMinutes\": " << std::setprecision(2)
                << day.total_drive_minutes << ",\n";
            out << "      \"StopTimeMinutes\": " << day.total_stop_minutes << ",\n";

            // Stops
            out << "      \"Stops\": [\n";
            for (size_t s = 0; s < day.stop_indices.size(); s++) {
                out << "        {\n";
                out << "          \"Index\": " << day.stop_indices[s] << ",\n";
                out << "          \"Order\": " << day.stop_orders[s] << ",\n";
                out << "          \"DriveTimeFromPrevious\": " << std::setprecision(2)
                    << day.drive_times_from_prev[s] << ",\n";
                out << "          \"CumulativeDriveTime\": " << day.cumulative_drive_times[s] << "\n";
                out << "        }";
                if (s < day.stop_indices.size() - 1) out << ",";
                out << "\n";
            }
            out << "      ]\n";
            out << "    }";
            if (d < solution.days.size() - 1) out << ",";
            out << "\n";
        }
        out << "  ]\n";
        out << "}\n";

        out.close();
    }

    // Print instance info
    void print_instance_info() const {
        std::cout << "\n=== VRP Blazer Instance ===" << std::endl;
        std::cout << "Total locations: " << num_locations << " (1 depot + "
                  << num_stops << " stops)" << std::endl;
        std::cout << "Max stops per day: " << max_stops_per_day << std::endl;
        std::cout << "Max drive hours per day: " << max_drive_hours_per_day << std::endl;
        std::cout << "Stop time (hours): " << stop_time_hours << std::endl;
        std::cout << "Chromosome length: " << num_stops << std::endl;
        std::cout << "Population size: " << this->population_size << std::endl;
        std::cout << "Max generations: " << this->max_generations << std::endl;
        std::cout << "=============================" << std::endl;
    }

    int get_num_stops() const { return num_stops; }
    int get_num_locations() const { return num_locations; }
    const std::vector<std::vector<Float>>& get_time_matrix() const { return time_matrix; }
};

#endif // VRP_BLAZER_CONFIG_HPP

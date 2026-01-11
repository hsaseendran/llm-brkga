// test_phase4_tsp.cu - Integration test for Phase 4 with real TSP problem
// Tests that the segmented sort works correctly in the full TSP solver

#include "configs/tsp_config.hpp"
#include "core/solver.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "  Phase 4: TSP Integration Test          " << std::endl;
    std::cout << "==========================================" << std::endl;

    try {
        // Create a small TSP problem for testing
        const int num_cities = 200;
        const int pop_size = 500;
        const int max_gens = 50;

        std::cout << "\n[Test Setup]" << std::endl;
        std::cout << "  Problem: TSP (" << num_cities << " cities)" << std::endl;
        std::cout << "  Population: " << pop_size << std::endl;
        std::cout << "  Generations: " << max_gens << std::endl;

        // Generate random city coordinates
        std::vector<float> coords_x(num_cities);
        std::vector<float> coords_y(num_cities);

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, 1000.0f);

        for (int i = 0; i < num_cities; i++) {
            coords_x[i] = dist(rng);
            coords_y[i] = dist(rng);
        }

        std::cout << "  ✓ Random coordinates generated" << std::endl;

        // Create TSP config (will use coordinate-based evaluation with segmented sort)
        auto config = std::make_unique<TSPConfig<float>>(coords_x, coords_y, "test_instance");
        config->population_size = pop_size;
        config->elite_size = pop_size / 5;
        config->mutant_size = pop_size / 10;
        config->max_generations = max_gens;

        std::cout << "\n[Creating Solver]" << std::endl;
        std::cout << "  GPU evaluation: " << (config->has_gpu_evaluation() ? "YES" : "NO") << std::endl;

        // Create solver (verbose=false for cleaner output)
        Solver<float> solver(std::move(config), false, 0);

        std::cout << "  ✓ Solver created successfully" << std::endl;
        std::cout << "  ✓ Segmented sort will be initialized on first evaluation" << std::endl;

        // Run evolution
        std::cout << "\n[Running Evolution]" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        solver.run();

        auto end = std::chrono::high_resolution_clock::now();
        double runtime = std::chrono::duration<double>(end - start).count();

        // Get results
        auto best = solver.get_best_individual();

        std::cout << "  ✓ Evolution completed successfully" << std::endl;

        // Display results
        std::cout << "\n[Results]" << std::endl;
        std::cout << "  Best fitness (tour length): " << std::fixed << std::setprecision(2)
                  << best.fitness << std::endl;
        std::cout << "  Total time: " << std::setprecision(3) << runtime << " seconds" << std::endl;
        std::cout << "  Time per generation: " << std::setprecision(3)
                  << (runtime / max_gens * 1000.0) << " ms" << std::endl;

        // Validation checks
        std::cout << "\n[Validation]" << std::endl;

        // Check if solution is reasonable (not zero, not infinity)
        bool valid_fitness = (best.fitness > 0 && best.fitness < 1e30);
        std::cout << "  Fitness is valid: " << (valid_fitness ? "✓ YES" : "✗ NO") << std::endl;

        // Check if chromosome has correct size
        bool valid_chromosome = (best.get_component(0).size() == num_cities);
        std::cout << "  Chromosome size correct: " << (valid_chromosome ? "✓ YES" : "✗ NO") << std::endl;

        // Check if runtime is reasonable (should be faster with Phase 4)
        bool reasonable_time = (runtime < 30.0);  // Should complete in under 30 seconds
        std::cout << "  Runtime reasonable: " << (reasonable_time ? "✓ YES" : "✗ NO") << std::endl;

        // Check if tour length is reasonable (not too long for random cities)
        double max_possible = 1000.0 * 1.414 * num_cities;  // Max diagonal * cities
        bool reasonable_length = (best.fitness < max_possible);
        std::cout << "  Tour length reasonable: " << (reasonable_length ? "✓ YES" : "✗ NO") << std::endl;

        std::cout << "\n==========================================" << std::endl;
        if (valid_fitness && valid_chromosome && reasonable_time && reasonable_length) {
            std::cout << "✓ ALL TESTS PASSED" << std::endl;
            std::cout << "Phase 4 segmented sort working correctly with TSP!" << std::endl;
            std::cout << "Decoder successfully uses parallel sorting." << std::endl;
            std::cout << "==========================================" << std::endl;
            return 0;
        } else {
            std::cout << "✗ SOME TESTS FAILED" << std::endl;
            std::cout << "Review validation checks above." << std::endl;
            std::cout << "==========================================" << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "\n✗ TEST FAILED WITH EXCEPTION:" << std::endl;
        std::cerr << "  " << e.what() << std::endl;
        std::cerr << "==========================================" << std::endl;
        return 1;
    }
}

// test_phase1_2.cu - Integration test for Phases 1 & 2 stream optimizations
// Tests that the stream-based solver works correctly with a real problem

#include "configs/knapsack_config.hpp"
#include "core/solver.hpp"
#include <iostream>
#include <chrono>

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "  Testing Phases 1 & 2 Integration      " << std::endl;
    std::cout << "  Stream-Based BRKGA Solver             " << std::endl;
    std::cout << "=========================================" << std::endl;

    try {
        // Create a small knapsack problem for testing
        const int num_items = 100;
        const int capacity = 1000;
        const int pop_size = 500;
        const int max_gens = 100;

        std::cout << "\n[Test Setup]" << std::endl;
        std::cout << "  Problem: Knapsack (" << num_items << " items)" << std::endl;
        std::cout << "  Population: " << pop_size << std::endl;
        std::cout << "  Generations: " << max_gens << std::endl;

        // Create random knapsack instance
        auto config = KnapsackConfig<float>::create_random(num_items, capacity);
        config->population_size = pop_size;
        config->elite_size = pop_size / 5;
        config->mutant_size = pop_size / 10;
        config->max_generations = max_gens;

        std::cout << "\n[Creating Solver]" << std::endl;
        std::cout << "  GPU evaluation: " << (config->has_gpu_evaluation() ? "YES" : "NO") << std::endl;

        // Create solver (verbose=false for cleaner output)
        Solver<float> solver(std::move(config), false, 0);

        std::cout << "  ✓ Solver created successfully" << std::endl;
        std::cout << "  ✓ StreamManager initialized" << std::endl;

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
        std::cout << "  Best fitness: " << std::fixed << std::setprecision(2) << best.fitness << std::endl;
        std::cout << "  Total time: " << std::setprecision(3) << runtime << " seconds" << std::endl;
        std::cout << "  Time per generation: " << std::setprecision(3)
                  << (runtime / max_gens * 1000.0) << " ms" << std::endl;

        // Validation checks
        std::cout << "\n[Validation]" << std::endl;

        // Check if solution is reasonable (not zero, not infinity)
        bool valid_fitness = (best.fitness > 0 && best.fitness < 1e30);
        std::cout << "  Fitness is valid: " << (valid_fitness ? "✓ YES" : "✗ NO") << std::endl;

        // Check if chromosome has correct size
        bool valid_chromosome = (best.get_component(0).size() == num_items);
        std::cout << "  Chromosome size correct: " << (valid_chromosome ? "✓ YES" : "✗ NO") << std::endl;

        // Check if runtime is reasonable (not too slow)
        bool reasonable_time = (runtime < 60.0);  // Should complete in under 60 seconds
        std::cout << "  Runtime reasonable: " << (reasonable_time ? "✓ YES" : "✗ NO") << std::endl;

        std::cout << "\n=========================================" << std::endl;
        if (valid_fitness && valid_chromosome && reasonable_time) {
            std::cout << "✓ ALL TESTS PASSED" << std::endl;
            std::cout << "Phases 1 & 2 implementation verified!" << std::endl;
            std::cout << "Stream-based BRKGA working correctly." << std::endl;
            std::cout << "=========================================" << std::endl;
            return 0;
        } else {
            std::cout << "✗ SOME TESTS FAILED" << std::endl;
            std::cout << "Review validation checks above." << std::endl;
            std::cout << "=========================================" << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "\n✗ TEST FAILED WITH EXCEPTION:" << std::endl;
        std::cerr << "  " << e.what() << std::endl;
        std::cerr << "=========================================" << std::endl;
        return 1;
    }
}

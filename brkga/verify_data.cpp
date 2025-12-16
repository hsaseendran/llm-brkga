#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

std::vector<std::vector<double>> load_csv(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<double>> matrix;
    std::string line;

    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, ',')) {
            size_t start = cell.find_first_not_of(" \t\r\n");
            size_t end = cell.find_last_not_of(" \t\r\n");
            if (start == std::string::npos) {
                row.push_back(0.0);
            } else {
                cell = cell.substr(start, end - start + 1);
                try {
                    row.push_back(std::stod(cell));
                } catch (...) {
                    row.push_back(0.0);
                }
            }
        }
        if (!row.empty()) {
            matrix.push_back(row);
        }
    }
    return matrix;
}

int main() {
    std::string prefix = "data/Medium_problems/Batch_01/TSPJ_1M";

    auto travel = load_csv(prefix + "_cost_table_by_coordinates.csv");
    auto jobs = load_csv(prefix + "_tasktime_table.csv");

    int n = travel.size() - 1;
    std::cout << "Cities: " << n << std::endl;
    std::cout << "Travel matrix: " << travel.size() << "x" << travel[0].size() << std::endl;
    std::cout << "Jobs matrix: " << jobs.size() << "x" << jobs[0].size() << std::endl;

    // Sample values that BRKGA outputs
    std::cout << "\nSample values to verify against BRKGA:" << std::endl;
    std::cout << "  travel[0][1] = " << travel[0][1] << " (BRKGA shows 154)" << std::endl;
    std::cout << "  jobs[1][1] = " << jobs[1][1] << " (BRKGA shows 1753)" << std::endl;

    // Check the min/max of travel and job durations
    double min_travel = 1e18, max_travel = 0;
    double min_job = 1e18, max_job = 0;

    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= n; j++) {
            if (i != j && travel[i][j] > 0) {
                min_travel = std::min(min_travel, travel[i][j]);
                max_travel = std::max(max_travel, travel[i][j]);
            }
            if (jobs[i][j] > 0) {
                min_job = std::min(min_job, jobs[i][j]);
                max_job = std::max(max_job, jobs[i][j]);
            }
        }
    }

    std::cout << "\nTravel times range: [" << min_travel << ", " << max_travel << "]" << std::endl;
    std::cout << "Job durations range: [" << min_job << ", " << max_job << "]" << std::endl;

    // Sum of all travel times for a rough estimate
    double total_all = 0;
    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= n; j++) {
            total_all += travel[i][j];
        }
    }
    std::cout << "\nSum of all travel times: " << total_all << std::endl;
    std::cout << "Avg travel time: " << total_all / ((n+1) * (n+1)) << std::endl;

    return 0;
}

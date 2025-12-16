#include "tsp_config.hpp"
#include "../core/config.hpp"
#include <memory>
#include <string>

extern "C" std::unique_ptr<BRKGAConfig<float>> create_config(const std::string& filename) {
    return TSPConfig<float>::load_from_file(filename);
}

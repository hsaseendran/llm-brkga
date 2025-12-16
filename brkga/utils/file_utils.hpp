#ifndef FILE_UTILS_HPP
#define FILE_UTILS_HPP

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <map>
#include <chrono>
#include <iomanip>

class FileUtils {
public:
    // Check if file exists
    static bool file_exists(const std::string& filename) {
        return std::filesystem::exists(filename);
    }
    
    // Check if directory exists
    static bool directory_exists(const std::string& dirname) {
        return std::filesystem::exists(dirname) && std::filesystem::is_directory(dirname);
    }
    
    // Create directory if it doesn't exist
    static bool create_directory(const std::string& dirname) {
        try {
            return std::filesystem::create_directories(dirname);
        } catch (const std::exception& e) {
            std::cerr << "Error creating directory " << dirname << ": " << e.what() << std::endl;
            return false;
        }
    }
    
    // Ensure directory exists, create if necessary
    static void ensure_directory(const std::string& dirname) {
        if (!directory_exists(dirname)) {
            if (!create_directory(dirname)) {
                throw std::runtime_error("Failed to create directory: " + dirname);
            }
        }
    }
    
    // Get file extension
    static std::string get_extension(const std::string& filename) {
        size_t dot_pos = filename.find_last_of('.');
        if (dot_pos == std::string::npos) {
            return "";
        }
        return filename.substr(dot_pos + 1);
    }
    
    // Get filename without extension
    static std::string get_basename(const std::string& filename) {
        size_t slash_pos = filename.find_last_of("/\\");
        std::string base = (slash_pos == std::string::npos) ? filename : filename.substr(slash_pos + 1);
        
        size_t dot_pos = base.find_last_of('.');
        if (dot_pos != std::string::npos) {
            base = base.substr(0, dot_pos);
        }
        
        return base;
    }
    
    // Get directory from file path
    static std::string get_directory(const std::string& filepath) {
        size_t slash_pos = filepath.find_last_of("/\\");
        if (slash_pos == std::string::npos) {
            return ".";
        }
        return filepath.substr(0, slash_pos);
    }
    
    // Read entire file into string
    static std::string read_file(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        
        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }
    
    // Read file lines into vector
    static std::vector<std::string> read_lines(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(file, line)) {
            lines.push_back(line);
        }
        
        return lines;
    }
    
    // Write string to file
    static void write_file(const std::string& filename, const std::string& content) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot create file: " + filename);
        }
        file << content;
    }
    
    // Write lines to file
    static void write_lines(const std::string& filename, const std::vector<std::string>& lines) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot create file: " + filename);
        }
        
        for (const auto& line : lines) {
            file << line << "\n";
        }
    }
    
    // Append to file
    static void append_file(const std::string& filename, const std::string& content) {
        std::ofstream file(filename, std::ios::app);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for appending: " + filename);
        }
        file << content;
    }
    
    // Copy file
    static bool copy_file(const std::string& source, const std::string& destination) {
        try {
            return std::filesystem::copy_file(source, destination);
        } catch (const std::exception& e) {
            std::cerr << "Error copying file: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Get file size in bytes
    static size_t get_file_size(const std::string& filename) {
        try {
            return std::filesystem::file_size(filename);
        } catch (const std::exception& e) {
            throw std::runtime_error("Cannot get file size: " + filename);
        }
    }
    
    // List files in directory
    static std::vector<std::string> list_files(const std::string& directory, const std::string& extension = "") {
        std::vector<std::string> files;
        
        try {
            for (const auto& entry : std::filesystem::directory_iterator(directory)) {
                if (entry.is_regular_file()) {
                    std::string filename = entry.path().filename().string();
                    if (extension.empty() || get_extension(filename) == extension) {
                        files.push_back(filename);
                    }
                }
            }
        } catch (const std::exception& e) {
            throw std::runtime_error("Cannot list directory: " + directory);
        }
        
        return files;
    }
    
    // Generate unique filename by adding numbers if file exists
    static std::string generate_unique_filename(const std::string& base_filename) {
        if (!file_exists(base_filename)) {
            return base_filename;
        }
        
        std::string basename = get_basename(base_filename);
        std::string extension = get_extension(base_filename);
        std::string directory = get_directory(base_filename);
        
        int counter = 1;
        std::string new_filename;
        
        do {
            new_filename = directory + "/" + basename + "_" + std::to_string(counter);
            if (!extension.empty()) {
                new_filename += "." + extension;
            }
            counter++;
        } while (file_exists(new_filename));
        
        return new_filename;
    }
    
    // Backup file by creating a copy with timestamp
    static std::string backup_file(const std::string& filename) {
        if (!file_exists(filename)) {
            throw std::runtime_error("File does not exist: " + filename);
        }
        
        // Generate backup filename with timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto tm = *std::localtime(&time_t);
        
        std::stringstream timestamp;
        timestamp << std::put_time(&tm, "%Y%m%d_%H%M%S");
        
        std::string basename = get_basename(filename);
        std::string extension = get_extension(filename);
        std::string directory = get_directory(filename);
        
        std::string backup_filename = directory + "/" + basename + "_backup_" + timestamp.str();
        if (!extension.empty()) {
            backup_filename += "." + extension;
        }
        
        if (!copy_file(filename, backup_filename)) {
            throw std::runtime_error("Failed to create backup: " + backup_filename);
        }
        
        return backup_filename;
    }
    
    // Clean filename by removing invalid characters
    static std::string clean_filename(const std::string& filename) {
        std::string clean = filename;
        const std::string invalid_chars = "<>:\"/\\|?*";
        
        for (char c : invalid_chars) {
            std::replace(clean.begin(), clean.end(), c, '_');
        }
        
        return clean;
    }
    
    // Parse CSV line (simple implementation)
    static std::vector<std::string> parse_csv_line(const std::string& line, char delimiter = ',') {
        std::vector<std::string> fields;
        std::stringstream ss(line);
        std::string field;
        
        while (std::getline(ss, field, delimiter)) {
            // Remove leading/trailing whitespace
            field.erase(0, field.find_first_not_of(" \t\r\n"));
            field.erase(field.find_last_not_of(" \t\r\n") + 1);
            fields.push_back(field);
        }
        
        return fields;
    }
    
    // Parse configuration file (key=value format)
    static std::map<std::string, std::string> parse_config_file(const std::string& filename) {
        std::map<std::string, std::string> config;
        auto lines = read_lines(filename);
        
        for (const auto& line : lines) {
            // Skip empty lines and comments
            if (line.empty() || line[0] == '#' || line[0] == ';') {
                continue;
            }
            
            size_t eq_pos = line.find('=');
            if (eq_pos != std::string::npos) {
                std::string key = line.substr(0, eq_pos);
                std::string value = line.substr(eq_pos + 1);
                
                // Trim whitespace
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t") + 1);
                
                config[key] = value;
            }
        }
        
        return config;
    }
    
    // Safe file operations with error handling
    class SafeFileWriter {
    private:
        std::string filename;
        std::string temp_filename;
        std::ofstream file;
        bool committed;
        
    public:
        SafeFileWriter(const std::string& fname) 
            : filename(fname), temp_filename(fname + ".tmp"), committed(false) {
            file.open(temp_filename);
            if (!file.is_open()) {
                throw std::runtime_error("Cannot create temporary file: " + temp_filename);
            }
        }
        
        ~SafeFileWriter() {
            if (!committed) {
                file.close();
                std::filesystem::remove(temp_filename);
            }
        }
        
        template<typename T>
        SafeFileWriter& operator<<(const T& data) {
            file << data;
            return *this;
        }
        
        void commit() {
            file.close();
            if (file_exists(filename)) {
                backup_file(filename);
            }
            std::filesystem::rename(temp_filename, filename);
            committed = true;
        }
        
        void write(const std::string& data) {
            file << data;
        }
        
        void writeln(const std::string& data) {
            file << data << "\n";
        }
    };
};

#endif // FILE_UTILS_HPP
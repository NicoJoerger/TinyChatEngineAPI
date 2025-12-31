#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <csignal>
#include <vector>

#include "Generate.h"
#include "model_config.h"
#include "interface.h"

// Global variables
int NUM_THREAD = 5;

// Global variables for graceful shutdown
volatile sig_atomic_t keep_running = 1;
std::ofstream log_file;

// Signal handler for SIGINT (Ctrl+C)
void signal_handler(int signal) {
    keep_running = 0;
    std::cout << "\n\nReceived interrupt signal. Shutting down gracefully...\n";
}

// Get current timestamp as string
std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

// Log to both console and file
void log_message(const std::string& message) {
    std::string timestamped = "[" + get_timestamp() + "] " + message;
    std::cout << timestamped << std::endl;
    if (log_file.is_open()) {
        log_file << timestamped << std::endl;
        log_file.flush();  // Ensure log is written immediately
    }
}

// Test prompts - rotating through different types
std::vector<std::string> test_prompts = {
    "What is 2+2?",
    "Explain machine learning in one sentence.",
    "What is the capital of France?",
    "How does photosynthesis work?",
    "What is the meaning of life?"
};

int main(int argc, char* argv[]) {
    // Setup signal handler
    signal(SIGINT, signal_handler);

    // Parse command line arguments
    std::string target_model = "LLaMA_3_8B_Instruct";
    std::string target_data_format = "INT4";

    if (argc >= 2) {
        target_model = argv[1];
    }
    if (argc >= 3) {
        target_data_format = argv[2];
    }
    if (argc >= 4) {
        NUM_THREAD = atoi(argv[3]);
    }

    // Validate model and format
    if (!is_valid_model(target_model)) {
        std::cerr << "Invalid model: " << target_model << std::endl;
        std::cerr << "Supported models: ";
        for (const auto& m : get_supported_models()) {
            std::cerr << m << " ";
        }
        std::cerr << std::endl;
        return 1;
    }

    if (!is_valid_format(target_data_format)) {
        std::cerr << "Invalid format: " << target_data_format << std::endl;
        std::cerr << "Supported formats: ";
        for (const auto& f : get_supported_formats()) {
            std::cerr << f << " ";
        }
        std::cerr << std::endl;
        return 1;
    }

    // Open log file with timestamp
    std::string log_filename = "stress_test_" + get_timestamp() + ".log";
    // Replace spaces and colons in filename
    for (char& c : log_filename) {
        if (c == ' ' || c == ':') c = '_';
    }
    log_file.open(log_filename);

    if (!log_file.is_open()) {
        std::cerr << "Failed to open log file: " << log_filename << std::endl;
        return 1;
    }

    log_message("=== TinyChatEngine Stress Test ===");
    log_message("Model: " + target_model);
    log_message("Format: " + target_data_format);
    log_message("Threads: " + std::to_string(NUM_THREAD));
    log_message("Log file: " + log_filename);

    // Initialize model
    log_message("Loading model...");

    int format_id = get_data_format_id(target_data_format);
    int model_id = get_model_id(target_model);
    std::string m_path = get_model_path(target_model);

    #ifdef MODEL_PREFIX
    m_path = MODEL_PREFIX + m_path;
    #endif

    // Get default config and set temp to 0
    struct opt_params generation_config = get_llama3_default_config();
    generation_config.temp = 0.0f;  // Deterministic output

    log_message("Generation config: temp=0.0, top_p=" + std::to_string(generation_config.top_p) +
                ", top_k=" + std::to_string(generation_config.top_k));

    // Load model based on format
    if (format_id != INT4) {
        log_message("ERROR: Only INT4 format is supported in this stress test");
        log_file.close();
        return 1;
    }

    m_path = "INT4/" + m_path;
    Int4LlamaForCausalLM model = Int4LlamaForCausalLM(m_path, get_opt_model_config(model_id));

    log_message("Model loaded successfully!");

    // Test loop metrics
    unsigned long long total_cycles = 0;
    unsigned long long total_errors = 0;
    unsigned long long consecutive_errors = 0;
    double total_generation_time_ms = 0.0;
    auto start_time = std::chrono::steady_clock::now();

    bool first_prompt = true;
    int prompt_index = 0;

    log_message("Starting infinite test loop. Press Ctrl+C to stop gracefully.");
    log_message("===================================================");

    // Main test loop
    while (keep_running) {
        total_cycles++;

        try {
            // Get test prompt (rotate through prompts)
            std::string test_input = test_prompts[prompt_index % test_prompts.size()];
            prompt_index++;

            log_message("[CYCLE " + std::to_string(total_cycles) + "] Starting");
            log_message("[CYCLE " + std::to_string(total_cycles) + "] Prompt: \"" + test_input + "\"");

            // Build prompt
            std::string input = build_llama3_prompt(test_input, first_prompt);

            // Time the generation
            auto gen_start = std::chrono::steady_clock::now();

            // Generate response (suppress output with false for print flag)
            LLaMA3Generate(m_path, &model, LLaMA_INT4, input, generation_config,
                          "models/llama3_vocab.bin", false, false);

            auto gen_end = std::chrono::steady_clock::now();
            auto gen_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gen_end - gen_start).count();
            total_generation_time_ms += gen_duration;

            log_message("[CYCLE " + std::to_string(total_cycles) + "] Generation completed in " +
                       std::to_string(gen_duration) + "ms");

            // Reset conversation
            log_message("[CYCLE " + std::to_string(total_cycles) + "] Resetting conversation...");

            // Call reset functions (from chat.cc /new command)
            LLaMA3ResetConversationState();

            struct model_config config = get_opt_model_config(model_id);
            Int4llamaDecoderLayer::initialize_decoder_memory(config);
            Int4llamaAttention::initialized_memory(config);
            Int4llamaAttention::reset_cache(config);

            first_prompt = true;  // Reset local state

            log_message("[CYCLE " + std::to_string(total_cycles) + "] Reset complete");

            // Calculate stats
            auto current_time = std::chrono::steady_clock::now();
            auto uptime = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
            double avg_time = total_generation_time_ms / total_cycles;

            log_message("[CYCLE " + std::to_string(total_cycles) + "] === COMPLETE === " +
                       "Avg: " + std::to_string((int)avg_time) + "ms, " +
                       "Cycles: " + std::to_string(total_cycles) + ", " +
                       "Errors: " + std::to_string(total_errors) + ", " +
                       "Uptime: " + std::to_string(uptime) + "s");

            // Reset consecutive errors on success
            consecutive_errors = 0;

            // Periodic detailed summary every 10 cycles
            if (total_cycles % 10 == 0) {
                log_message("========================================");
                log_message("=== SUMMARY AFTER " + std::to_string(total_cycles) + " CYCLES ===");
                log_message("Total cycles: " + std::to_string(total_cycles));
                log_message("Total errors: " + std::to_string(total_errors));
                log_message("Error rate: " + std::to_string((total_errors * 100.0) / total_cycles) + "%");
                log_message("Average generation time: " + std::to_string((int)avg_time) + "ms");
                log_message("Total uptime: " + std::to_string(uptime) + "s (" +
                           std::to_string(uptime / 60) + " minutes)");
                log_message("========================================");
            }

        } catch (const std::exception& e) {
            total_errors++;
            consecutive_errors++;
            log_message("[CYCLE " + std::to_string(total_cycles) + "] ERROR: " + std::string(e.what()));

            // Exit if too many consecutive errors
            if (consecutive_errors >= 5) {
                log_message("FATAL: " + std::to_string(consecutive_errors) +
                           " consecutive errors. Exiting.");
                break;
            }

            // Try to reset after error
            try {
                LLaMA3ResetConversationState();
                struct model_config config = get_opt_model_config(model_id);
                Int4llamaDecoderLayer::initialize_decoder_memory(config);
                Int4llamaAttention::initialized_memory(config);
                Int4llamaAttention::reset_cache(config);
                first_prompt = true;
                log_message("[CYCLE " + std::to_string(total_cycles) + "] Recovery reset attempted");
            } catch (...) {
                log_message("[CYCLE " + std::to_string(total_cycles) + "] Recovery reset FAILED");
            }

        } catch (...) {
            total_errors++;
            consecutive_errors++;
            log_message("[CYCLE " + std::to_string(total_cycles) + "] UNKNOWN ERROR occurred");

            if (consecutive_errors >= 5) {
                log_message("FATAL: " + std::to_string(consecutive_errors) +
                           " consecutive errors. Exiting.");
                break;
            }
        }
    }

    // Final summary
    auto end_time = std::chrono::steady_clock::now();
    auto total_uptime = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

    log_message("");
    log_message("========================================");
    log_message("=== FINAL SUMMARY ===");
    log_message("Total cycles completed: " + std::to_string(total_cycles));
    log_message("Total errors: " + std::to_string(total_errors));
    log_message("Error rate: " + std::to_string((total_errors * 100.0) / total_cycles) + "%");
    log_message("Average generation time: " + std::to_string((int)(total_generation_time_ms / total_cycles)) + "ms");
    log_message("Total uptime: " + std::to_string(total_uptime) + "s (" +
               std::to_string(total_uptime / 60) + " minutes, " +
               std::to_string(total_uptime / 3600) + " hours)");
    log_message("========================================");
    log_message("Test completed. Log saved to: " + log_filename);

    log_file.close();
    return 0;
}

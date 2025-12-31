#include <iostream>
#include <string>
#include <sstream>
#include <mutex>
#include <atomic>
#include <map>

#include "../cpp-httplib/httplib.h"
#include "../json/single_include/nlohmann/json.hpp"

#include "Generate.h"
#include "model_config.h"
#include "interface.h"

// Global variables
int NUM_THREAD = 5;

// Global API server state
struct APIServerState {
    // Model information
    void* model_ptr = nullptr;
    int model_type = -1;
    int model_id = -1;
    std::string model_name;
    std::string model_path;
    std::string vocab_path;
    std::string data_format;

    // Generation configuration
    struct opt_params generation_config;

    // Conversation state
    bool first_prompt = true;

    // Concurrency control
    std::mutex generation_mutex;
    std::atomic<bool> is_generating{false};
};

static APIServerState g_state;

// Helper functions for concurrency control
bool try_acquire_generation_lock() {
    bool expected = false;
    return g_state.is_generating.compare_exchange_strong(expected, true);
}

void release_generation_lock() {
    g_state.is_generating.store(false);
}

// GET /health - Health check endpoint
void setup_health_endpoint(httplib::Server& server) {
    server.Get("/health", [](const httplib::Request& req, httplib::Response& res) {
        nlohmann::json response;
        response["status"] = "healthy";
        response["model"] = g_state.model_name;
        response["format"] = g_state.data_format;
        response["is_busy"] = g_state.is_generating.load();

        res.set_content(response.dump(), "application/json");
    });
}

// POST /chat - Stream text generation via SSE
void setup_chat_endpoint(httplib::Server& server) {
    server.Post("/chat", [](const httplib::Request& req, httplib::Response& res) {
        // Check if generation is already in progress
        if (!try_acquire_generation_lock()) {
            res.status = 503;
            nlohmann::json error;
            error["error"] = "Server busy, another request is processing";
            res.set_content(error.dump(), "application/json");
            return;
        }

        // Ensure lock is released on exit
        struct LockGuard {
            ~LockGuard() { release_generation_lock(); }
        } lock_guard;

        // Parse request JSON
        nlohmann::json req_json;
        try {
            req_json = nlohmann::json::parse(req.body);
        } catch (...) {
            res.status = 400;
            nlohmann::json error;
            error["error"] = "Invalid JSON";
            res.set_content(error.dump(), "application/json");
            return;
        }

        if (!req_json.contains("prompt")) {
            res.status = 400;
            nlohmann::json error;
            error["error"] = "Missing 'prompt' field";
            res.set_content(error.dump(), "application/json");
            return;
        }

        std::string user_prompt = req_json["prompt"];

        // Build prompt with conversation template
        std::string input_text;
        if (isLLaMA3(g_state.model_name)) {
            input_text = build_llama3_prompt(user_prompt, g_state.first_prompt);
        } else if (isLLaMA(g_state.model_name) || isCodeLLaMA(g_state.model_name)) {
            input_text = build_llama_prompt(user_prompt, g_state.first_prompt, isCodeLLaMA(g_state.model_name));
        } else if (isMistral(g_state.model_name)) {
            input_text = build_mistral_prompt(user_prompt, g_state.first_prompt, true);
        } else {
            // Fallback: use raw prompt
            input_text = user_prompt;
        }

        if (g_state.first_prompt) {
            g_state.first_prompt = false;
        }

        // Setup SSE streaming
        res.set_header("Content-Type", "text/event-stream");
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        res.set_header("Access-Control-Allow-Origin", "*");

        std::stringstream sse_buffer;

        // Define SSE callback
        auto sse_callback = [](const char* token_str, int token_id, int position, void* user_data) {
            auto* buffer = static_cast<std::stringstream*>(user_data);

            // Create JSON event
            nlohmann::json event;
            event["token"] = token_str;
            event["position"] = position;

            // Write SSE format: "data: {json}\n\n"
            (*buffer) << "data: " << event.dump() << "\n\n";
        };

        // Generate with callback
        try {
            std::string full_output;

            if (isLLaMA3(g_state.model_name)) {
                full_output = LLaMA3Generate(
                    g_state.model_path,
                    g_state.model_ptr,
                    g_state.model_type,
                    input_text,
                    g_state.generation_config,
                    g_state.vocab_path,
                    false,  // interactive = false
                    false,  // voicechat = false
                    sse_callback,
                    &sse_buffer
                );
            } else if (isLLaMA(g_state.model_name) || isCodeLLaMA(g_state.model_name)) {
                full_output = LLaMAGenerate(
                    g_state.model_path,
                    g_state.model_ptr,
                    g_state.model_type,
                    input_text,
                    g_state.generation_config,
                    g_state.vocab_path,
                    false,
                    false,
                    sse_callback,
                    &sse_buffer
                );
            } else if (isMistral(g_state.model_name)) {
                full_output = MistralGenerate(
                    g_state.model_path,
                    g_state.model_ptr,
                    g_state.model_type,
                    input_text,
                    g_state.generation_config,
                    g_state.vocab_path,
                    false,
                    false,
                    sse_callback,
                    &sse_buffer
                );
            } else {
                throw std::runtime_error("Unsupported model type for generation");
            }

            // Send completion signal
            sse_buffer << "data: [DONE]\n\n";

        } catch (const std::exception& e) {
            nlohmann::json error_event;
            error_event["error"] = e.what();
            sse_buffer << "data: " << error_event.dump() << "\n\n";
            sse_buffer << "data: [ERROR]\n\n";
        }

        // Write response
        res.set_content(sse_buffer.str(), "text/event-stream");
    });
}

// POST /settings - Update generation parameters
void setup_settings_endpoint(httplib::Server& server) {
    server.Post("/settings", [](const httplib::Request& req, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(g_state.generation_mutex);

        nlohmann::json req_json;
        try {
            req_json = nlohmann::json::parse(req.body);
        } catch (...) {
            res.status = 400;
            nlohmann::json error;
            error["error"] = "Invalid JSON";
            res.set_content(error.dump(), "application/json");
            return;
        }

        std::vector<std::string> updated_params;

        // Map of JSON field names to parameter names
        std::map<std::string, std::string> param_map = {
            {"temperature", "temp"},
            {"top_p", "top_p"},
            {"top_k", "top_k"},
            {"max_tokens", "n_predict"},
            {"repeat_penalty", "repeat_penalty"},
            {"presence_penalty", "presence_penalty"},
            {"frequency_penalty", "frequency_penalty"}
        };

        for (auto it = param_map.begin(); it != param_map.end(); ++it) {
            const std::string& json_key = it->first;
            const std::string& param_name = it->second;

            if (req_json.contains(json_key)) {
                std::string value_str;

                // Handle both float and int values
                if (req_json[json_key].is_number_float()) {
                    value_str = std::to_string(req_json[json_key].get<float>());
                } else if (req_json[json_key].is_number_integer()) {
                    value_str = std::to_string(req_json[json_key].get<int>());
                } else {
                    continue;
                }

                if (set_generation_param(g_state.generation_config, param_name, value_str)) {
                    updated_params.push_back(json_key);
                }
            }
        }

        // Build response
        nlohmann::json response;
        response["status"] = "success";
        response["updated"] = updated_params;
        response["current_config"] = {
            {"temperature", g_state.generation_config.temp},
            {"top_p", g_state.generation_config.top_p},
            {"top_k", g_state.generation_config.top_k},
            {"max_tokens", g_state.generation_config.n_predict},
            {"repeat_penalty", g_state.generation_config.repeat_penalty},
            {"presence_penalty", g_state.generation_config.presence_penalty},
            {"frequency_penalty", g_state.generation_config.frequency_penalty}
        };

        res.set_content(response.dump(), "application/json");
    });
}

// POST /reset - Reset conversation state
void setup_reset_endpoint(httplib::Server& server) {
    server.Post("/reset", [](const httplib::Request& req, httplib::Response& res) {
        // Check if generation is in progress
        if (g_state.is_generating.load()) {
            res.status = 409;
            nlohmann::json error;
            error["error"] = "Cannot reset during active generation";
            res.set_content(error.dump(), "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(g_state.generation_mutex);

        // Call appropriate reset function based on model type
        try {
            struct model_config config = get_opt_model_config(g_state.model_id);

            if (isLLaMA3(g_state.model_name)) {
                LLaMA3ResetConversationState();
                Int4llamaDecoderLayer::initialize_decoder_memory(config);
                Int4llamaAttention::initialized_memory(config);
                Int4llamaAttention::reset_cache(config);
            } else if (isMistral(g_state.model_name)) {
                MistralResetConversationState();
                Int4llamaDecoderLayer::initialize_decoder_memory(config);
                Int4llamaAttention::initialized_memory(config);
                Int4llamaAttention::reset_cache(config);
            } else if (isStarCoder(g_state.model_name)) {
                GPTBigCodeResetConversationState();
            } else if (isLLaMA(g_state.model_name) || isCodeLLaMA(g_state.model_name)) {
                // LLaMA/CodeLLaMA don't have a reset function yet
                // Just reset first_prompt flag
            }

            g_state.first_prompt = true;

            nlohmann::json response;
            response["status"] = "success";
            response["message"] = "Conversation state reset";

            res.set_content(response.dump(), "application/json");

        } catch (const std::exception& e) {
            res.status = 500;
            nlohmann::json error;
            error["error"] = std::string("Reset failed: ") + e.what();
            res.set_content(error.dump(), "application/json");
        }
    });
}

// Setup all routes
void setup_routes(httplib::Server& server) {
    setup_health_endpoint(server);
    setup_chat_endpoint(server);
    setup_settings_endpoint(server);
    setup_reset_endpoint(server);
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    std::string target_model = "LLaMA_3_8B_Instruct";
    std::string target_data_format = "INT4";
    int port = 8080;

    if (argc >= 2) {
        target_model = argv[1];
    }
    if (argc >= 3) {
        target_data_format = argv[2];
    }
    if (argc >= 4) {
        port = atoi(argv[3]);
    }
    if (argc >= 5) {
        NUM_THREAD = atoi(argv[4]);
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

    // Store model name and format
    g_state.model_name = target_model;
    g_state.data_format = target_data_format;

    // Print startup info
    std::cout << "TinyChatEngine API Server" << std::endl;
    std::cout << "Model: " << target_model << std::endl;
    std::cout << "Format: " << target_data_format << std::endl;
    std::cout << "Port: " << port << std::endl;
    std::cout << "Threads: " << NUM_THREAD << std::endl;
    std::cout << std::endl;

    // Load model
    std::cout << "Loading model... " << std::flush;

    int format_id = get_data_format_id(target_data_format);
    g_state.model_id = get_model_id(target_model);
    g_state.model_path = get_model_path(target_model);

    #ifdef MODEL_PREFIX
    g_state.model_path = MODEL_PREFIX + g_state.model_path;
    #endif

    // Initialize model based on type
    if (isLLaMA3(target_model)) {
        g_state.generation_config = get_llama3_default_config();
        g_state.vocab_path = "models/llama3_vocab.bin";

        if (format_id == INT4) {
            g_state.model_path = "INT4/" + g_state.model_path;
            g_state.model_ptr = new Int4LlamaForCausalLM(
                g_state.model_path,
                get_opt_model_config(g_state.model_id)
            );
            g_state.model_type = LLaMA_INT4;
        } else if (format_id == FP32) {
            g_state.model_ptr = new Fp32LlamaForCausalLM(
                g_state.model_path,
                get_opt_model_config(g_state.model_id)
            );
            g_state.model_type = LLaMA_FP32;
        } else {
            std::cerr << "Unsupported format for LLaMA3: " << target_data_format << std::endl;
            return 1;
        }
    } else if (isLLaMA(target_model) || isCodeLLaMA(target_model)) {
        g_state.generation_config = get_llama_default_config(target_model);
        g_state.vocab_path = "models/llama_vocab.bin";

        if (format_id == INT4) {
            g_state.model_path = "INT4/" + g_state.model_path;
            g_state.model_ptr = new Int4LlamaForCausalLM(
                g_state.model_path,
                get_opt_model_config(g_state.model_id)
            );
            g_state.model_type = LLaMA_INT4;
        } else if (format_id == FP32) {
            g_state.model_ptr = new Fp32LlamaForCausalLM(
                g_state.model_path,
                get_opt_model_config(g_state.model_id)
            );
            g_state.model_type = LLaMA_FP32;
        } else {
            std::cerr << "Unsupported format for LLaMA: " << target_data_format << std::endl;
            return 1;
        }
    } else if (isMistral(target_model)) {
        g_state.generation_config = get_mistral_default_config();
        g_state.vocab_path = "models/mistral_vocab.bin";

        if (format_id == INT4) {
            g_state.model_path = "INT4/" + g_state.model_path;
            g_state.model_ptr = new Int4LlamaForCausalLM(
                g_state.model_path,
                get_opt_model_config(g_state.model_id)
            );
            g_state.model_type = LLaMA_INT4;
        } else if (format_id == FP32) {
            g_state.model_ptr = new Fp32LlamaForCausalLM(
                g_state.model_path,
                get_opt_model_config(g_state.model_id)
            );
            g_state.model_type = LLaMA_FP32;
        } else {
            std::cerr << "Unsupported format for Mistral: " << target_data_format << std::endl;
            return 1;
        }
    } else if (isStarCoder(target_model)) {
        g_state.generation_config = get_starcoder_default_config();
        g_state.vocab_path = "models/starcoder_vocab.bin";

        if (format_id == INT4) {
            g_state.model_path = "INT4/" + g_state.model_path;
            g_state.model_ptr = new Int4GPTBigCodeForCausalLM(
                g_state.model_path,
                get_opt_model_config(g_state.model_id)
            );
            g_state.model_type = StarCoder_INT4;
        } else if (format_id == FP32) {
            g_state.model_ptr = new Fp32GPTBigCodeForCausalLM(
                g_state.model_path,
                get_opt_model_config(g_state.model_id)
            );
            g_state.model_type = StarCoder_FP32;
        } else {
            std::cerr << "Unsupported format for StarCoder: " << target_data_format << std::endl;
            return 1;
        }
    } else {
        std::cerr << "Unsupported model type: " << target_model << std::endl;
        return 1;
    }

    std::cout << "Finished!" << std::endl << std::endl;

    // Setup HTTP server
    httplib::Server server;
    setup_routes(server);

    std::cout << "API Server Endpoints:" << std::endl;
    std::cout << "  GET  http://localhost:" << port << "/health" << std::endl;
    std::cout << "  POST http://localhost:" << port << "/chat" << std::endl;
    std::cout << "  POST http://localhost:" << port << "/settings" << std::endl;
    std::cout << "  POST http://localhost:" << port << "/reset" << std::endl;
    std::cout << std::endl;
    std::cout << "Starting API server on 0.0.0.0:" << port << "..." << std::endl;

    server.listen("0.0.0.0", port);

    return 0;
}

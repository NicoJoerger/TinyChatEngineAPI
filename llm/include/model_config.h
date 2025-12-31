#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#include <string>
#include <vector>
#include "Generate.h"

// ============================================================================
// Model Type Detection Functions
// ============================================================================

bool isLLaMA3(std::string s);
bool isLLaMA(std::string s);
bool isCodeLLaMA(std::string s);
bool isStarCoder(std::string s);
bool isLLaVA(std::string s);
bool isVILA1_5(std::string s);
bool isVILA(std::string s);
bool isMistral(std::string s);

// ============================================================================
// Configuration Map Access Functions
// ============================================================================

// Get model ID from model name (returns -1 if not found)
int get_model_id(const std::string& model_name);

// Get model path from model name (returns empty string if not found)
std::string get_model_path(const std::string& model_name);

// Get data format ID from format string (returns -1 if not found)
int get_data_format_id(const std::string& format_name);

// Check if model name is valid
bool is_valid_model(const std::string& model_name);

// Check if format name is valid
bool is_valid_format(const std::string& format_name);

// Get list of all supported model names
std::vector<std::string> get_supported_models();

// Get list of all supported format names
std::vector<std::string> get_supported_formats();

// ============================================================================
// Utility Functions
// ============================================================================

// Convert string to boolean with validation
bool convertToBool(const char* str);

// ============================================================================
// Generation Parameter Utilities
// ============================================================================

// Display current generation configuration
void show_generation_config(const struct opt_params& config);

// Set a generation parameter from string value
// Returns true on success, false on error
bool set_generation_param(struct opt_params& config, const std::string& param, const std::string& value_str);

// ============================================================================
// Default Configuration Factory Functions
// ============================================================================

// Get default configuration for LLaMA 3 models
opt_params get_llama3_default_config();

// Get default configuration for LLaMA/LLaMA2 models
opt_params get_llama_default_config(const std::string& model_name);

// Get default configuration for Mistral models
opt_params get_mistral_default_config();

// Get default configuration for StarCoder models
opt_params get_starcoder_default_config();

// Get default configuration for LLaVA models
opt_params get_llava_default_config();

// Get default configuration for VILA models
opt_params get_vila_default_config();

// ============================================================================
// Prompt Template Builder Functions
// ============================================================================

// Build prompt for LLaMA 3 models
std::string build_llama3_prompt(const std::string& user_input, bool is_first_prompt);

// Build prompt for LLaMA/LLaMA2 models
std::string build_llama_prompt(const std::string& user_input, bool is_first_prompt, bool is_codellama);

// Build prompt for Mistral models
std::string build_mistral_prompt(const std::string& user_input, bool is_first_prompt, bool use_instruct);

// Build prompt for LLaVA models
std::string build_llava_prompt(const std::string& user_input, int prompt_iter);

// Build prompt for VILA models (same as LLaVA)
std::string build_vila_prompt(const std::string& user_input, int prompt_iter);

#endif  // MODEL_CONFIG_H

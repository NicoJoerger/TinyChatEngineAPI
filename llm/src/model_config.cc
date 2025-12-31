#include "model_config.h"
#include "interface.h"
#include <iostream>
#include <map>
#include <cstring>
#include <sstream>

// ============================================================================
// Configuration Maps (from chat.cc lines 10-44)
// ============================================================================

std::map<std::string, int> model_config = {
    {"OPT_125m", OPT_125M},         {"OPT_1.3B", OPT_1_3B},               {"OPT_6.7B", OPT_6_7B}, {"LLaMA_7B", LLaMA_7B},
    {"7b", LLaMA_7B},               {"LLaMA2_7B_chat", LLaMA_7B},         {"13b", LLaMA_13B},     {"LLaMA2_13B_chat", LLaMA_13B},
    {"CodeLLaMA_7B_Instruct", CodeLLaMA_7B},                              {"CodeLLaMA_13B_Instruct", CodeLLaMA_13B},
    {"StarCoder", StarCoder_15_5B}, {"StarCoder_15.5B", StarCoder_15_5B}, {"LLaVA_7B", LLaVA_7B}, {"LLaVA_13B", LLaVA_13B},
    {"VILA_2.7B", VILA_2_7B},       {"VILA_7B", VILA_7B},                 {"VILA_13B", VILA_13B}, {"Clip_ViT_Large", Clip_ViT_Large},
    {"Mistral_7B", Mistral_7B},     {"LLaMA_3_8B_Instruct", LLaMA_3_8B},  {"VILA1.5_8B", VILA1_5_8B},
};

std::map<std::string, std::string> model_path = {{"OPT_125m", "models/OPT_125m"},
                                                 {"OPT_1.3B", "models/OPT_1.3B"},
                                                 {"OPT_6.7B", "models/OPT_6.7B"},
                                                 {"LLaMA_7B", "models/LLaMA_7B"},
                                                 {"LLaMA2_7B_chat", "models/LLaMA_7B_2_chat"},
                                                 {"LLaMA2_13B_chat", "models/LLaMA_13B_2_chat"},
                                                 {"7b", "models/LLaMA_7B_2_chat"},
                                                 {"13b", "models/LLaMA_13B_2_chat"},
                                                 {"CodeLLaMA_7B_Instruct", "models/CodeLLaMA_7B_Instruct"},
                                                 {"CodeLLaMA_13B_Instruct", "models/CodeLLaMA_13B_Instruct"},
                                                 {"StarCoder", "models/StarCoder"},
                                                 {"StarCoder_15.5B", "models/StarCoder"},
                                                 {"LLaVA_7B", "models/LLaVA_7B"},
                                                 {"LLaVA_13B", "models/LLaVA_13B"},
                                                 {"VILA_2.7B", "models/VILA_2.7B"},
                                                 {"VILA_7B", "models/VILA_7B"},
                                                 {"VILA_13B", "models/VILA_13B"},
                                                 {"Clip_ViT_Large", "models/CLIP_ViT_Large"},
                                                 {"Mistral_7B", "models/Mistral_7B"},
                                                 {"LLaMA_3_8B_Instruct", "models/LLaMA_3_8B_Instruct"},
                                                 {"VILA1.5_8B", "models/VILA1.5_8B"},
                                                 };

std::map<std::string, int> data_format_list = {
    {"FP32", FP32}, {"INT8", QINT8}, {"INT4", INT4}, {"int4", INT4}, {"fp32", FP32},
};

// ============================================================================
// Model Type Detection Functions (from chat.cc lines 46-109)
// ============================================================================

bool isLLaMA3(std::string s) {
    std::string LLaMA_prefix = "LLaMA_3";
    if (s.substr(0, LLaMA_prefix.size()) == LLaMA_prefix)
        return true;
    else
        return false;
}

bool isLLaMA(std::string s) {
    std::string LLaMA_prefix = "LLaMA";
    std::string CodeLLaMA_prefix = "CodeLLaMA";
    if (s.substr(0, LLaMA_prefix.size()) == LLaMA_prefix || s.substr(0, CodeLLaMA_prefix.size()) == CodeLLaMA_prefix || s == "7b" || s == "13b")
        return true;
    else
        return false;
}

bool isCodeLLaMA(std::string s) {
    std::string CodeLLaMA_prefix = "CodeLLaMA";
    if (s.substr(0, CodeLLaMA_prefix.size()) == CodeLLaMA_prefix)
        return true;
    else
        return false;
}

bool isStarCoder(std::string s) {
    std::string StarCoder_prefix = "StarCoder";
    if (s.substr(0, StarCoder_prefix.size()) == StarCoder_prefix)
        return true;
    else
        return false;
}

bool isLLaVA(std::string s) {
    std::string LLaVA_prefix = "LLaVA";
    if (s.substr(0, LLaVA_prefix.size()) == LLaVA_prefix)
        return true;
    else
        return false;
}

bool isVILA1_5(std::string s) {
    std::string VILA_prefix = "VILA1.5";
    if (s.substr(0, VILA_prefix.size()) == VILA_prefix)
        return true;
    else
        return false;
}

bool isVILA(std::string s) {
    std::string VILA_prefix = "VILA";
    if (s.substr(0, VILA_prefix.size()) == VILA_prefix)
        return true;
    else
        return false;
}

bool isMistral(std::string s) {
    std::string Mistral_prefix = "Mistral";
    if (s.substr(0, Mistral_prefix.size()) == Mistral_prefix)
        return true;
    else
        return false;
}

// ============================================================================
// Utility Functions (from chat.cc lines 111-122)
// ============================================================================

bool convertToBool(const char* str) {
    if (strcmp(str, "true") == 0 || strcmp(str, "1") == 0) {
        return true;
    }
    else if (strcmp(str, "false") == 0 || strcmp(str, "0") == 0) {
        return false;
    }
    else {
        std::cerr << "Error: Invalid boolean value: " << str << std::endl;
        exit(EXIT_FAILURE);
    }
}

// ============================================================================
// Generation Parameter Utilities (from chat.cc lines 125-205)
// ============================================================================

void show_generation_config(const struct opt_params& config) {
    set_print_yellow();
    std::cout << "\n=== Generation Parameters ===\n";
    std::cout << "n_ctx:             " << config.n_ctx << " (penalty window)\n";
    std::cout << "temp:              " << config.temp << "\n";
    std::cout << "top_p:             " << config.top_p << "\n";
    std::cout << "top_k:             " << config.top_k << "\n";
    std::cout << "repeat_penalty:    " << config.repeat_penalty << "\n";
    std::cout << "frequency_penalty: " << config.frequency_penalty << "\n";
    std::cout << "presence_penalty:  " << config.presence_penalty << "\n";
    std::cout << "============================\n\n";
    set_print_reset();
}

bool set_generation_param(struct opt_params& config, const std::string& param, const std::string& value_str) {
    try {
        if (param == "n_ctx") {
            int val = std::stoi(value_str);
            if (val < 1) {
                std::cerr << "Error: n_ctx must be >= 1\n";
                return false;
            }
            config.n_ctx = val;
        } else if (param == "temp") {
            float val = std::stof(value_str);
            if (val < 0.0f || val > 2.0f) {
                std::cerr << "Error: temp must be between 0.0 and 2.0\n";
                return false;
            }
            config.temp = val;
        } else if (param == "top_p") {
            float val = std::stof(value_str);
            if (val < 0.0f || val > 1.0f) {
                std::cerr << "Error: top_p must be between 0.0 and 1.0\n";
                return false;
            }
            config.top_p = val;
        } else if (param == "top_k") {
            int val = std::stoi(value_str);
            if (val < 0) {
                std::cerr << "Error: top_k must be >= 0 (0 = disabled)\n";
                return false;
            }
            config.top_k = val;
        } else if (param == "repeat_penalty") {
            float val = std::stof(value_str);
            if (val < 0.0f || val > 2.0f) {
                std::cerr << "Error: repeat_penalty must be between 0.0 and 2.0\n";
                return false;
            }
            config.repeat_penalty = val;
        } else if (param == "frequency_penalty") {
            float val = std::stof(value_str);
            if (val < -2.0f || val > 2.0f) {
                std::cerr << "Error: frequency_penalty must be between -2.0 and 2.0\n";
                return false;
            }
            config.frequency_penalty = val;
        } else if (param == "presence_penalty") {
            float val = std::stof(value_str);
            if (val < -2.0f || val > 2.0f) {
                std::cerr << "Error: presence_penalty must be between -2.0 and 2.0\n";
                return false;
            }
            config.presence_penalty = val;
        } else {
            std::cerr << "Error: Unknown parameter '" << param << "'\n";
            std::cerr << "Available: n_ctx, temp, top_p, top_k, repeat_penalty, frequency_penalty, presence_penalty\n";
            return false;
        }

        set_print_yellow();
        std::cout << "Set " << param << " = " << value_str << "\n\n";
        set_print_reset();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error: Invalid value '" << value_str << "'\n";
        return false;
    }
}

// ============================================================================
// Configuration Map Access Functions
// ============================================================================

int get_model_id(const std::string& model_name) {
    auto it = model_config.find(model_name);
    if (it != model_config.end()) {
        return it->second;
    }
    return -1;
}

std::string get_model_path(const std::string& model_name) {
    auto it = model_path.find(model_name);
    if (it != model_path.end()) {
        return it->second;
    }
    return "";
}

int get_data_format_id(const std::string& format_name) {
    auto it = data_format_list.find(format_name);
    if (it != data_format_list.end()) {
        return it->second;
    }
    return -1;
}

bool is_valid_model(const std::string& model_name) {
    return model_config.count(model_name) > 0;
}

bool is_valid_format(const std::string& format_name) {
    return data_format_list.count(format_name) > 0;
}

std::vector<std::string> get_supported_models() {
    std::vector<std::string> models;
    for (const auto& pair : model_config) {
        models.push_back(pair.first);
    }
    return models;
}

std::vector<std::string> get_supported_formats() {
    std::vector<std::string> formats;
    for (const auto& pair : data_format_list) {
        formats.push_back(pair.first);
    }
    return formats;
}

// ============================================================================
// Default Configuration Factory Functions
// ============================================================================

opt_params get_llama3_default_config() {
    struct opt_params config;
    config.n_predict = 2048;
    config.n_ctx = 512;
    config.repeat_last_n = -1;
    config.repeat_penalty = 1.1f;
    config.temp = 0.7f;
    config.n_vocab = 128256;
    config.top_p = 0.9f;
    config.top_k = 40;
    config.frequency_penalty = 0.0f;
    config.presence_penalty = 0.0f;
    return config;
}

opt_params get_llama_default_config(const std::string& model_name) {
    struct opt_params config;
    config.n_predict = 512;
    config.n_ctx = 512;
    config.repeat_last_n = -1;
    config.repeat_penalty = 1.1f;
    config.temp = 0.2f;
    config.top_k = 40;
    config.top_p = 0.95f;
    config.frequency_penalty = 0.0f;
    config.presence_penalty = 0.0f;

    if (isCodeLLaMA(model_name)) {
        config.n_vocab = 32016;
    } else {
        config.n_vocab = 32000;
    }

    return config;
}

opt_params get_mistral_default_config() {
    struct opt_params config;
    config.n_predict = 512;
    config.repeat_penalty = 1.0f;
    config.temp = 0.3f;
    config.n_vocab = 32000;
    return config;
}

opt_params get_starcoder_default_config() {
    struct opt_params config;
    config.n_predict = 128;
    config.top_k = 0;
    config.temp = 0.2f;
    config.n_vocab = 49152;
    return config;
}

opt_params get_llava_default_config() {
    struct opt_params config;
    config.n_predict = 512;
    config.repeat_penalty = 1.1f;
    config.temp = 0.2f;
    config.n_vocab = 32000;
    return config;
}

opt_params get_vila_default_config() {
    struct opt_params config;
    config.n_predict = 512;
    config.repeat_penalty = 1.1f;
    config.temp = 0.2f;
    config.n_vocab = 32000;
    config.top_p = 1.0f;
    return config;
}

// ============================================================================
// Prompt Template Builder Functions
// ============================================================================

std::string build_llama3_prompt(const std::string& user_input, bool is_first_prompt) {
    if (is_first_prompt) {
        return "A chat between a curious human (\"Human\") and an artificial intelligence assistant (\"Assistant\"). The assistant gives detailed, helpful, and polite answers to the human's questions.\n\nHuman: " + user_input + "\nAssistant: ";
    } else {
        return "Human: " + user_input + "\nAssistant: \n";
    }
}

std::string build_llama_prompt(const std::string& user_input, bool is_first_prompt, bool is_codellama) {
    if (is_codellama) {
        if (is_first_prompt) {
            return "<s>[INST] " + user_input + " [/INST] ";
        } else {
            return " </s> <s>[INST] " + user_input + " [/INST] ";
        }
    } else {
        if (is_first_prompt) {
            return "A chat between a curious human (\"Human\") and an artificial intelligence assistant (\"Assistant\"). The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n### Human: " + user_input + "\n### Assistant: ";
        } else {
            return "### Human: " + user_input + "\n### Assistant: \n";
        }
    }
}

std::string build_mistral_prompt(const std::string& user_input, bool is_first_prompt, bool use_instruct) {
    if (use_instruct) {
        if (is_first_prompt) {
            return "<s>[INST] " + user_input + " [/INST] ";
        } else {
            return " </s> <s>[INST] " + user_input + " [/INST] ";
        }
    } else {
        if (is_first_prompt) {
            return "A chat between a curious human (\"Human\") and an artificial intelligence assistant (\"Assistant\"). The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n### Human: " + user_input + "\n### Assistant: ";
        } else {
            return "### Human: " + user_input + "\n### Assistant: \n";
        }
    }
}

std::string build_llava_prompt(const std::string& user_input, int prompt_iter) {
    if (prompt_iter == 0) {
        return "This is a chat between a user and an assistant.\n\n### USER: ";
    } else if (prompt_iter == 1) {
        return "\n" + user_input + "\n### ASSISTANT:";
    } else {
        return "### USER: " + user_input + "\n### ASSISTANT: \n";
    }
}

std::string build_vila_prompt(const std::string& user_input, int prompt_iter) {
    // VILA uses the same prompt format as LLaVA
    return build_llava_prompt(user_input, prompt_iter);
}

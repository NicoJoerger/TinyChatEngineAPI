#include <iostream>
#include <map>
#include <string>
#include <cstring>
#include <sstream>

#include "Generate.h"
#include "interface.h"
#include "model_config.h"

int NUM_THREAD = 5;

int main(int argc, char* argv[]) {
    bool use_voicechat = false;

    // Check for optional arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-v") == 0) {
            use_voicechat = true;
            // Remove the flag from argc and argv
            for (int j = i; j < argc - 1; ++j) {
                argv[j] = argv[j + 1];
            }
            --argc;
            break;
        }
    }

    std::string target_model = "LLaMA_3_8B_Instruct";
    std::string target_data_format = "INT4";
    bool instruct = true;
    std::string img_path = "images/monalisa.jpg";
    Profiler::getInstance().for_demo = true;

    // Set prompt color
    set_print_yellow();
    std::cout << "TinyChatEngine by MIT HAN Lab: https://github.com/mit-han-lab/TinyChatEngine" << std::endl;
    
    if (argc >= 3 && argc <= 5) {
        auto target_str = argv[1];
        target_model = argv[1];
        
        if (argc >= 4) {
            NUM_THREAD = atoi(argv[3]);
        }
        if (argc == 5) {
            if (isCodeLLaMA(target_model) or isMistral(target_model)) {
                instruct = convertToBool(argv[4]);
            }
            else if (isLLaVA(target_model) || isVILA(target_model)) {
                img_path = argv[4];
            }
        }

        if (!is_valid_model(target_model)) {
            std::cerr << "Model config:" << target_str << " unsupported" << std::endl;
            std::cerr << "Please select one of the following:";
            for (const auto& k : get_supported_models()) {
                std::cerr << k << ", ";
            }
            std::cerr << std::endl;
            throw("Unsupported model\n");
        }
        std::cout << "Using model: " << argv[1] << std::endl;

        auto data_format_input = argv[2];
        if (!is_valid_format(data_format_input)) {
            std::cerr << "Data format:" << data_format_input << " unsupported" << std::endl;
            std::cerr << "Please select one of the following: ";
            for (const auto& k : get_supported_formats()) {
                std::cerr << k << ", ";
            }
            std::cerr << std::endl;
            throw("Unsupported data format\n");
        }
        target_data_format = argv[2];
        if (target_data_format == "INT4" || target_data_format == "int4")
            std::cout << "Using AWQ for 4bit quantization: https://github.com/mit-han-lab/llm-awq" << std::endl;
        else
            std::cout << "Using data format: " << argv[2] << std::endl;
    } else if (argc == 2) {
        auto target_str = argv[1];
        target_model = argv[1];
        if (!is_valid_model(target_model)) {
            std::cerr << "Model config:" << target_str << " unsupported" << std::endl;
            std::cerr << "Please select one of the following: ";
            for (const auto& k : get_supported_models()) {
                std::cerr << k << ", ";
            }
            std::cerr << std::endl;
            throw("Unsupported model\n");
        }
        std::cout << "Using model: " << argv[1] << std::endl;

        auto data_format_input = "INT4";
    } else {
        if (isLLaMA(target_model) || isCodeLLaMA(target_model) || isStarCoder(target_model) || isLLaVA(target_model) || isVILA(target_model) || isMistral(target_model)) {
            std::cout << "Using model: " + target_model << std::endl;
            if (target_data_format == "INT4" || target_data_format == "int4")
                std::cout << "Using AWQ for 4bit quantization: https://github.com/mit-han-lab/llm-awq" << std::endl;
            else
                std::cout << "Using data format: " << target_data_format << std::endl;
        }
        else {  // OPT
            target_model = "OPT6.7B";
            target_data_format = "INT8";
            std::cout << "Using model: " + target_model << std::endl;
            std::cout << "Using data format: " + target_data_format << std::endl;
        }
    }

    if (isLLaMA3(target_model)) {
        int format_id = get_data_format_id(target_data_format);

        // Voicechat instructions
        if (use_voicechat) {
            std::cout << "You are using the TinyVoiceChat." << std::endl;
            std::cout << "*Usage instructions*" << std::endl;
            std::cout << "- Please use this mode in a quiet environment to have a better user experience and avoid speech misdetection." << std::endl;
            std::cout << "- Please start speaking after \"USER: [Start speaking]\" shows up." << std::endl;
            std::cout << "- Please press `Ctrl+C` multiple times to exit the program." << std::endl << std::endl;
        }

        // Load model
        std::cout << "Loading model... " << std::flush;
        int model_id = get_model_id(target_model);
        std::string m_path = get_model_path(target_model);

        #ifdef MODEL_PREFIX
        m_path = MODEL_PREFIX + m_path;
        #endif

        struct opt_params generation_config = get_llama3_default_config();

        bool first_prompt = true;

        if (format_id == FP32) {
            Fp32LlamaForCausalLM model = Fp32LlamaForCausalLM(m_path, get_opt_model_config(model_id));
            std::cout << "Finished!" << std::endl;
            set_print_yellow();
            std::cout << "Available commands: /show (display settings), /set <param> <value> (change setting), /new (reset conversation), quit" << std::endl << std::endl;
            set_print_reset();

            // Get input from the user
            while (true) {
                std::string input;
                if (use_voicechat) {
                    // Set prompt color
                    set_print_yellow();
                    int result = std::system("./application/sts_utils/listen");
                    std::ifstream in("tmpfile");
                    // set user input color
                    set_print_red();
                    std::getline(in, input);
                    result = std::system("rm tmpfile");
                    (void)result;
                    std::cout << input << std::endl;
                    // reset color
                    set_print_reset();
                } else {
                    // Set prompt color
                    set_print_yellow();
                    std::cout << "USER: ";
                    // set user input color
                    set_print_red();
                    std::getline(std::cin, input);
                    // reset color
                    set_print_reset();
                }
                if (input == "quit" || input == "Quit" || input == "Quit." || input == "quit.")
                    break;

                // Check for /new command
                if (input == "/new") {
                    // 1. Clear conversation state vectors
                    LLaMA3ResetConversationState();

                    // 2. Get model config for reinitialization
                    struct model_config config = get_opt_model_config(model_id);

                    // 3. Free and reinitialize all static memory buffers
                    // This prevents memory leaks by deallocating before reallocating
                    Int4llamaDecoderLayer::initialize_decoder_memory(config);
                    Int4llamaAttention::initialized_memory(config);

                    // 4. Reset cache buffer indices
                    Int4llamaAttention::reset_cache(config);

                    // 5. Reset local state
                    first_prompt = true;

                    // Show confirmation message
                    set_print_yellow();
                    std::cout << "\n[Conversation reset - starting fresh]\n\n";
                    set_print_reset();
                    continue;
                }

                // Check for /show command
                if (input == "/show") {
                    show_generation_config(generation_config);
                    continue;
                }

                // Check for /set command
                if (input.substr(0, 5) == "/set ") {
                    std::istringstream iss(input.substr(5));
                    std::string param, value;
                    if (iss >> param >> value) {
                        set_generation_param(generation_config, param, value);
                    } else {
                        set_print_yellow();
                        std::cout << "Usage: /set <parameter> <value>\n";
                        std::cout << "Example: /set temp 0.8\n\n";
                        set_print_reset();
                    }
                    continue;
                }

                if (instruct) {
                    std::cout << "ASSISTANT: ";
                }

                input = build_llama3_prompt(input, first_prompt);
                if (first_prompt) {
                    first_prompt = false;
                }

                LLaMA3Generate(m_path, &model, LLaMA_FP32, input, generation_config, "models/llama3_vocab.bin", true, false);
            }
        } else if (format_id == INT4) {
            m_path = "INT4/" + m_path;
            Int4LlamaForCausalLM model = Int4LlamaForCausalLM(m_path, get_opt_model_config(model_id));
            std::cout << "Finished!" << std::endl;
            set_print_yellow();
            std::cout << "Available commands: /show (display settings), /set <param> <value> (change setting), /new (reset conversation), quit" << std::endl << std::endl;
            set_print_reset();

            // Get input from the user
            while (true) {
                std::string input;
                if (use_voicechat) {
                    // Set prompt color
                    set_print_yellow();
                    int result = std::system("./application/sts_utils/listen");
                    std::ifstream in("tmpfile");
                    // set user input color
                    set_print_red();
                    std::getline(in, input);
                    result = std::system("rm tmpfile");
                    (void)result;
                    std::cout << input << std::endl;
                    // reset color
                    set_print_reset();
                } else {
                    // Set prompt color
                    set_print_yellow();
                    std::cout << "USER: ";
                    // set user input color
                    set_print_red();
                    std::getline(std::cin, input);
                    // reset color
                    set_print_reset();
                }
                if (input == "quit" || input == "Quit" || input == "Quit." || input == "quit.")
                    break;

                // Check for /new command
                if (input == "/new") {
                    // 1. Clear conversation state vectors
                    LLaMA3ResetConversationState();

                    // 2. Get model config for reinitialization
                    struct model_config config = get_opt_model_config(model_id);

                    // 3. Free and reinitialize all static memory buffers
                    // This prevents memory leaks by deallocating before reallocating
                    Int4llamaDecoderLayer::initialize_decoder_memory(config);
                    Int4llamaAttention::initialized_memory(config);

                    // 4. Reset cache buffer indices
                    Int4llamaAttention::reset_cache(config);

                    // 5. Reset local state
                    first_prompt = true;

                    // Show confirmation message
                    set_print_yellow();
                    std::cout << "\n[Conversation reset - starting fresh]\n\n";
                    set_print_reset();
                    continue;
                }

                // Check for /show command
                if (input == "/show") {
                    show_generation_config(generation_config);
                    continue;
                }

                // Check for /set command
                if (input.substr(0, 5) == "/set ") {
                    std::istringstream iss(input.substr(5));
                    std::string param, value;
                    if (iss >> param >> value) {
                        set_generation_param(generation_config, param, value);
                    } else {
                        set_print_yellow();
                        std::cout << "Usage: /set <parameter> <value>\n";
                        std::cout << "Example: /set temp 0.8\n\n";
                        set_print_reset();
                    }
                    continue;
                }

                if (instruct) {
                    std::cout << "ASSISTANT: ";
                }

                input = build_llama3_prompt(input, first_prompt);
                if (first_prompt) {
                    first_prompt = false;
                }

                LLaMA3Generate(m_path, &model, LLaMA_INT4, input, generation_config, "models/llama3_vocab.bin", true, use_voicechat);
            }
        } else {
            std::cout << std::endl;
            std::cerr << "At this time, we only support FP32 and INT4 for LLaMA_3_8B_Instruct." << std::endl;
        }
    } else if (isLLaMA(target_model)) {
        int format_id = get_data_format_id(target_data_format);

        // Voicechat instructions
        if (use_voicechat) {
            std::cout << "You are using the TinyVoiceChat." << std::endl;
            std::cout << "*Usage instructions*" << std::endl;
            std::cout << "- Please use this mode in a quiet environment to have a better user experience and avoid speech misdetection." << std::endl;
            std::cout << "- Please start speaking after \"USER: [Start speaking]\" shows up." << std::endl;
            std::cout << "- Please press `Ctrl+C` multiple times to exit the program." << std::endl << std::endl;
        }

        // Load model
        std::cout << "Loading model... " << std::flush;
        int model_id = get_model_id(target_model);
        std::string m_path = get_model_path(target_model);

        #ifdef MODEL_PREFIX
        m_path = MODEL_PREFIX + m_path;
        #endif

        struct opt_params generation_config = get_llama_default_config(target_model);

        bool first_prompt = true;

        if (format_id == FP32) {
            Fp32LlamaForCausalLM model = Fp32LlamaForCausalLM(m_path, get_opt_model_config(model_id));
            std::cout << "Finished!" << std::endl;
            set_print_yellow();
            std::cout << "Available commands: /show (display settings), /set <param> <value> (change setting), quit" << std::endl << std::endl;
            set_print_reset();

            // Get input from the user
            while (true) {
                std::string input;
                if (use_voicechat) {
                    // Set prompt color
                    set_print_yellow();
                    int result = std::system("./application/sts_utils/listen");
                    std::ifstream in("tmpfile");
                    // set user input color
                    set_print_red();
                    std::getline(in, input);
                    result = std::system("rm tmpfile");
                    (void)result;
                    std::cout << input << std::endl;
                    // reset color
                    set_print_reset();
                } else {
                    // Set prompt color
                    set_print_yellow();
                    std::cout << "USER: ";
                    // set user input color
                    set_print_red();
                    std::getline(std::cin, input);
                    // reset color
                    set_print_reset();
                }
                if (input == "quit" || input == "Quit" || input == "Quit." || input == "quit.")
                    break;

                // Check for /show command
                if (input == "/show") {
                    show_generation_config(generation_config);
                    continue;
                }

                // Check for /set command
                if (input.substr(0, 5) == "/set ") {
                    std::istringstream iss(input.substr(5));
                    std::string param, value;
                    if (iss >> param >> value) {
                        set_generation_param(generation_config, param, value);
                    } else {
                        set_print_yellow();
                        std::cout << "Usage: /set <parameter> <value>\n";
                        std::cout << "Example: /set temp 0.8\n\n";
                        set_print_reset();
                    }
                    continue;
                }

                if (instruct) {
                    std::cout << "ASSISTANT: ";
                } else {
                    if (isCodeLLaMA(target_model)) {
                        std::cout << input;
                    }
                }

                input = build_llama_prompt(input, first_prompt, isCodeLLaMA(target_model));
                if (first_prompt) {
                    first_prompt = false;
                }

                LLaMAGenerate(m_path, &model, LLaMA_FP32, input, generation_config, "models/llama_vocab.bin", true, false);
            }
        } else if (format_id == INT4) {
            m_path = "INT4/" + m_path;
            Int4LlamaForCausalLM model = Int4LlamaForCausalLM(m_path, get_opt_model_config(model_id));
            std::cout << "Finished!" << std::endl;
            set_print_yellow();
            std::cout << "Available commands: /show (display settings), /set <param> <value> (change setting), quit" << std::endl << std::endl;
            set_print_reset();

            // Get input from the user
            while (true) {
                std::string input;
                if (use_voicechat) {
                    // Set prompt color
                    set_print_yellow();
                    int result = std::system("./application/sts_utils/listen");
                    std::ifstream in("tmpfile");
                    // set user input color
                    set_print_red();
                    std::getline(in, input);
                    result = std::system("rm tmpfile");
                    (void)result;
                    std::cout << input << std::endl;
                    // reset color
                    set_print_reset();
                } else {
                    // Set prompt color
                    set_print_yellow();
                    std::cout << "USER: ";
                    // set user input color
                    set_print_red();
                    std::getline(std::cin, input);
                    // reset color
                    set_print_reset();
                }
                if (input == "quit" || input == "Quit" || input == "Quit." || input == "quit.")
                    break;

                // Check for /show command
                if (input == "/show") {
                    show_generation_config(generation_config);
                    continue;
                }

                // Check for /set command
                if (input.substr(0, 5) == "/set ") {
                    std::istringstream iss(input.substr(5));
                    std::string param, value;
                    if (iss >> param >> value) {
                        set_generation_param(generation_config, param, value);
                    } else {
                        set_print_yellow();
                        std::cout << "Usage: /set <parameter> <value>\n";
                        std::cout << "Example: /set temp 0.8\n\n";
                        set_print_reset();
                    }
                    continue;
                }

                if (instruct) {
                    std::cout << "ASSISTANT: ";
                } else {
                    if (isCodeLLaMA(target_model)) {
                        std::cout << input;
                    }
                }

                input = build_llama_prompt(input, first_prompt, isCodeLLaMA(target_model));
                if (first_prompt) {
                    first_prompt = false;
                }

                LLaMAGenerate(m_path, &model, LLaMA_INT4, input, generation_config, "models/llama_vocab.bin", true, use_voicechat);
            }
        } else {
            std::cout << std::endl;
            std::cerr << "At this time, we only support FP32 and INT4 for LLaMA_7B." << std::endl;
        }
    } else if (isStarCoder(target_model)) {
        int format_id = get_data_format_id(target_data_format);

        // Load model
        std::cout << "Loading model... " << std::flush;
        int model_id = get_model_id(target_model);
        std::string m_path = get_model_path(target_model);

        #ifdef MODEL_PREFIX
        m_path = MODEL_PREFIX + m_path;
        #endif

        struct opt_params generation_config = get_starcoder_default_config();

        if (format_id == FP32) {
            Fp32GPTBigCodeForCausalLM model = Fp32GPTBigCodeForCausalLM(m_path, get_opt_model_config(model_id));
            std::cout << "Finished!" << std::endl << std::endl;

            // Get input from the user
            while (true) {
                // Set prompt color
                set_print_yellow();
                std::cout << "USER: ";
                std::string input;
                // set user input color
                set_print_red();
                std::getline(std::cin, input);
                std::cout << input;
                // reset color
                set_print_reset();

                GPTBigCodeGenerate(m_path, &model, StarCoder_FP32, input, generation_config, "models/starcoder_vocab.bin", true);
            }
        } else if (format_id == INT4) {
            m_path = "INT4/" + m_path;
            Int4GPTBigCodeForCausalLM model = Int4GPTBigCodeForCausalLM(m_path, get_opt_model_config(model_id));
            std::cout << "Finished!" << std::endl << std::endl;

            // Get input from the user
            while (true) {
                // Set prompt color
                set_print_yellow();
                std::cout << "USER: ";
                std::string input;
                // set user input color
                set_print_red();
                std::getline(std::cin, input);
                std::cout << input;
                // reset color
                set_print_reset();

                GPTBigCodeGenerate(m_path, &model, StarCoder_INT4, input, generation_config, "models/starcoder_vocab.bin", true);    
            }
        } else {
            std::cout << std::endl;
            std::cerr << "At this time, we only support FP32 and INT4 for StarCoder." << std::endl;
        }
    } else if (isLLaVA(target_model)) {
        int format_id = get_data_format_id(target_data_format);

        // Voicechat instructions
        if (use_voicechat) {
            std::cout << "You are using the TinyVoiceChat." << std::endl;
            std::cout << "*Usage instructions*" << std::endl;
            std::cout << "- Please use this mode in a quiet environment to have a better user experience and avoid speech misdetection." << std::endl;
            std::cout << "- Please start speaking after \"USER: [Start speaking]\" shows up." << std::endl;
            std::cout << "- Please press `Ctrl+C` multiple times to exit the program." << std::endl << std::endl;
        }

        // Load model
        std::cout << "Loading model... " << std::flush;
        std::string clip_m_path = get_model_path("Clip_ViT_Large");
        std::string llama_m_path = get_model_path(target_model);

        int clip_model_id = get_model_id("Clip_ViT_Large");
        int llama_model_id = get_model_id(target_model);

        #ifdef MODEL_PREFIX
        llama_m_path = MODEL_PREFIX + llama_m_path;
        #endif

        struct opt_params generation_config = get_llava_default_config();

        int prompt_iter = 0;

        if (format_id == FP32) {
            Fp32CLIPVisionTransformer clip_model = Fp32CLIPVisionTransformer(clip_m_path, get_opt_model_config(clip_model_id), false);
            Fp32LlamaForCausalLM llama_model = Fp32LlamaForCausalLM(llama_m_path, get_opt_model_config(llama_model_id));

            // Get input from the user
            while (true) {
                std::string input;
                if (prompt_iter == 1) {
                    // Set prompt color
                    set_print_yellow();
                    std::cout << "Finished!" << std::endl << std::endl;
                    // reset color
                    set_print_reset();
                }
                if (prompt_iter > 0) {
                    if (use_voicechat) {
                        // Set prompt color
                        set_print_yellow();
                        int result = std::system("./application/sts_utils/listen");
                        std::ifstream in("tmpfile");
                        // set user input color
                        set_print_red();
                        std::getline(in, input);
                        result = std::system("rm tmpfile");
                        (void)result;
                        std::cout << input << std::endl;
                        // reset color
                        set_print_reset();
                    } else {
                        // Set prompt color
                        set_print_yellow();
                        std::cout << "USER: ";
                        // set user input color
                        set_print_red();
                        std::getline(std::cin, input);
                        // reset color
                        set_print_reset();
                    }
                    if (input == "quit" || input == "Quit" || input == "Quit." || input == "quit.")
                        break;
                    std::cout << "ASSISTANT: ";
                }

                input = build_llava_prompt(input, prompt_iter);
                prompt_iter++;

                LLaVAGenerate(llama_m_path, &llama_model, clip_m_path, &clip_model, LLaVA_FP32, input, img_path, generation_config, "models/llama_vocab.bin", true, false, false);
            }
        } else if (format_id == INT4) {
            Fp32CLIPVisionTransformer clip_model = Fp32CLIPVisionTransformer(clip_m_path, get_opt_model_config(clip_model_id), false);
            llama_m_path = "INT4/" + llama_m_path;
            Int4LlamaForCausalLM llama_model = Int4LlamaForCausalLM(llama_m_path, get_opt_model_config(llama_model_id));

            // Get input from the user
            while (true) {
                if (prompt_iter == 1) {
                    // Set prompt color
                    set_print_yellow();
                    std::cout << "Finished!" << std::endl << std::endl;
                    // reset color
                    set_print_reset();
                }
                std::string input;
                if (prompt_iter > 0) {
                    if (use_voicechat) {
                        // Set prompt color
                        set_print_yellow();
                        int result = std::system("./application/sts_utils/listen");
                        std::ifstream in("tmpfile");
                        // set user input color
                        set_print_red();
                        std::getline(in, input);
                        result = std::system("rm tmpfile");
                        (void)result;
                        std::cout << input << std::endl;
                        // reset color
                        set_print_reset();
                    } else {
                        // Set prompt color
                        set_print_yellow();
                        std::cout << "USER: ";
                        // set user input color
                        set_print_red();
                        std::getline(std::cin, input);
                        // reset color
                        set_print_reset();
                    }
                    if (input == "quit" || input == "Quit" || input == "Quit." || input == "quit.")
                        break;
                    std::cout << "ASSISTANT: ";
                }

                input = build_llava_prompt(input, prompt_iter);
                prompt_iter++;

                LLaVAGenerate(llama_m_path, &llama_model, clip_m_path, &clip_model, LLaVA_INT4, input, img_path, generation_config, "models/llama_vocab.bin", true, use_voicechat, false);
            }
        } else {
            std::cout << std::endl;
            std::cerr << "At this time, we only support FP32 and INT4 for LLaVA_7B." << std::endl;
        }
    } else if (isVILA1_5(target_model)) {
        int format_id = get_data_format_id(target_data_format);

        // Voicechat instructions
        if (use_voicechat) {
            std::cout << "You are using the TinyVoiceChat." << std::endl;
            std::cout << "*Usage instructions*" << std::endl;
            std::cout << "- Please use this mode in a quiet environment to have a better user experience and avoid speech misdetection." << std::endl;
            std::cout << "- Please start speaking after \"USER: [Start speaking]\" shows up." << std::endl;
            std::cout << "- Please press `Ctrl+C` multiple times to exit the program." << std::endl << std::endl;
        }

        // Load model
        std::cout << "Loading model... " << std::flush;
        std::string clip_m_path = get_model_path("Clip_ViT_Large");
        std::string llama_m_path = get_model_path(target_model);

        int clip_model_id = get_model_id("Clip_ViT_Large");
        int llama_model_id = get_model_id(target_model);

        #ifdef MODEL_PREFIX
        llama_m_path = MODEL_PREFIX + llama_m_path;
        #endif

        struct opt_params generation_config = get_vila_default_config();

        int prompt_iter = 0;

        if (format_id == FP32) {
            Fp32CLIPVisionTransformer clip_model = Fp32CLIPVisionTransformer(clip_m_path, get_opt_model_config(clip_model_id), true);
            Fp32LlamaForCausalLM llama_model = Fp32LlamaForCausalLM(llama_m_path, get_opt_model_config(llama_model_id));

            // Get input from the user
            while (true) {
                std::string input;
                if (prompt_iter == 1) {
                    // Set prompt color
                    set_print_yellow();
                    std::cout << "Finished!" << std::endl << std::endl;
                    // reset color
                    set_print_reset();
                }
                if (prompt_iter > 0) {
                    if (use_voicechat) {
                        // Set prompt color
                        set_print_yellow();
                        int result = std::system("./application/sts_utils/listen");
                        std::ifstream in("tmpfile");
                        // set user input color
                        set_print_red();
                        std::getline(in, input);
                        result = std::system("rm tmpfile");
                        (void)result;
                        std::cout << input << std::endl;
                        // reset color
                        set_print_reset();
                    } else {
                        // Set prompt color
                        set_print_yellow();
                        std::cout << "USER: ";
                        // set user input color
                        set_print_red();
                        std::getline(std::cin, input);
                        // reset color
                        set_print_reset();
                    }
                    if (input == "quit" || input == "Quit" || input == "Quit." || input == "quit.")
                        break;
                    std::cout << "ASSISTANT: ";
                }

                input = build_llava_prompt(input, prompt_iter);
                prompt_iter++;

                LLaVAGenerate(llama_m_path, &llama_model, clip_m_path, &clip_model, VILA_FP32, input, img_path, generation_config, "models/llama_vocab.bin", true, false, true);
            }
        } else if (format_id == INT4) {
            Fp32CLIPVisionTransformer clip_model = Fp32CLIPVisionTransformer(clip_m_path, get_opt_model_config(clip_model_id), true);
            llama_m_path = "INT4/" + llama_m_path;
            Int4LlamaForCausalLM llama_model = Int4LlamaForCausalLM(llama_m_path, get_opt_model_config(llama_model_id));

            // Get input from the user
            while (true) {
                if (prompt_iter == 1) {
                    // Set prompt color
                    set_print_yellow();
                    std::cout << "Finished!" << std::endl << std::endl;
                    // reset color
                    set_print_reset();
                }
                std::string input;
                if (prompt_iter > 0) {
                    if (use_voicechat) {
                        // Set prompt color
                        set_print_yellow();
                        int result = std::system("./application/sts_utils/listen");
                        std::ifstream in("tmpfile");
                        // set user input color
                        set_print_red();
                        std::getline(in, input);
                        result = std::system("rm tmpfile");
                        (void)result;
                        std::cout << input << std::endl;
                        // reset color
                        set_print_reset();
                    } else {
                        // Set prompt color
                        set_print_yellow();
                        std::cout << "USER: ";
                        // set user input color
                        set_print_red();
                        std::getline(std::cin, input);
                        // reset color
                        set_print_reset();
                    }
                    if (input == "quit" || input == "Quit" || input == "Quit." || input == "quit.")
                        break;
                    std::cout << "ASSISTANT: ";
                }

                input = build_llava_prompt(input, prompt_iter);
                prompt_iter++;

                LLaVAGenerate(llama_m_path, &llama_model, clip_m_path, &clip_model, VILA_INT4, input, img_path, generation_config, "models/llama_vocab.bin", true, use_voicechat, true);
            }
        } else {
            std::cout << std::endl;
            std::cerr << "At this time, we only support FP32 and INT4 for VILA1.5_8B." << std::endl;
        }
    } else if (isVILA(target_model)) {
        int format_id = get_data_format_id(target_data_format);

        // Voicechat instructions
        if (use_voicechat) {
            std::cout << "You are using the TinyVoiceChat." << std::endl;
            std::cout << "*Usage instructions*" << std::endl;
            std::cout << "- Please use this mode in a quiet environment to have a better user experience and avoid speech misdetection." << std::endl;
            std::cout << "- Please start speaking after \"USER: [Start speaking]\" shows up." << std::endl;
            std::cout << "- Please press `Ctrl+C` multiple times to exit the program." << std::endl << std::endl;
        }

        // Load model
        std::cout << "Loading model... " << std::flush;
        std::string clip_m_path = get_model_path("Clip_ViT_Large");
        std::string llama_m_path = get_model_path(target_model);

        int clip_model_id = get_model_id("Clip_ViT_Large");
        int llama_model_id = get_model_id(target_model);

        #ifdef MODEL_PREFIX
        llama_m_path = MODEL_PREFIX + llama_m_path;
        #endif

        struct opt_params generation_config = get_vila_default_config();

        int prompt_iter = 0;

        if (format_id == FP32) {
            Fp32CLIPVisionTransformer clip_model = Fp32CLIPVisionTransformer(clip_m_path, get_opt_model_config(clip_model_id), true);
            Fp32LlamaForCausalLM llama_model = Fp32LlamaForCausalLM(llama_m_path, get_opt_model_config(llama_model_id));

            // Get input from the user
            while (true) {
                std::string input;
                if (prompt_iter == 1) {
                    // Set prompt color
                    set_print_yellow();
                    std::cout << "Finished!" << std::endl << std::endl;
                    // reset color
                    set_print_reset();
                }
                if (prompt_iter > 0) {
                    if (use_voicechat) {
                        // Set prompt color
                        set_print_yellow();
                        int result = std::system("./application/sts_utils/listen");
                        std::ifstream in("tmpfile");
                        // set user input color
                        set_print_red();
                        std::getline(in, input);
                        result = std::system("rm tmpfile");
                        (void)result;
                        std::cout << input << std::endl;
                        // reset color
                        set_print_reset();
                    } else {
                        // Set prompt color
                        set_print_yellow();
                        std::cout << "USER: ";
                        // set user input color
                        set_print_red();
                        std::getline(std::cin, input);
                        // reset color
                        set_print_reset();
                    }
                    if (input == "quit" || input == "Quit" || input == "Quit." || input == "quit.")
                        break;
                    std::cout << "ASSISTANT: ";
                }

                input = build_llava_prompt(input, prompt_iter);
                prompt_iter++;

                LLaVAGenerate(llama_m_path, &llama_model, clip_m_path, &clip_model, VILA_FP32, input, img_path, generation_config, "models/llama_vocab.bin", true, false, true);
            }
        } else if (format_id == INT4) {
            Fp32CLIPVisionTransformer clip_model = Fp32CLIPVisionTransformer(clip_m_path, get_opt_model_config(clip_model_id), true);
            llama_m_path = "INT4/" + llama_m_path;
            Int4LlamaForCausalLM llama_model = Int4LlamaForCausalLM(llama_m_path, get_opt_model_config(llama_model_id));

            // Get input from the user
            while (true) {
                if (prompt_iter == 1) {
                    // Set prompt color
                    set_print_yellow();
                    std::cout << "Finished!" << std::endl << std::endl;
                    // reset color
                    set_print_reset();
                }
                std::string input;
                if (prompt_iter > 0) {
                    if (use_voicechat) {
                        // Set prompt color
                        set_print_yellow();
                        int result = std::system("./application/sts_utils/listen");
                        std::ifstream in("tmpfile");
                        // set user input color
                        set_print_red();
                        std::getline(in, input);
                        result = std::system("rm tmpfile");
                        (void)result;
                        std::cout << input << std::endl;
                        // reset color
                        set_print_reset();
                    } else {
                        // Set prompt color
                        set_print_yellow();
                        std::cout << "USER: ";
                        // set user input color
                        set_print_red();
                        std::getline(std::cin, input);
                        // reset color
                        set_print_reset();
                    }
                    if (input == "quit" || input == "Quit" || input == "Quit." || input == "quit.")
                        break;
                    std::cout << "ASSISTANT: ";
                }

                input = build_llava_prompt(input, prompt_iter);
                prompt_iter++;

                LLaVAGenerate(llama_m_path, &llama_model, clip_m_path, &clip_model, VILA_INT4, input, img_path, generation_config, "models/llama_vocab.bin", true, use_voicechat, true);
            }
        } else {
            std::cout << std::endl;
            std::cerr << "At this time, we only support FP32 and INT4 for VILA_7B." << std::endl;
        }
    } else if (isMistral(target_model)) {
        int format_id = get_data_format_id(target_data_format);

        // Voicechat instructions
        if (use_voicechat) {
            std::cout << "You are using the TinyVoiceChat." << std::endl;
            std::cout << "*Usage instructions*" << std::endl;
            std::cout << "- Please use this mode in a quiet environment to have a better user experience and avoid speech misdetection." << std::endl;
            std::cout << "- Please start speaking after \"USER: [Start speaking]\" shows up." << std::endl;
            std::cout << "- Please press `Ctrl+C` multiple times to exit the program." << std::endl << std::endl;
        }

        // Load model
        std::cout << "Loading model... " << std::flush;
        int model_id = get_model_id(target_model);
        std::string m_path = get_model_path(target_model);

        #ifdef MODEL_PREFIX
        m_path = MODEL_PREFIX + m_path;
        #endif

        struct opt_params generation_config = get_mistral_default_config();

        bool first_prompt = true;

        if (format_id == FP32) {
            Fp32LlamaForCausalLM model = Fp32LlamaForCausalLM(m_path, get_opt_model_config(model_id));
            std::cout << "Finished!" << std::endl << std::endl;

            // Get input from the user
            while (true) {
                std::string input;
                if (use_voicechat) {
                    // Set prompt color
                    set_print_yellow();
                    int result = std::system("./application/sts_utils/listen");
                    std::ifstream in("tmpfile");
                    // set user input color
                    set_print_red();
                    std::getline(in, input);
                    result = std::system("rm tmpfile");
                    (void)result;
                    std::cout << input << std::endl;
                    // reset color
                    set_print_reset();
                } else {
                    // Set prompt color
                    set_print_yellow();
                    std::cout << "USER: ";
                    // set user input color
                    set_print_red();
                    std::getline(std::cin, input);
                    // reset color
                    set_print_reset();
                }
                if (input == "quit" || input == "Quit" || input == "Quit." || input == "quit.")
                    break;

                std::cout << "ASSISTANT: ";
                input = build_mistral_prompt(input, first_prompt, instruct);
                if (first_prompt) {
                    first_prompt = false;
                }

                MistralGenerate(m_path, &model, LLaMA_FP32, input, generation_config, "models/mistral_vocab.bin", true, false);
            }
        } else if (format_id == INT4) {
            m_path = "INT4/" + m_path;
            Int4LlamaForCausalLM model = Int4LlamaForCausalLM(m_path, get_opt_model_config(model_id));
            std::cout << "Finished!" << std::endl << std::endl;
            
            // Get input from the user
            while (true) {
                std::string input;
                if (use_voicechat) {
                    // Set prompt color
                    set_print_yellow();
                    int result = std::system("./application/sts_utils/listen");
                    std::ifstream in("tmpfile");
                    // set user input color
                    set_print_red();
                    std::getline(in, input);
                    result = std::system("rm tmpfile");
                    (void)result;
                    std::cout << input << std::endl;
                    // reset color
                    set_print_reset();
                } else {
                    // Set prompt color
                    set_print_yellow();
                    std::cout << "USER: ";
                    // set user input color
                    set_print_red();
                    std::getline(std::cin, input);
                    // reset color
                    set_print_reset();
                }
                if (input == "quit" || input == "Quit" || input == "Quit." || input == "quit.")
                    break;

                std::cout << "ASSISTANT: ";
                input = build_mistral_prompt(input, first_prompt, instruct);
                if (first_prompt) {
                    first_prompt = false;
                }

                MistralGenerate(m_path, &model, LLaMA_INT4, input, generation_config, "models/mistral_vocab.bin", true, use_voicechat);
            }
        } else {
            std::cout << std::endl;
            std::cerr << "At this time, we only support FP32 and INT4 for Mistral-7B." << std::endl;
        }
    } else {  // OPT
#ifdef QM_CUDA
        printf("OPT is not supported with CUDA backend yet.");
        exit(-1);
#else
        // Load model
        std::cout << "Loading model... " << std::flush;
        int model_id = get_model_id(target_model);
        std::string m_path = get_model_path(target_model);
        int format_id = get_data_format_id(target_data_format);

        // Load encoder
        std::string bpe_file = "models/opt_merges.txt";
        std::string vocab_file = "models/opt_vocab.json";
        Encoder encoder = get_encoder(vocab_file, bpe_file);
        std::string decode;

        struct opt_params generation_config;
        generation_config.n_predict = 512;
        if (format_id == QINT8) {
            OPTForCausalLM model = OPTForCausalLM("INT8/" + m_path, get_opt_model_config(model_id));
            std::cout << "Finished!" << std::endl << std::endl;
            
            // Get input from the user
            std::string input;
            if (use_voicechat) {
                // Set prompt color
                set_print_yellow();
                int result = std::system("./application/sts_utils/listen");
                std::ifstream in("tmpfile");
                // set user input color
                set_print_red();
                std::getline(in, input);
                result = std::system("rm tmpfile");
                (void)result;
                std::cout << input << std::endl;
                // reset color
                set_print_reset();
            } else {
                // Set prompt color
                set_print_yellow();
                std::cout << "USER: ";
                // set user input color
                set_print_red();
                std::getline(std::cin, input);
                // reset color
                set_print_reset();
            }
            std::vector<int> input_ids = encoder.encode(input);
            std::string decoded = encoder.decode(input_ids);

            // Generate
            std::vector<int> generated_ids =
                OPTGenerate(&model, OPT_INT8, input_ids, generation_config, &encoder, true, use_voicechat);
        } else if (format_id == FP32) {
            Fp32OPTForCausalLM model = Fp32OPTForCausalLM(m_path, get_opt_model_config(model_id));
            std::cout << "Finished!" << std::endl << std::endl;

            // Get input from the user
            std::string input;
            if (use_voicechat) {
                // Set prompt color
                set_print_yellow();
                int result = std::system("./application/sts_utils/listen");
                std::ifstream in("tmpfile");
                // set user input color
                set_print_red();
                std::getline(in, input);
                result = std::system("rm tmpfile");
                (void)result;
                std::cout << input << std::endl;
                // reset color
                set_print_reset();
            } else {
                // Set prompt color
                set_print_yellow();
                std::cout << "USER: ";
                // set user input color
                set_print_red();
                std::getline(std::cin, input);
                // reset color
                set_print_reset();
            }
            std::vector<int> input_ids = encoder.encode(input);
            std::string decoded = encoder.decode(input_ids);

            // Generate
            std::vector<int> generated_ids =
                OPTGenerate(&model, OPT_FP32, input_ids, generation_config, &encoder, true, use_voicechat);
        } else if (format_id == INT4) {
            Int4OPTForCausalLM model = Int4OPTForCausalLM("INT4/" + m_path, get_opt_model_config(model_id));
            std::cout << "Finished!" << std::endl << std::endl;

            // Get input from the user
            std::string input;
            if (use_voicechat) {
                // Set prompt color
                set_print_yellow();
                int result = std::system("./application/sts_utils/listen");
                std::ifstream in("tmpfile");
                // set user input color
                set_print_red();
                std::getline(in, input);
                result = std::system("rm tmpfile");
                (void)result;
                std::cout << input << std::endl;
                // reset color
                set_print_reset();
            } else {
                // Set prompt color
                set_print_yellow();
                std::cout << "USER: ";
                // set user input color
                set_print_red();
                std::getline(std::cin, input);
                // reset color
                set_print_reset();
            }
            
            std::vector<int> input_ids = encoder.encode(input);
            std::string decoded = encoder.decode(input_ids);

            // Generate
            std::vector<int> generated_ids =
                OPTGenerate(&model, OPT_INT4, input_ids, generation_config, &encoder, true, use_voicechat);
        }
#endif  // QN_CUDA
    }
};

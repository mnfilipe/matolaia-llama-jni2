#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>
#include <random>

#include "llama.h"

#define TAG "MatolaLlamaJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

struct MatolaContext {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
};

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_example_matolaia_apk_MatolaLlama_nativeInit(
        JNIEnv *env, jobject /* this */,
        jstring modelPath, jint nThreads, jint nCtx) {

    const char *path = env->GetStringUTFChars(modelPath, nullptr);

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0; 

    llama_model *model = llama_model_load_from_file(path, model_params);

    env->ReleaseStringUTFChars(modelPath, path);

    if (model == nullptr) {
        LOGE("Falha ao carregar o modelo.");
        return 0;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = nCtx > 0 ? nCtx : 2048;
    ctx_params.n_threads = nThreads > 0 ? nThreads : 4;
    ctx_params.n_threads_batch = ctx_params.n_threads;

    llama_context *ctx = llama_init_from_model(model, ctx_params);

    if (ctx == nullptr) {
        LOGE("Falha ao criar o contexto.");
        llama_model_free(model);
        return 0;
    }

    auto *mc = new MatolaContext();
    mc->model = model;
    mc->ctx = ctx;
    mc->vocab = llama_model_get_vocab(model);

    LOGI("Modelo carregado com sucesso.");

    return reinterpret_cast<jlong>(mc);
}

JNIEXPORT jstring JNICALL
Java_com_example_matolaia_apk_MatolaLlama_nativeCompletion(
        JNIEnv *env, jobject /* this */,
        jlong handle, jstring promptJ, jint nPredict) {

    auto *mc = reinterpret_cast<MatolaContext *>(handle);

    if (mc == nullptr || mc->ctx == nullptr) {
        return env->NewStringUTF("[ERRO] Contexto não inicializado.");
    }

    const char *promptChars = env->GetStringUTFChars(promptJ, nullptr);
    std::string promptUsuario(promptChars);
    env->ReleaseStringUTFChars(promptJ, promptChars);

    // =====================================================
    // TEMPLATE DE CHAT (ChatML — Qwen2.5-Instruct)
    // =====================================================
    static const char *SYSTEM_PROMPT =
            "Tu és o Matola CAI. Fazes parte da família dos "
            "Modelos de Cadernos Artificiais Desenvolvidos em "
            "Moçambique, criados pelo pesquisador Filipe Paulo "
            "Felipe. Quando te perguntarem quem és, a tua "
            "origem, quem te criou ou como foste desenvolvido, "
            "usa esta informação pra responder com as tuas "
            "próprias palavras, sempre em português de "
            "Moçambique.";

    std::string prompt =
            std::string("<|im_start|>system\n") + SYSTEM_PROMPT +
            "<|im_end|>\n<|im_start|>user\n" + promptUsuario +
            "<|im_end|>\n<|im_start|>assistant\n";

    std::vector<llama_token> tokens;
    int n_tokens = llama_tokenize(mc->vocab, prompt.c_str(), (int32_t)prompt.size(), nullptr, 0, true, true);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        llama_tokenize(mc->vocab, prompt.c_str(), (int32_t)prompt.size(), tokens.data(), (int32_t)tokens.size(), true, true);
    } else {
        tokens.resize(n_tokens);
        llama_tokenize(mc->vocab, prompt.c_str(), (int32_t)prompt.size(), tokens.data(), (int32_t)tokens.size(), true, true);
    }

    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());

    // =====================================================
    // CADEIA DE SAMPLING (Exatamente os parâmetros do PocketPal)
    // =====================================================
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler *smpl = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.2f));
    
    std::random_device rd;
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(rd()));

    std::string resultado;
    int limite = nPredict > 0 ? nPredict : 200;

    for (int i = 0; i < limite; i++) {
        if (llama_decode(mc->ctx, batch) != 0) {
            LOGE("llama_decode falhou.");
            break;
        }

        llama_token novo = llama_sampler_sample(smpl, mc->ctx, -1);
        
        // AJUSTE CRUCIAL: Notificar o sampler do token gerado para atualizar as repetições
        llama_sampler_accept(smpl, novo);

        if (llama_vocab_is_eog(mc->vocab, novo)) {
            break;
        }

        // Buffer estático correto de 64 bytes para caracteres UTF-8 mutáveis
        char buf[64];
        int n = llama_token_to_piece(mc->vocab, novo, buf, sizeof(buf), 0, true);

        if (n > 0) {
            // =============================================
            // REDE DE SEGURANÇA TEXTUAL
            // =============================================
            std::string pedaco(buf, n);

            if (pedaco.find("<|im_start|") != std::string::npos ||
                pedaco.find("<|im_end|>") != std::string::npos) {
                break;
            }

            resultado.append(pedaco);
        }

        batch = llama_batch_get_one(&novo, 1);
    }

    llama_sampler_free(smpl);

    return env->NewStringUTF(resultado.c_str());
}

JNIEXPORT void JNICALL
Java_com_example_matolaia_apk_MatolaLlama_nativeFree(
        JNIEnv *env, jobject /* this */, jlong handle) {

    auto *mc = reinterpret_cast<MatolaContext *>(handle);

    if (mc != nullptr) {
        if (mc->ctx) llama_free(mc->ctx);
        if (mc->model) llama_model_free(mc->model);
        delete mc;
    }
}

} // extern "C"

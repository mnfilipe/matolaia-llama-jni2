#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>
#include <random>
#include <cctype>

#include "llama.h"

#define TAG "MatolaLlamaJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// Contexto mínimo. O LFM2.5-350M aguenta muito mais, e o KV cache dele é
// pequeno. Se o Kotlin pedir MAIS que isto, respeitamos; se pedir menos
// (ou 0), subimos para este valor — contexto curto era uma das causas do corte.
static const int CTX_MINIMO = 4096;

struct MatolaContext {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
};

// Se a geração parar no meio de um caractere UTF-8 (fim do contexto, erro de
// decode), remove os bytes soltos do fim para o JNI não receber lixo.
static void aparar_utf8_incompleto(std::string &s) {
    size_t n = s.size();
    if (n == 0) return;

    size_t i = n;
    int cont = 0;
    while (i > 0 && cont < 4 && (static_cast<unsigned char>(s[i - 1]) & 0xC0) == 0x80) {
        i--;
        cont++;
    }
    if (i == 0) { s.clear(); return; }

    unsigned char lead = static_cast<unsigned char>(s[i - 1]);
    if (lead < 0x80) return; // ASCII no fim: completo

    size_t precisa = lead >= 0xF0 ? 4 : (lead >= 0xE0 ? 3 : (lead >= 0xC0 ? 2 : 1));
    size_t tem = static_cast<size_t>(cont) + 1;
    if (tem < precisa) s.erase(i - 1);
}

// Normaliza para procurar frases: minúsculas, sem acentos, pontuação vira espaço.
static std::string normalizar_para_busca(const std::string &in) {
    std::string out;
    out.reserve(in.size() + 2);
    auto separador = [&out]() {
        if (!out.empty() && out.back() != ' ') out += ' ';
    };
    for (size_t i = 0; i < in.size(); i++) {
        unsigned char c = static_cast<unsigned char>(in[i]);
        if (c < 0x80) {
            if (std::isalnum(c)) out += static_cast<char>(std::tolower(c));
            else separador();
            continue;
        }
        if (c == 0xC3 && i + 1 < in.size()) {
            unsigned char d = static_cast<unsigned char>(in[++i]);
            if (d >= 0x80 && d <= 0x9F) d += 0x20; // maiúscula -> minúscula
            char base = 0;
            switch (d) {
                case 0xA0: case 0xA1: case 0xA2: case 0xA3: base = 'a'; break;
                case 0xA7: base = 'c'; break;
                case 0xA8: case 0xA9: case 0xAA: base = 'e'; break;
                case 0xAC: case 0xAD: base = 'i'; break;
                case 0xB2: case 0xB3: case 0xB4: case 0xB5: base = 'o'; break;
                case 0xB9: case 0xBA: base = 'u'; break;
            }
            if (base) out += base; else separador();
            continue;
        }
        separador();
    }
    return out;
}

// A identidade (Matola CAI / Filipe Paulo Felipe / Moçambique) só entra no
// prompt quando a pergunta é sobre o próprio modelo. Num modelo de 350M, um
// system prompt fixo "vaza" para respostas que não têm nada a ver.
// A busca é por palavras inteiras (" quem es " não apanha "quem escreveu").
static bool pergunta_sobre_identidade(const std::string &pergunta) {
    static const char *CHAVES[] = {
        "quem es", "quem e voce", "quem e tu", "o que es", "o que e voce",
        "como te chamas", "como voce se chama",
        "qual e o teu nome", "qual e o seu nome",
        "teu criador", "seu criador", "teus criadores", "tua origem",
        "te criou", "te fez", "te desenvolveu", "te programou", "te treinou",
        "te inventou", "te construiu", "te criaram", "te desenvolveram",
        "criou te", "fez te", "desenvolveu te", "programou te",
        "criou voce", "fez voce", "desenvolveu voce",
        "foste criado", "foste desenvolvido", "foste feito",
        "voce foi criado", "voce foi desenvolvido", "voce foi feito",
        "apresenta te", "apresente se", "matola cai",
        "de onde vens", "de onde voce vem"
    };
    const std::string p = " " + normalizar_para_busca(pergunta) + " ";
    for (const char *chave : CHAVES) {
        if (p.find(std::string(" ") + chave + " ") != std::string::npos) return true;
    }
    return false;
}

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

    const int n_ctx = nCtx > CTX_MINIMO ? nCtx : CTX_MINIMO;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_ctx;
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

    LOGI("Modelo carregado com sucesso. n_ctx=%d", n_ctx);

    return reinterpret_cast<jlong>(mc);
}

// nPredict é IGNORADO de propósito: a geração só termina quando o modelo
// emite o token de fim (EOG) ou quando o contexto enche. Sem limite artificial.
JNIEXPORT jstring JNICALL
Java_com_example_matolaia_apk_MatolaLlama_nativeCompletion(
        JNIEnv *env, jobject /* this */,
        jlong handle, jstring promptJ, jint /* nPredict */) {

    auto *mc = reinterpret_cast<MatolaContext *>(handle);

    if (mc == nullptr || mc->ctx == nullptr) {
        return env->NewStringUTF("[ERRO] Contexto não inicializado.");
    }

    const char *promptChars = env->GetStringUTFChars(promptJ, nullptr);
    std::string promptUsuario(promptChars);
    env->ReleaseStringUTFChars(promptJ, promptChars);

    // =====================================================
    // CADA PERGUNTA COMEÇA DO ZERO
    // Antes, o KV cache nunca era limpo: cada pergunta somava-se às
    // anteriores até o contexto encher, e aí o llama_decode falhava e a
    // resposta saía cortada. A página não envia histórico, então limpar é seguro.
    // =====================================================
    llama_memory_clear(llama_get_memory(mc->ctx), true);

    // =====================================================
    // TEMPLATE DE CHAT (ChatML — é o formato do LFM2.5 e do Qwen)
    // O BOS (<|startoftext|>) é adicionado pelo tokenizador (add_special=true).
    //
    // Perguntas normais vão SEM system prompt (como no PocketPal).
    // O system prompt de identidade só entra se perguntarem sobre o modelo.
    // =====================================================
    static const char *SYSTEM_IDENTIDADE =
            "Tu és o Matola CAI. Fazes parte da família dos "
            "Modelos de Cadernos Artificiais Desenvolvidos em "
            "Moçambique, criados pelo pesquisador moçambicano "
            "Filipe Paulo Felipe. Responde à pergunta com as "
            "tuas próprias palavras.";

    const bool identidade = pergunta_sobre_identidade(promptUsuario);

    std::string prompt;
    if (identidade) {
        prompt += std::string("<|im_start|>system\n") + SYSTEM_IDENTIDADE + "<|im_end|>\n";
    }
    prompt += std::string("<|im_start|>user\n") + promptUsuario +
              "<|im_end|>\n<|im_start|>assistant\n";

    LOGI("Pergunta de identidade: %s", identidade ? "sim" : "nao");

    std::vector<llama_token> tokens;
    int n_tokens = llama_tokenize(mc->vocab, prompt.c_str(), (int32_t)prompt.size(), nullptr, 0, true, true);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        llama_tokenize(mc->vocab, prompt.c_str(), (int32_t)prompt.size(), tokens.data(), (int32_t)tokens.size(), true, true);
    } else {
        tokens.resize(n_tokens);
        llama_tokenize(mc->vocab, prompt.c_str(), (int32_t)prompt.size(), tokens.data(), (int32_t)tokens.size(), true, true);
    }

    const int n_ctx = (int) llama_n_ctx(mc->ctx);
    const int n_prompt = (int) tokens.size();

    if (n_prompt >= n_ctx - 16) {
        LOGE("Prompt grande demais: %d tokens (n_ctx=%d).", n_prompt, n_ctx);
        return env->NewStringUTF("[ERRO] Pergunta grande demais para o contexto.");
    }

    // O único limite é o espaço que sobra no contexto.
    const int limite = n_ctx - n_prompt;

    // =====================================================
    // CADEIA DE SAMPLING
    // Igual ao PocketPal (top_k 40, top_p 0.95, min_p 0.05, XTC/typical/
    // mirostat desligados, seed aleatória) com duas diferenças:
    //  - temperatura 0.2 (determinística, intencional)
    //  - repeat penalty 1.05 nos últimos 64 tokens: com temperatura baixa
    //    e SEM limite de tokens, um modelo de 350M tende a entrar em loop
    //    ("...manteiga, manteiga, manteiga...") e nunca mais parar.
    //    Para ficar 100% idêntico ao PocketPal, troca 1.05f por 1.0f.
    // =====================================================
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler *smpl = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(llama_vocab_n_tokens(mc->vocab), 64, 1.05f, 0.0f, 0.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.2f));

    std::random_device rd;
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(rd()));

    std::string resultado;

    // O token tem de viver fora do loop: o batch guarda um ponteiro para ele
    // e é usado no decode da iteração seguinte.
    llama_token novo = 0;
    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());

    for (int i = 0; i < limite; i++) {
        int rc = llama_decode(mc->ctx, batch);
        if (rc != 0) {
            LOGE("llama_decode falhou (rc=%d) após %zu bytes gerados.", rc, resultado.size());
            break;
        }

        novo = llama_sampler_sample(smpl, mc->ctx, -1);
        llama_sampler_accept(smpl, novo);

        if (llama_vocab_is_eog(mc->vocab, novo)) {
            break;
        }

        // Peça do token (buffer cresce se o token for grande)
        char buf[256];
        std::string pedaco;
        int n = llama_token_to_piece(mc->vocab, novo, buf, sizeof(buf), 0, true);
        if (n < 0) {
            std::vector<char> grande(-n);
            n = llama_token_to_piece(mc->vocab, novo, grande.data(), (int32_t)grande.size(), 0, true);
            if (n > 0) pedaco.assign(grande.data(), n);
        } else if (n > 0) {
            pedaco.assign(buf, n);
        }

        if (!pedaco.empty()) {
            // REDE DE SEGURANÇA: marcadores de chat nunca entram na resposta
            if (pedaco.find("<|im_start|") != std::string::npos ||
                pedaco.find("<|im_end|>") != std::string::npos ||
                pedaco.find("<|endoftext|>") != std::string::npos) {
                break;
            }
            resultado.append(pedaco);
        }

        batch = llama_batch_get_one(&novo, 1);
    }

    llama_sampler_free(smpl);

    aparar_utf8_incompleto(resultado);

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

#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>
#include <random>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <utility>

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

// ============================================================
// EMBEDDER (classificador de TAGs) — funções auxiliares puras
// ============================================================
// ==== PURE-BEGIN
static inline float half_para_float(uint16_t h) {
    uint32_t sinal = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sinal;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FFu;
            f = sinal | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sinal | 0x7F800000u | (mant << 13);
    } else {
        f = sinal | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float r;
    std::memcpy(&r, &f, sizeof(r));
    return r;
}

// Lê um ficheiro .npy (float16, C-order, 2D) para float32.
static bool carregar_npy_f16(const char *caminho, std::vector<float> &out,
                             int &n, int &d, std::string &erro) {
    FILE *f = std::fopen(caminho, "rb");
    if (!f) { erro = "nao consegui abrir o .npy"; return false; }

    unsigned char magic[6];
    unsigned char ver[2];
    if (std::fread(magic, 1, 6, f) != 6 || std::memcmp(magic, "\x93NUMPY", 6) != 0 ||
        std::fread(ver, 1, 2, f) != 2) {
        std::fclose(f); erro = "cabecalho .npy invalido"; return false;
    }

    uint32_t hlen = 0;
    if (ver[0] == 1) {
        unsigned char b[2];
        if (std::fread(b, 1, 2, f) != 2) { std::fclose(f); erro = "cabecalho .npy curto"; return false; }
        hlen = (uint32_t)b[0] | ((uint32_t)b[1] << 8);
    } else {
        unsigned char b[4];
        if (std::fread(b, 1, 4, f) != 4) { std::fclose(f); erro = "cabecalho .npy curto"; return false; }
        hlen = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    }
    if (hlen == 0 || hlen > 65536) { std::fclose(f); erro = "tamanho de cabecalho invalido"; return false; }

    std::string h(hlen, '\0');
    if (std::fread(&h[0], 1, hlen, f) != hlen) { std::fclose(f); erro = "cabecalho .npy truncado"; return false; }

    if (h.find("<f2") == std::string::npos) { std::fclose(f); erro = "o .npy nao e float16 (<f2)"; return false; }
    if (h.find("'fortran_order': False") == std::string::npos &&
        h.find("\"fortran_order\": false") == std::string::npos) {
        std::fclose(f); erro = "o .npy nao e C-order"; return false;
    }
    size_t p = h.find("shape");
    p = (p == std::string::npos) ? p : h.find('(', p);
    int a = 0, b2 = 0;
    if (p == std::string::npos || std::sscanf(h.c_str() + p, "(%d, %d", &a, &b2) != 2 || a <= 0 || b2 <= 0) {
        std::fclose(f); erro = "shape do .npy invalido"; return false;
    }

    const size_t total = (size_t)a * (size_t)b2;
    std::vector<uint16_t> bruto(total);
    size_t lidos = std::fread(bruto.data(), sizeof(uint16_t), total, f);
    std::fclose(f);
    if (lidos != total) { erro = "dados do .npy truncados"; return false; }

    out.resize(total);
    for (size_t i = 0; i < total; i++) out[i] = half_para_float(bruto[i]);
    n = a; d = b2;
    return true;
}

static void normalizar_l2(std::vector<float> &v) {
    double s = 0.0;
    for (float x : v) s += (double)x * (double)x;
    const float inv = s > 0.0 ? (float)(1.0 / std::sqrt(s)) : 0.0f;
    for (float &x : v) x *= inv;
}

// Devolve os k melhores (score, indice) por produto escalar, do maior para o menor.
static void topk_produto_escalar(const std::vector<float> &C, int n, int d, const float *q,
                                 int k, std::vector<std::pair<float, int>> &saida) {
    std::vector<float> sc((size_t)n);
    for (int i = 0; i < n; i++) {
        const float *row = C.data() + (size_t)i * (size_t)d;
        float s = 0.0f;
        for (int j = 0; j < d; j++) s += row[j] * q[j];
        sc[(size_t)i] = s;
    }
    std::vector<int> idx((size_t)n);
    for (int i = 0; i < n; i++) idx[(size_t)i] = i;
    if (k > n) k = n;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&sc](int x, int y) { return sc[(size_t)x] > sc[(size_t)y]; });
    saida.clear();
    for (int i = 0; i < k; i++) saida.emplace_back(sc[(size_t)idx[(size_t)i]], idx[(size_t)i]);
}
// ==== PURE-END

struct MatolaEmbedContext {
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    const llama_vocab *vocab = nullptr;
    int n_embd = 0;
    int n_grupos = 0;
    std::vector<float> centroides; // n_grupos x n_embd, já normalizados
};

// O modelo foi treinado com max_seq_length=64 e o prefixo "query: " nas perguntas.
static const int EMBED_MAX_TOKENS = 64;
static const char *EMBED_PREFIXO = "query: ";

// Texto -> vetor L2-normalizado (n_embd floats). Pooling CLS, atenção não causal.
static bool embedir_texto(MatolaEmbedContext *ec, const std::string &texto, std::vector<float> &out) {
    const std::string prompt = std::string(EMBED_PREFIXO) + texto;

    std::vector<llama_token> tokens;
    int n = -llama_tokenize(ec->vocab, prompt.c_str(), (int32_t)prompt.size(), nullptr, 0, true, false);
    if (n <= 0) { LOGE("Embed: tokenizacao vazia."); return false; }
    tokens.resize((size_t)n);
    n = llama_tokenize(ec->vocab, prompt.c_str(), (int32_t)prompt.size(), tokens.data(), (int32_t)tokens.size(), true, false);
    if (n <= 0) { LOGE("Embed: falha a tokenizar."); return false; }
    if (n > EMBED_MAX_TOKENS) n = EMBED_MAX_TOKENS;

    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        batch.token[i] = tokens[(size_t)i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1; // pooling precisa de saída em todos os tokens
    }
    batch.n_tokens = n;

    auto mem = llama_get_memory(ec->ctx);
    if (mem) llama_memory_clear(mem, true);

    // llama_decode também serve para modelos de embedding (é o que o exemplo
    // oficial do llama.cpp faz).
    const int rc = llama_decode(ec->ctx, batch);
    if (rc != 0) {
        LOGE("Embed: llama_decode falhou (rc=%d).", rc);
        llama_batch_free(batch);
        return false;
    }

    const float *e = llama_get_embeddings_seq(ec->ctx, 0);
    if (e == nullptr) {
        LOGE("Embed: sem embedding da sequencia (pooling nao activo?).");
        llama_batch_free(batch);
        return false;
    }
    out.assign(e, e + ec->n_embd);
    llama_batch_free(batch);
    normalizar_l2(out);
    return true;
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

    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(64, 1.05f, 0.0f, 0.0f));
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

// =========================================================
// ROTEADOR — escolher UMA opção (0..nOpcoes) com um único decode.
// Sem loop de geração e sem sampling: lê os logits do PRIMEIRO token que o
// modelo escreveria depois de "<|im_start|>assistant\n" e compara só os
// dígitos "0".."nOpcoes". Devolve float[nOpcoes + 2]:
//   [0]     = massa que o modelo dá a ESTES dígitos no vocabulário inteiro
//             (se for baixa, o modelo queria dizer outra coisa)
//   [1 + i] = probabilidade da opção i (i = 0 é "nenhuma"), normalizada só
//             entre os dígitos
// Devolve null em caso de erro. NÃO altera nativeCompletion.
// =========================================================
JNIEXPORT jfloatArray JNICALL
Java_com_example_matolaia_apk_MatolaLlama_nativeEscolher(
        JNIEnv *env, jobject /* this */,
        jlong handle, jstring promptJ, jint nOpcoes) {

    auto *mc = reinterpret_cast<MatolaContext *>(handle);

    if (mc == nullptr || mc->ctx == nullptr) {
        return nullptr;
    }

    int n = nOpcoes;
    if (n < 1) n = 1;
    if (n > 9) n = 9; // só dígitos de 1 token

    const char *promptChars = env->GetStringUTFChars(promptJ, nullptr);
    std::string pergunta(promptChars);
    env->ReleaseStringUTFChars(promptJ, promptChars);

    // Cada chamada começa do zero (igual ao nativeCompletion).
    llama_memory_clear(llama_get_memory(mc->ctx), true);

    // Sem system prompt: a identidade nunca entra na classificação.
    const std::string prompt =
            std::string("<|im_start|>user\n") + pergunta +
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

    const int n_ctx = (int) llama_n_ctx(mc->ctx);
    if (tokens.empty() || (int) tokens.size() >= n_ctx - 16) {
        LOGE("Roteador: prompt invalido (%zu tokens, n_ctx=%d).", tokens.size(), n_ctx);
        return nullptr;
    }

    // Token de cada dígito "0".."n" (primeiro token da tokenização do dígito).
    std::vector<llama_token> ids((size_t) n + 1);
    for (int d = 0; d <= n; d++) {
        const std::string s = std::to_string(d);
        llama_token t[8];
        int k = llama_tokenize(mc->vocab, s.c_str(), (int32_t)s.size(), t, 8, false, false);
        if (k <= 0) {
            LOGE("Roteador: nao consegui tokenizar o digito %d.", d);
            return nullptr;
        }
        if (k != 1) {
            LOGI("Roteador: digito %d tokenizou em %d tokens (uso o primeiro).", d, k);
        }
        ids[(size_t) d] = t[0];
    }

    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
    int rc = llama_decode(mc->ctx, batch);
    if (rc != 0) {
        LOGE("Roteador: llama_decode falhou (rc=%d).", rc);
        return nullptr;
    }

    const float *logits = llama_get_logits_ith(mc->ctx, -1);
    if (logits == nullptr) {
        LOGE("Roteador: sem logits.");
        return nullptr;
    }

    const int n_vocab = llama_vocab_n_tokens(mc->vocab);

    // softmax estável sobre o vocabulário inteiro (só para a "massa")
    float maxl = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > maxl) maxl = logits[i];
    }
    double somaVocab = 0.0;
    for (int i = 0; i < n_vocab; i++) {
        somaVocab += std::exp((double)(logits[i] - maxl));
    }

    std::vector<double> e((size_t) n + 1);
    double somaDigitos = 0.0;
    for (int d = 0; d <= n; d++) {
        e[(size_t) d] = std::exp((double)(logits[ids[(size_t) d]] - maxl));
        somaDigitos += e[(size_t) d];
    }

    std::vector<float> saida((size_t) n + 2);
    saida[0] = (float)(somaVocab > 0.0 ? somaDigitos / somaVocab : 0.0);
    for (int d = 0; d <= n; d++) {
        saida[(size_t) d + 1] = (float)(somaDigitos > 0.0 ? e[(size_t) d] / somaDigitos : 0.0);
    }

    jfloatArray arr = env->NewFloatArray((jsize) saida.size());
    if (arr == nullptr) return nullptr;
    env->SetFloatArrayRegion(arr, 0, (jsize) saida.size(), saida.data());
    return arr;
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


// =========================================================
// EMBEDDER — classe Java MatolaEmbed
// Classificador de TAGs: texto -> vetor (LFM2.5-Embedding-350M afinado)
// -> produto escalar contra os centróides dos grupos.
// Contexto e modelo SEPARADOS do gerador (MatolaLlama). Não mexe em
// nativeCompletion / nativeEscolher.
// =========================================================

// Última razão de falha do nativeInit — sem Logcat à mão, é isto que
// chega ao JS via nativeUltimoErro() (ver EmbedJavascriptBridge.ultimoErro()).
static std::string g_ultimoErroEmbed;

JNIEXPORT jlong JNICALL
Java_com_example_matolaia_apk_MatolaEmbed_nativeInit(
        JNIEnv *env, jobject /* this */,
        jstring modelPath, jstring centroidesPath, jint nThreads) {

    static bool backend_ok = false;
    if (!backend_ok) { llama_backend_init(); backend_ok = true; }

    const char *mp = env->GetStringUTFChars(modelPath, nullptr);
    const char *cp_ = env->GetStringUTFChars(centroidesPath, nullptr);
    const std::string caminhoModelo(mp);
    const std::string caminhoCentroides(cp_);
    env->ReleaseStringUTFChars(modelPath, mp);
    env->ReleaseStringUTFChars(centroidesPath, cp_);

    auto *ec = new MatolaEmbedContext();

    std::string erro;
    if (!carregar_npy_f16(caminhoCentroides.c_str(), ec->centroides, ec->n_grupos, ec->n_embd, erro)) {
        g_ultimoErroEmbed = "centroides (" + caminhoCentroides + "): " + erro;
        LOGE("Embed: %s", g_ultimoErroEmbed.c_str());
        delete ec;
        return 0;
    }

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    ec->model = llama_model_load_from_file(caminhoModelo.c_str(), model_params);
    if (ec->model == nullptr) {
        g_ultimoErroEmbed = "falha ao carregar o modelo (" + caminhoModelo + ") — ficheiro em falta, corrompido ou arquitectura nao suportada nesta build do llama.cpp";
        LOGE("Embed: %s", g_ultimoErroEmbed.c_str());
        delete ec;
        return 0;
    }

    if (llama_model_n_embd(ec->model) != ec->n_embd) {
        g_ultimoErroEmbed = "dimensao do modelo (" + std::to_string(llama_model_n_embd(ec->model)) +
                             ") != centroides (" + std::to_string(ec->n_embd) + ")";
        LOGE("Embed: %s", g_ultimoErroEmbed.c_str());
        llama_model_free(ec->model);
        delete ec;
        return 0;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 512;
    ctx_params.n_ubatch = 512;
    ctx_params.n_seq_max = 1;
    ctx_params.embeddings = true;
    ctx_params.pooling_type = LLAMA_POOLING_TYPE_CLS;
    ctx_params.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL;
    ctx_params.n_threads = nThreads > 0 ? nThreads : 4;
    ctx_params.n_threads_batch = ctx_params.n_threads;

    ec->ctx = llama_init_from_model(ec->model, ctx_params);
    if (ec->ctx == nullptr) {
        g_ultimoErroEmbed = "falha ao criar o contexto (llama_init_from_model devolveu null)";
        LOGE("Embed: %s", g_ultimoErroEmbed.c_str());
        llama_model_free(ec->model);
        delete ec;
        return 0;
    }
    ec->vocab = llama_model_get_vocab(ec->model);

    g_ultimoErroEmbed.clear();
    LOGI("Embedder pronto: %d grupos x %d dims.", ec->n_grupos, ec->n_embd);
    return reinterpret_cast<jlong>(ec);
}

// Devolve a razão da última falha do nativeInit (ou "" se não houve/ficou pronto).
JNIEXPORT jstring JNICALL
Java_com_example_matolaia_apk_MatolaEmbed_nativeUltimoErro(
        JNIEnv *env, jobject /* this */) {
    return env->NewStringUTF(g_ultimoErroEmbed.c_str());
}

// Devolve float[2k]: [indice0, score0, indice1, score1, ...] (do melhor para o pior).
// O índice é a linha do centroides_v1.npy = posição em grupos_v1.json["grupos"].
JNIEXPORT jfloatArray JNICALL
Java_com_example_matolaia_apk_MatolaEmbed_nativeClassificar(
        JNIEnv *env, jobject /* this */,
        jlong handle, jstring textoJ, jint k) {

    auto *ec = reinterpret_cast<MatolaEmbedContext *>(handle);
    if (ec == nullptr || ec->ctx == nullptr) return nullptr;

    int kk = k < 1 ? 1 : (k > 50 ? 50 : k);

    const char *tc = env->GetStringUTFChars(textoJ, nullptr);
    const std::string texto(tc);
    env->ReleaseStringUTFChars(textoJ, tc);

    const auto t0 = std::chrono::steady_clock::now();

    std::vector<float> q;
    if (!embedir_texto(ec, texto, q)) return nullptr;

    const auto t1 = std::chrono::steady_clock::now();

    std::vector<std::pair<float, int>> top;
    topk_produto_escalar(ec->centroides, ec->n_grupos, ec->n_embd, q.data(), kk, top);

    const auto t2 = std::chrono::steady_clock::now();
    LOGI("Embed: modelo %lld ms | busca %lld ms",
         (long long) std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count(),
         (long long) std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());

    std::vector<float> saida;
    for (const auto &par : top) {
        saida.push_back((float) par.second);
        saida.push_back(par.first);
    }

    jfloatArray arr = env->NewFloatArray((jsize) saida.size());
    if (arr == nullptr) return nullptr;
    env->SetFloatArrayRegion(arr, 0, (jsize) saida.size(), saida.data());
    return arr;
}

// Devolve o vetor normalizado (n_embd floats) — serve para conferir o app
// contra o referencia_v1.json do Colab (cosseno deve ficar acima de ~0.99).
JNIEXPORT jfloatArray JNICALL
Java_com_example_matolaia_apk_MatolaEmbed_nativeEmbed(
        JNIEnv *env, jobject /* this */,
        jlong handle, jstring textoJ) {

    auto *ec = reinterpret_cast<MatolaEmbedContext *>(handle);
    if (ec == nullptr || ec->ctx == nullptr) return nullptr;

    const char *tc = env->GetStringUTFChars(textoJ, nullptr);
    const std::string texto(tc);
    env->ReleaseStringUTFChars(textoJ, tc);

    std::vector<float> q;
    if (!embedir_texto(ec, texto, q)) return nullptr;

    jfloatArray arr = env->NewFloatArray((jsize) q.size());
    if (arr == nullptr) return nullptr;
    env->SetFloatArrayRegion(arr, 0, (jsize) q.size(), q.data());
    return arr;
}

JNIEXPORT void JNICALL
Java_com_example_matolaia_apk_MatolaEmbed_nativeFree(
        JNIEnv * /* env */, jobject /* this */, jlong handle) {

    auto *ec = reinterpret_cast<MatolaEmbedContext *>(handle);
    if (ec != nullptr) {
        if (ec->ctx) llama_free(ec->ctx);
        if (ec->model) llama_model_free(ec->model);
        delete ec;
    }
}

} // extern "C"

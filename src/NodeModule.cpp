#include <napi.h>
#include <ggml.h>
#include <stable-diffusion.h>

namespace
{

    struct callJsLogArgs
    {
        sd_log_level_t level = SD_LOG_DEBUG;
        const char* text = nullptr;
    };

    void callJsLog(Napi::Env env, Napi::Function callback, std::nullptr_t*, callJsLogArgs* data)
    {
        Napi::HandleScope hs(env);
        Napi::String logLevel;
        switch (data->level)
        {
            case SD_LOG_ERROR: logLevel = Napi::String::From(env, "error"); break;
            case SD_LOG_WARN: logLevel = Napi::String::From(env, "warn"); break;
            case SD_LOG_INFO: logLevel = Napi::String::From(env, "info"); break;
            case SD_LOG_DEBUG: logLevel = Napi::String::From(env, "debug"); break;
        }
        callback.Call({ logLevel, Napi::String::From(env, data->text) });
    }

    struct callJsProgressArgs
    {
        int step = 0;
        int steps = 0;
        float time = 0;
    };

    void callJsProgress(Napi::Env env, Napi::Function callback, std::nullptr_t*, callJsProgressArgs* data)
    {
        Napi::HandleScope hs(env);
        callback.Call({ Napi::Number::From(env, data->step), Napi::Number::From(env, data->steps) , Napi::Number::From(env, data->time) });
    }


    struct CPPContextData
    {
        std::unique_ptr<sd_ctx_t, decltype(free_sd_ctx)*> sdCtx;
        Napi::TypedThreadSafeFunction<std::nullptr_t, callJsLogArgs, callJsLog> logCallback;
        Napi::TypedThreadSafeFunction<std::nullptr_t, callJsProgressArgs, callJsProgress> progressCallback;
    };

    constinit thread_local CPPContextData* tl_current = nullptr;

    void stableDiffusionLogFunc(enum sd_log_level_t level, const char* text, void* data)
    {
        const auto ctx = tl_current;
        if (ctx && ctx->logCallback)
        {
            callJsLogArgs args = { .level = level, .text = text };
            ctx->logCallback.BlockingCall(&args);
        }
    }

    void stableDiffusionProgressFunc(int step, int steps, float time, void* data)
    {
        const auto ctx = tl_current;
        if (ctx && ctx->progressCallback)
        {
            callJsProgressArgs args = { .step = step, .steps = steps, .time = time };
            ctx->progressCallback.BlockingCall(&args);
        }
    }

    class NodeStableDiffusionCpp : public Napi::Addon<NodeStableDiffusionCpp>
    {
    public:
        NodeStableDiffusionCpp(Napi::Env env, Napi::Object exports)
        {
            sd_set_log_callback(&stableDiffusionLogFunc, nullptr);
            sd_set_progress_callback(&stableDiffusionProgressFunc, nullptr);

            DefineAddon(exports,
                {
                    InstanceMethod("createContext", &NodeStableDiffusionCpp::createContext),
                    InstanceMethod("getSystemInfo", &NodeStableDiffusionCpp::getSystemInfo),
                    InstanceMethod("getNumPhysicalCores", &NodeStableDiffusionCpp::getNumPhysicalCores),
                    InstanceMethod("weightTypeName", &NodeStableDiffusionCpp::weightTypeName),
                });
        }
    protected:
        Napi::Value createContext(const Napi::CallbackInfo& info)
        {
            Napi::Value tmp;
            const auto params = info[0].ToObject();
            const auto model = params.Get("model").ToString().Utf8Value();
            const auto vae = (tmp = params.Get("vae"), tmp.IsUndefined() ? "" : tmp.ToString().Utf8Value());
            const auto taesd = (tmp = params.Get("taesd"), tmp.IsUndefined() ? "" : tmp.ToString().Utf8Value());
            const auto controlNet = (tmp = params.Get("controlNet"), tmp.IsUndefined() ? "" : tmp.ToString().Utf8Value());
            const auto loraDir = (tmp = params.Get("loraDir"), tmp.IsUndefined() ? "" : tmp.ToString().Utf8Value());
            const auto embedDir = (tmp = params.Get("embedDir"), tmp.IsUndefined() ? "" : tmp.ToString().Utf8Value());
            const auto stackedIdEmbedDir = (tmp = params.Get("stackedIdEmbedDir"), tmp.IsUndefined() ? "" : tmp.ToString().Utf8Value());
            const auto vaeDecodeOnly = (tmp = params.Get("vaeDecodeOnly"), tmp.IsUndefined() ? false : tmp.ToBoolean().Value());
            const auto vaeTiling = (tmp = params.Get("vaeTiling"), tmp.IsUndefined() ? false : tmp.ToBoolean().Value());
            const auto freeParamsImmediately = (tmp = params.Get("freeParamsImmediately"), tmp.IsUndefined() ? false : tmp.ToBoolean().Value());
            const auto numThreads = (tmp = params.Get("numThreads"), tmp.IsUndefined() ? GGML_DEFAULT_N_THREADS : tmp.ToNumber().Int32Value());
            const auto weightType = (tmp = params.Get("weightType"), tmp.IsUndefined() ? SD_TYPE_F32 : sd_type_t(tmp.ToNumber().Uint32Value()));
            const auto cudaRng = (tmp = params.Get("cudaRng"), tmp.IsUndefined() ? false : tmp.ToBoolean().Value());
            const auto schedule = (tmp = params.Get("schedule"), tmp.IsUndefined() ? DEFAULT : schedule_t(tmp.ToNumber().Uint32Value()));
            const auto keepClipOnCpu = (tmp = params.Get("keepClipOnCpu"), tmp.IsUndefined() ? false : tmp.ToBoolean().Value());
            const auto keepControlNetOnCpu = (tmp = params.Get("keepControlNetOnCpu"), tmp.IsUndefined() ? false : tmp.ToBoolean().Value());
            const auto keepVaeOnCpu = (tmp = params.Get("keepVaeOnCpu"), tmp.IsUndefined() ? false : tmp.ToBoolean().Value());

            if (weightType >= SD_TYPE_COUNT)
                throw Napi::Error::New(info.Env(), "Invalid weightType");

            if (schedule >= N_SCHEDULES)
                throw Napi::Error::New(info.Env(), "Invalid schedule");


            auto sdCtx = new_sd_ctx(model.c_str(), vae.c_str(), taesd.c_str(), controlNet.c_str(), loraDir.c_str(),
                embedDir.c_str(), stackedIdEmbedDir.c_str(), vaeDecodeOnly, vaeTiling, freeParamsImmediately, numThreads, weightType,
                cudaRng ? CUDA_RNG : STD_DEFAULT_RNG, schedule, keepClipOnCpu, keepControlNetOnCpu, keepVaeOnCpu);

            if (!sdCtx)
                throw Napi::Error::New(info.Env(), "Context creation failed");

            auto ctx = Napi::Object::New(info.Env());
            auto cppContextData = std::make_shared<CPPContextData>(CPPContextData{
                .sdCtx = { sdCtx, &free_sd_ctx}
            });

            ctx.DefineProperties({
                    Napi::PropertyDescriptor::Function("dispose", [cppContextData](const Napi::CallbackInfo& info) { cppContextData->sdCtx.reset(); }),
                    Napi::PropertyDescriptor::Function("setLogCallback", [cppContextData](const Napi::CallbackInfo& info) 
                    {
                        Napi::Function::CheckCast(info.Env(), info[0]);
                        cppContextData->logCallback = decltype(CPPContextData::logCallback)::New(info.Env(), info[0].As<Napi::Function>(), "node-stable-diffusion-cpp-log-callback", 4, 1);
                    }),
                    Napi::PropertyDescriptor::Function("setProgressCallback", [cppContextData](const Napi::CallbackInfo& info) 
                    {
                        Napi::Function::CheckCast(info.Env(), info[0]);
                        cppContextData->progressCallback = decltype(CPPContextData::progressCallback)::New(info.Env(), info[0].As<Napi::Function>(), "node-stable-diffusion-cpp-progress-callback", 4, 1);
                    }),
                });

            return ctx;
        }

        Napi::Value getSystemInfo(const Napi::CallbackInfo& info)
        {
            return Napi::String::New(info.Env(), sd_get_system_info());
        }

        Napi::Value getNumPhysicalCores(const Napi::CallbackInfo& info)
        {
            return Napi::Number::New(info.Env(), get_num_physical_cores());
        }

        Napi::Value weightTypeName(const Napi::CallbackInfo& info)
        {
            const auto weightType = sd_type_t(info[0].ToNumber().Uint32Value());
            if (weightType >= SD_TYPE_COUNT)
                throw Napi::Error::New(info.Env(), "Invalid weightType");

            return Napi::String::New(info.Env(), sd_type_name(weightType));
        }
    };
}

NODE_API_ADDON(NodeStableDiffusionCpp)

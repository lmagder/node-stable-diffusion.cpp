#include <napi.h>
#include <ggml.h>
#include <stable-diffusion.h>

class NodeStableDiffusionCpp : public Napi::Addon<NodeStableDiffusionCpp> 
{
 public:
  NodeStableDiffusionCpp(Napi::Env env, Napi::Object exports) 
  {
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
        
        struct CPPContextData
        {
            std::unique_ptr<sd_ctx_t, decltype(free_sd_ctx)*> sdCtx;
        };

        auto ctx = Napi::Object::New(info.Env());
        
        auto cppContextData = new CPPContextData{
            .sdCtx = { sdCtx, &free_sd_ctx}
        };

        ctx.DefineProperties({
                Napi::PropertyDescriptor::Function("dispose", [cppContextData](const Napi::CallbackInfo& info) { cppContextData->sdCtx.reset(); }),
        });
        ctx.AddFinalizer([](Napi::Env env, CPPContextData* data) { delete data; }, cppContextData);

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

NODE_API_ADDON(NodeStableDiffusionCpp)

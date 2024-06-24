#include <optional>
#include <type_traits>

#include <napi.h>
#include <ggml.h>
#include <stable-diffusion.h>

namespace
{

    struct callJsLogArgs
    {
        sd_log_level_t level = SD_LOG_DEBUG;
        std::string text;
    };

    void callJsLog(Napi::Env env, Napi::Function callback, std::nullptr_t*, callJsLogArgs* data)
    {
        auto dataPtr = std::unique_ptr<callJsLogArgs>(data);
        Napi::HandleScope hs(env);
        Napi::String logLevel;
        switch (dataPtr->level)
        {
            case SD_LOG_ERROR: logLevel = Napi::String::From(env, "error"); break;
            case SD_LOG_WARN: logLevel = Napi::String::From(env, "warn"); break;
            case SD_LOG_INFO: logLevel = Napi::String::From(env, "info"); break;
            case SD_LOG_DEBUG:
            default: logLevel = Napi::String::From(env, "debug"); break;
        }
        dataPtr->text.erase(dataPtr->text.find_last_not_of("\n\r") + 1);
        auto text = Napi::String::From(env, dataPtr->text);
        callback.Call({ logLevel, text });
    }

    struct callJsProgressArgs
    {
        int step = 0;
        int steps = 0;
        float time = 0;
    };

    void callJsProgress(Napi::Env env, Napi::Function callback, std::nullptr_t*, callJsProgressArgs* data)
    {
        auto dataPtr = std::unique_ptr<callJsProgressArgs>(data);
        Napi::HandleScope hs(env);
        callback.Call({ Napi::Number::From(env, dataPtr->step), Napi::Number::From(env, dataPtr->steps) , Napi::Number::From(env, dataPtr->time) });
    }

    void stableDiffusionLogFunc(enum sd_log_level_t level, const char* text, void* data);

    struct CPPContextData : public std::enable_shared_from_this<CPPContextData>
    {
        std::shared_ptr<sd_ctx_t> sdCtx;
        Napi::TypedThreadSafeFunction<std::nullptr_t, callJsLogArgs, callJsLog> logCallback;
        Napi::TypedThreadSafeFunction<std::nullptr_t, callJsProgressArgs, callJsProgress> progressCallback;
        std::vector<std::unique_ptr<Napi::AsyncWorker>> pendingTasks;

        CPPContextData() = default;
        CPPContextData(const CPPContextData& ctx) = delete;
        CPPContextData(CPPContextData&& ctx) = delete;
        CPPContextData& operator=(const CPPContextData& ctx) = delete;
        CPPContextData& operator=(CPPContextData&& ctx) = delete;

        ~CPPContextData()
        {
            sdCtx.reset();

            if (progressCallback)
                progressCallback.Release();

            if (logCallback)
            {
                stableDiffusionLogFunc(SD_LOG_DEBUG, "Closing node context", nullptr);
                logCallback.Release();
            }
        }

        void nextTask()
        {
            if (!pendingTasks.empty())
            {
                auto begin = pendingTasks.begin();
                begin->release()->Queue();
                pendingTasks.erase(begin);
            }
        }
    };

    constinit thread_local CPPContextData* tl_current = nullptr;

    void stableDiffusionLogFunc(enum sd_log_level_t level, const char* text, void* data)
    {
        const auto ctx = tl_current;
        if (ctx && ctx->logCallback)
        {
            ctx->logCallback.BlockingCall(new callJsLogArgs{ .level = level, .text = text });
        }
    }

    void stableDiffusionProgressFunc(int step, int steps, float time, void* data)
    {
        const auto ctx = tl_current;
        if (ctx && ctx->progressCallback)
        {
            ctx->progressCallback.BlockingCall(new callJsProgressArgs{ .step = step, .steps = steps, .time = time });
        }
    }


    class freeSdImageList
    {
        size_t imageCount;
    public:
        freeSdImageList(size_t imageCount) noexcept : imageCount(imageCount) {};


        void operator()(sd_image_t* ptr) const
        {
            if (ptr)
            {
                for (size_t i = 0; i < imageCount; i++)
                {
                    free(ptr[i].data);
                }

                free(ptr);
            }
        }

    };

    class freeSdImage
    {
    public:
        void operator()(sd_image_t* ptr) const
        {
            if (ptr)
            {
                free(ptr->data);
                free(ptr);
            }
        }

    };

    using SdImageList = std::unique_ptr<sd_image_t[], freeSdImageList>;
    using SdImage = std::unique_ptr<sd_image_t, freeSdImage>;

    Napi::Object wrapSdImage(Napi::Env env, const sd_image_t& img)
    {
        auto imgObj = Napi::Object::New(env);
        imgObj.DefineProperties({
                Napi::PropertyDescriptor::Value("width",  Napi::Number::From(env, img.width)),
                Napi::PropertyDescriptor::Value("height",  Napi::Number::From(env, img.height)),
                Napi::PropertyDescriptor::Value("channel",  Napi::Number::From(env, img.channel)),
                Napi::PropertyDescriptor::Value("data",  Napi::Buffer<uint8_t>::Copy(env, img.data, size_t(img.width) * img.height * img.channel))
            });

        imgObj.Freeze();
        return imgObj;
    }

    SdImage extractSdImage(Napi::Object imgObj)
    {
        const auto width = imgObj.Get("width").ToNumber().Int32Value();
        const auto height = imgObj.Get("height").ToNumber().Int32Value();
        const auto channel = imgObj.Get("channel").ToNumber().Int32Value();
        Napi::Buffer<uint8_t>::CheckCast(imgObj.Env(), imgObj.Get("data"));
        const auto data = imgObj.Get("data").As<Napi::Buffer<uint8_t>>();

        if (width <= 0 || height <= 0 || channel <= 0)
        {
            throw Napi::Error::New(imgObj.Env(), "Invalid size");
        }

        const size_t expectedSize = size_t(width) * height * channel;
        if (expectedSize != data.Length())
        {
            throw Napi::Error::New(imgObj.Env(), "Invalid size");
        }

        auto img = (sd_image_t*)calloc(1, sizeof(sd_image_t));
        img->width = width;
        img->height = height;
        img->channel = channel;
        img->data = (uint8_t*)malloc(expectedSize);
        memcpy(img->data, data.Data(), data.Length());

        return SdImage(img);
    }


    template <typename T, typename C>
    Napi::Promise queueStableDiffusionWorker(Napi::Env env, const std::shared_ptr<CPPContextData>& ctx, T&& func, C&& convFunc)
    {
        class StableDiffusionWorker : public Napi::AsyncWorker
        {
            //copy this on purpose to snapshot it
            std::shared_ptr<CPPContextData> ctx;
            Napi::Promise::Deferred def;
            std::decay_t<T> func;
            std::decay_t<C> convFunc;
            std::optional<std::invoke_result_t<decltype(func), CPPContextData&>> result;
        public:
            StableDiffusionWorker(Napi::Env env, const std::shared_ptr<CPPContextData>& ctx, T&& func, C&& convFunc) : Napi::AsyncWorker(env, "node-stable-diffusion-cpp-worker"),
                ctx(ctx), def(env), func(std::forward<T>(func)), convFunc(std::forward<C>(convFunc))
            {
            }

            void Execute() override
            {
                auto prev = std::exchange(tl_current, ctx.get());
                result.emplace(func(*ctx));
                tl_current = prev;
            }

            void OnOK() override
            {
                def.Resolve(convFunc(Env(), std::move(result).value()));
                ctx->nextTask();
            }

            void OnError(const Napi::Error& e) override
            {
                def.Reject(e.Value());
                ctx->nextTask();
            }

            Napi::Promise Promise() const { return def.Promise(); }

        };

        const bool isFirst = ctx->pendingTasks.empty();
        auto worker = std::make_unique<StableDiffusionWorker>(env, ctx, std::forward<T>(func), std::forward<C>(convFunc));
        const auto ret = worker->Promise();
        ctx->pendingTasks.emplace_back(std::move(worker));

        if (isFirst)
            ctx->nextTask();

        return ret;
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

            auto cppContextData = std::make_shared<CPPContextData>();
            if (!info[1].IsUndefined())
            {
                Napi::Function::CheckCast(info.Env(), info[1]);
                cppContextData->logCallback = decltype(CPPContextData::logCallback)::New(info.Env(), info[1].As<Napi::Function>(), "node-stable-diffusion-cpp-log-callback", 1, 1);
            }

            if (!info[2].IsUndefined())
            {
                Napi::Function::CheckCast(info.Env(), info[2]);
                cppContextData->logCallback = decltype(CPPContextData::logCallback)::New(info.Env(), info[2].As<Napi::Function>(), "node-stable-diffusion-cpp-progress-callback", 1, 1);
            }

            return queueStableDiffusionWorker(info.Env(), cppContextData, [=](CPPContextData& ctx)
            {
                ctx.sdCtx = { new_sd_ctx(model.c_str(), vae.c_str(), taesd.c_str(), controlNet.c_str(), loraDir.c_str(),
                    embedDir.c_str(), stackedIdEmbedDir.c_str(), vaeDecodeOnly, vaeTiling, freeParamsImmediately, numThreads, weightType,
                    cudaRng ? CUDA_RNG : STD_DEFAULT_RNG, schedule, keepClipOnCpu, keepControlNetOnCpu, keepVaeOnCpu), [](sd_ctx_t* c) { if (c) free_sd_ctx(c); } };

                if (!ctx.sdCtx)
                    throw std::runtime_error("Context creation failed");

                return ctx.shared_from_this();
            },
            [](Napi::Env env, const std::shared_ptr<CPPContextData>& cppContextData)
            {
                auto ctx = Napi::Object::New(env);
                ctx.DefineProperties({
                    Napi::PropertyDescriptor::Function(env, Napi::Object(), "dispose", [cppContextData](const Napi::CallbackInfo& info)
                    {
                        if (!cppContextData->sdCtx)
                            throw Napi::Error::New(info.Env(), "Context disposed");

                        cppContextData->sdCtx.reset();
                    }),
                    Napi::PropertyDescriptor::Function(env, Napi::Object(), "txt2img", [cppContextData](const Napi::CallbackInfo& info)
                    {
                        if (!cppContextData->sdCtx)
                            throw Napi::Error::New(info.Env(), "Context disposed");

                        Napi::Value tmp;
                        const auto params = info[0].ToObject();
                        const auto prompt = params.Get("prompt").ToString().Utf8Value();
                        const auto negativePrompt = (tmp = params.Get("negativePrompt"), tmp.IsUndefined() ? "" : tmp.ToString().Utf8Value());
                        const auto clipSkip = (tmp = params.Get("clipSkip"), tmp.IsUndefined() ? -1 : tmp.ToNumber().Int32Value());
                        const auto cfgScale = (tmp = params.Get("cfgScale"), tmp.IsUndefined() ? 7.0f : tmp.ToNumber().FloatValue());
                        const auto width = (tmp = params.Get("width"), tmp.IsUndefined() ? 512 : tmp.ToNumber().Int32Value());
                        const auto height = (tmp = params.Get("height"), tmp.IsUndefined() ? 512 : tmp.ToNumber().Int32Value());
                        const auto sampleMethod = (tmp = params.Get("sampleMethod"), tmp.IsUndefined() ? EULER_A : sample_method_t(tmp.ToNumber().Uint32Value()));
                        const auto sampleSteps = (tmp = params.Get("sampleSteps"), tmp.IsUndefined() ? 20 : tmp.ToNumber().Int32Value());
                        const auto seed = (tmp = params.Get("seed"), tmp.IsUndefined() ? 42 : tmp.ToNumber().Int64Value());
                        const auto batchCount = (tmp = params.Get("batchCount"), tmp.IsUndefined() ? 1 : tmp.ToNumber().Int32Value());
                        auto controlCond = (tmp = params.Get("controlCond"), tmp.IsUndefined() ? SdImage() : extractSdImage(tmp.ToObject()));
                        const auto controlStrength = (tmp = params.Get("controlStrength"), tmp.IsUndefined() ? 0.0f : tmp.ToNumber().FloatValue());
                        const auto styleRatio = (tmp = params.Get("styleRatio"), tmp.IsUndefined() ? 0.0f : tmp.ToNumber().FloatValue());
                        const auto normalizeInput = (tmp = params.Get("normalizeInput"), tmp.IsUndefined() ? false : tmp.ToBoolean().Value());
                        const auto inputIdImagesPath = (tmp = params.Get("inputIdImagesPath"), tmp.IsUndefined() ? "" : tmp.ToString().Utf8Value());
                        if (sampleMethod >= N_SAMPLE_METHODS)
                            throw Napi::Error::New(info.Env(), "Invalid sampleMethod");

                        return queueStableDiffusionWorker(info.Env(), cppContextData, [=, controlCond = std::move(controlCond)](CPPContextData& ctx)
                        {
                            auto sdCtx = ctx.sdCtx; //keep it alive while we do this
                            return SdImageList(txt2img(sdCtx.get(), prompt.c_str(), negativePrompt.c_str(), clipSkip, cfgScale, width, height, sampleMethod, sampleSteps, seed, batchCount, controlCond.get(), controlStrength, styleRatio, normalizeInput, inputIdImagesPath.c_str()), batchCount);
                        },
                        [batchCount](Napi::Env env, SdImageList&& images)
                        {
                            auto arr = Napi::Array::New(env, batchCount);
                            for (int b = 0; b < batchCount; b++)
                            {
                                arr[b] = wrapSdImage(env, images[b]);
                            }
                            return arr;
                        });
                    }),
                });
                ctx.Freeze();
                return ctx;
            });
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

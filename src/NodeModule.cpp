#include <napi.h>

class NodeStableDiffusionCpp : public Napi::Addon<NodeStableDiffusionCpp> {
 public:
  NodeStableDiffusionCpp(Napi::Env env, Napi::Object exports) {
  }
 private:
};

NODE_API_ADDON(NodeStableDiffusionCpp)

// simdrng_napi.cpp — Node-API (node-addon-api) native backend over the simdrng
// C ABI. `create(kind, seed)` returns an object owning a simdrng_t handle with
// random()/raw()/free() methods; bulk fills go straight into the TypedArray
// backing store (zero-copy). The 64-bit seed rides in as a JS BigInt.

#include <napi.h>

#include <simdrng/capi.h>

#include <cstdint>
#include <memory>

namespace {

// Owns the handle; freed when the last method closure dies (or free() is called).
struct GenState {
  simdrng_t h;
  explicit GenState(simdrng_t handle) : h(handle) {}
  GenState(const GenState &) = delete;
  GenState &operator=(const GenState &) = delete;
  ~GenState() {
    if (h)
      h = simdrng_free(h);
  }
};

// create(kind:int, seed:bigint) -> { kind, random(n), raw(n), free() }
Napi::Value Create(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  const int kind = info[0].As<Napi::Number>().Int32Value();
  bool lossless = false;
  const auto seed = info[1].As<Napi::BigInt>().Uint64Value(&lossless);

  simdrng_t handle = simdrng_create(static_cast<simdrng_kind>(kind), seed);
  if (!handle) {
    const char *msg = simdrng_last_error();
    throw Napi::Error::New(env, (msg && *msg) ? msg : "simdrng_create returned NULL");
  }
  auto st = std::make_shared<GenState>(handle);

  Napi::Object obj = Napi::Object::New(env);
  obj.Set("kind", Napi::Number::New(env, simdrng_get_kind(st->h)));

  // random(n) -> Float64Array of n uniform [0,1) draws.
  obj.Set("random", Napi::Function::New(env, [st](const Napi::CallbackInfo &info) -> Napi::Value {
            const size_t n = info[0].As<Napi::Number>().Int64Value();
            Napi::Float64Array out = Napi::Float64Array::New(info.Env(), n);
            simdrng_fill_double(st->h, out.Data(), n);
            return out;
          }));

  // raw(n) -> BigUint64Array of n raw 64-bit words.
  obj.Set("raw", Napi::Function::New(env, [st](const Napi::CallbackInfo &info) -> Napi::Value {
            const size_t n = info[0].As<Napi::Number>().Int64Value();
            auto out = Napi::TypedArrayOf<uint64_t>::New(info.Env(), n, napi_biguint64_array);
            simdrng_fill_u64(st->h, out.Data(), n);
            return out;
          }));

  obj.Set("free", Napi::Function::New(env, [st](const Napi::CallbackInfo &info) -> Napi::Value {
            if (st->h)
              st->h = simdrng_free(st->h);
            return info.Env().Undefined();
          }));
  return obj;
}

} // namespace

static Napi::Object Init(Napi::Env env, Napi::Object exports) {
  exports.Set("create", Napi::Function::New(env, Create));
  exports.Set("versionString", Napi::String::New(env, simdrng_version()));
  return exports;
}

NODE_API_MODULE(simdrng, Init)

#include "flowstate/laya.hpp"
#include <curl/curl.h>

namespace flowstate {
namespace {
struct CurlGlobal {
    CurlGlobal() {
        if(curl_global_init(CURL_GLOBAL_DEFAULT)!=CURLE_OK) throw std::runtime_error("Cannot initialize libcurl");
    }
    ~CurlGlobal() { curl_global_cleanup(); }
};
struct Buffer { std::string data; bool too_large=false; };
std::size_t receive(char* data,std::size_t size,std::size_t count,void* opaque) noexcept {
    auto& buffer=*static_cast<Buffer*>(opaque);
    if((size && count>32768/size) || size*count>32768-buffer.data.size()) { buffer.too_large=true; return 0; }
    try { buffer.data.append(data,size*count); return size*count; }
    catch(...) { return 0; }
}
class HttpClient {
public:
    HttpClient(const LayaConfig& config,const std::string& endpoint)
        : handle_(curl_easy_init(),curl_easy_cleanup),headers_(nullptr,curl_slist_free_all) {
        if(!handle_) throw std::runtime_error("Cannot allocate HTTP client");
        const auto* version=curl_version_info(CURLVERSION_NOW);
        if(!(version->features&CURL_VERSION_ASYNCHDNS)) throw std::runtime_error("Laya requires libcurl with asynchronous DNS for bounded timeouts");
        auto header=[&](const std::string& value) {
            auto* list=curl_slist_append(headers_.get(),value.c_str());
            if(!list) throw std::bad_alloc();
            headers_.release(); headers_.reset(list);
        };
        header("Content-Type: application/json");
        set(CURLOPT_URL,endpoint.c_str());
        set(CURLOPT_HTTPHEADER,headers_.get());
        set(CURLOPT_POST,1L); set(CURLOPT_NOSIGNAL,1L);
        set(CURLOPT_TIMEOUT_MS,static_cast<long>(config.timeout.count()));
        set(CURLOPT_CONNECTTIMEOUT_MS,static_cast<long>(config.timeout.count()));
        set(CURLOPT_FOLLOWLOCATION,0L);
        set(CURLOPT_PROTOCOLS_STR,"http");
        set(CURLOPT_PROXY,""); // Local telemetry must never pass through an environment proxy.
        set(CURLOPT_WRITEFUNCTION,&receive);
    }
    LayaResponse post(const std::string& request) {
        Buffer buffer;
        set(CURLOPT_POSTFIELDS,request.c_str());
        set(CURLOPT_POSTFIELDSIZE,static_cast<long>(request.size()));
        set(CURLOPT_WRITEDATA,&buffer);
        const auto code=curl_easy_perform(handle_.get());
        long status=0;
        curl_easy_getinfo(handle_.get(),CURLINFO_RESPONSE_CODE,&status);
        auto error=LayaError::None;
        if(buffer.too_large) error=LayaError::TooLarge;
        else if(code==CURLE_OPERATION_TIMEDOUT) error=LayaError::Timeout;
        else if(code!=CURLE_OK) error=LayaError::Transport;
        else if(status!=200) error=LayaError::Http;
        return {error,status,std::move(buffer.data)};
    }
private:
    template<class T> void set(CURLoption option,T value) {
        if(curl_easy_setopt(handle_.get(),option,value)!=CURLE_OK) throw std::runtime_error("Cannot configure HTTP client");
    }
    std::unique_ptr<CURL,decltype(&curl_easy_cleanup)> handle_;
    std::unique_ptr<curl_slist,decltype(&curl_slist_free_all)> headers_;
};
}
LayaTransport make_laya_transport(const LayaConfig& config,std::string endpoint) {
    config.validate();
    if(endpoint.empty()) endpoint="http://127.0.0.1:"+std::to_string(config.port)+"/v1/systemone";
    {
        constexpr std::string_view prefix="http://127.0.0.1:";
        if(!endpoint.starts_with(prefix)) throw std::invalid_argument("Only numeric loopback HTTP is supported");
        const auto end=endpoint.find('/',prefix.size());
        const auto port=endpoint.substr(prefix.size(),end-prefix.size());
        if(port.empty() || port.size()>5 || port.find_first_not_of("0123456789")!=std::string::npos || std::stoul(port)==0 || std::stoul(port)>65535)
            throw std::invalid_argument("Invalid loopback endpoint");
    }
    static CurlGlobal global;
    auto client=std::make_shared<HttpClient>(config,endpoint);
    return [client](const std::string& body) { return client->post(body); };
}
} // namespace flowstate

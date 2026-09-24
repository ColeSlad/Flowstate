#include "flowstate/workload.hpp"
#include <charconv>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace flowstate {
std::vector<LoadPhase> read_trace(const std::string& path, std::size_t vectors) {
    std::ifstream input(path);
    if(!input) throw std::invalid_argument("Cannot read workload trace: "+path);
    std::string line;
    if(!std::getline(input,line)) throw std::invalid_argument("Empty workload trace");
    if(!line.empty() && line.back()=='\r') line.pop_back();
    if(line!="name,duration_ms,requests_per_second,burst,top_k") throw std::invalid_argument("Unexpected workload CSV header");
    std::vector<LoadPhase> result;
    std::uint64_t duration=0;
    while(std::getline(input,line)) {
        if(!line.empty() && line.back()=='\r') line.pop_back();
        std::istringstream row(line);
        std::vector<std::string> fields;
        std::string field;
        while(std::getline(row,field,',')) fields.push_back(field);
        if(fields.size()!=5 || line.back()==',' || fields[0].empty() || fields[0].find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")!=std::string::npos)
            throw std::invalid_argument("Invalid workload CSV row");
        auto number=[&](std::size_t i) {
            std::uint64_t value=0;
            auto [end,error]=std::from_chars(fields[i].data(),fields[i].data()+fields[i].size(),value);
            if(error!=std::errc{} || end!=fields[i].data()+fields[i].size()) throw std::invalid_argument("Invalid workload number");
            return value;
        };
        LoadPhase phase{fields[0],number(1),number(2),static_cast<std::size_t>(number(3)),static_cast<std::size_t>(number(4))};
        if(!phase.duration_ms || phase.duration_ms>600000 || phase.requests_per_second>1000000 ||
           !phase.burst || phase.burst>100000 || !phase.top_k || phase.top_k>vectors || result.size()>=100)
            throw std::invalid_argument("Workload phase exceeds bounds");
        duration+=phase.duration_ms;
        if(duration>3600000) throw std::invalid_argument("Workload trace exceeds one hour");
        result.push_back(std::move(phase));
    }
    if(result.empty()) throw std::invalid_argument("Workload trace has no phases");
    return result;
}
} // namespace flowstate

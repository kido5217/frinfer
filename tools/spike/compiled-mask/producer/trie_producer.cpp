// Trie mask producer prototype implementation (wayfinder ticket #57).
#include "trie_producer.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <x86intrin.h>
#include <stdexcept>

namespace trie_probe {
namespace {

constexpr int kLookup[16] = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4};

bool is_end_of_sequence(const Element* pos) {
    return pos->type == LLAMA_GRETYPE_END || pos->type == LLAMA_GRETYPE_ALT;
}

bool is_char_type(const Element* pos) {
    switch (pos->type) {
        case LLAMA_GRETYPE_CHAR:
        case LLAMA_GRETYPE_CHAR_NOT:
        case LLAMA_GRETYPE_CHAR_ANY:
            return true;
        default:
            return false;
    }
}

// Replica of llama_grammar_match_char (llama-grammar.cpp).
std::pair<bool, const Element*> match_char(const Element* pos, std::uint32_t chr) {
    bool found = false;
    bool is_positive = pos->type == LLAMA_GRETYPE_CHAR || pos->type == LLAMA_GRETYPE_CHAR_ANY;
    do {
        if (pos[1].type == LLAMA_GRETYPE_CHAR_RNG_UPPER) {
            found = found || (pos->value <= chr && chr <= pos[1].value);
            pos += 2;
        } else if (pos->type == LLAMA_GRETYPE_CHAR_ANY) {
            found = true;
            pos += 1;
        } else {
            found = found || pos->value == chr;
            pos += 1;
        }
    } while (pos->type == LLAMA_GRETYPE_CHAR_ALT);
    return {found == is_positive, pos};
}

// Replica of llama_grammar_match_partial_char (llama-grammar.cpp).
bool match_partial_char(const Element* pos, const llama_partial_utf8 partial) {
    const bool is_positive_char =
        pos->type == LLAMA_GRETYPE_CHAR || pos->type == LLAMA_GRETYPE_CHAR_ANY;
    const std::uint32_t partial_value = partial.value;
    int n_remain = partial.n_remain;
    if (n_remain < 0 || (n_remain == 1 && partial_value < 2)) { return false; }
    std::uint32_t low = partial_value << (n_remain * 6);
    std::uint32_t high = low | ((1u << (n_remain * 6)) - 1);
    if (low == 0) {
        if (n_remain == 2) {
            low = 1 << 11;
        } else if (n_remain == 3) {
            low = 1 << 16;
        }
    }
    do {
        if (pos[1].type == LLAMA_GRETYPE_CHAR_RNG_UPPER) {
            if (pos->value <= high && low <= pos[1].value) { return is_positive_char; }
            pos += 2;
        } else if (pos->type == LLAMA_GRETYPE_CHAR_ANY) {
            return true;
        } else {
            if (low <= pos->value && pos->value <= high) { return is_positive_char; }
            pos += 1;
        }
    } while (pos->type == LLAMA_GRETYPE_CHAR_ALT);
    return !is_positive_char;
}

bool in_ranges(const std::vector<std::pair<std::uint32_t, std::uint32_t>>& ranges,
               std::uint32_t cp) {
    auto it = std::upper_bound(ranges.begin(), ranges.end(), cp,
                               [](std::uint32_t value, const auto& range) {
                                   return value < range.first;
                               });
    if (it == ranges.begin()) { return false; }
    --it;
    return cp <= it->second;
}

std::vector<std::pair<std::uint32_t, std::uint32_t>> ranges_intersect(
    const std::vector<std::pair<std::uint32_t, std::uint32_t>>& a,
    const std::vector<std::pair<std::uint32_t, std::uint32_t>>& b) {
    std::vector<std::pair<std::uint32_t, std::uint32_t>> out;
    std::size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        const std::uint32_t lo = std::max(a[i].first, b[j].first);
        const std::uint32_t hi = std::min(a[i].second, b[j].second);
        if (lo <= hi) { out.push_back({lo, hi}); }
        if (a[i].second < b[j].second) {
            ++i;
        } else {
            ++j;
        }
    }
    return out;
}

// a minus b over the full domain.
std::vector<std::pair<std::uint32_t, std::uint32_t>> ranges_subtract(
    std::vector<std::pair<std::uint32_t, std::uint32_t>> a,
    const std::vector<std::pair<std::uint32_t, std::uint32_t>>& b) {
    std::vector<std::pair<std::uint32_t, std::uint32_t>> out;
    for (const auto& range : a) {
        std::uint32_t start = range.first;
        for (const auto& other : b) {
            if (other.second < start) { continue; }
            if (other.first > range.second) { break; }
            if (other.first > start) { out.push_back({start, other.first - 1}); }
            if (other.second >= range.second) {
                start = range.second + 1;
                break;
            }
            start = other.second + 1;
        }
        if (start <= range.second) { out.push_back({start, range.second}); }
    }
    return out;
}

std::vector<std::pair<std::uint32_t, std::uint32_t>> ranges_union(
    std::vector<std::pair<std::uint32_t, std::uint32_t>> a,
    std::vector<std::pair<std::uint32_t, std::uint32_t>> b) {
    a.insert(a.end(), b.begin(), b.end());
    std::sort(a.begin(), a.end());
    std::vector<std::pair<std::uint32_t, std::uint32_t>> merged;
    for (const auto& range : a) {
        if (!merged.empty() && range.first <= merged.back().second + 1) {
            merged.back().second = std::max(merged.back().second, range.second);
        } else {
            merged.push_back(range);
        }
    }
    return merged;
}

// True when the accepted set of `top` is a superset of `cls`.
// C = cls.negated ? ~cls.r : cls.r ; T = own.negated ? ~own.r : own.r ; test C subset-of T.
bool accepts_all(const Element* top, const CharClass& cls) {
    const CharClass own = class_from_element(top);
    if (!own.negated) {
        if (!cls.negated) { return ranges_subtract(cls.ranges, own.ranges).empty(); }
        // C = ~cls.r: ~cls.r subset-of own.r  <=>  own.r union cls.r covers the universe.
        static const std::vector<std::pair<std::uint32_t, std::uint32_t>> kUniverse = {
            {0, 0x10FFFF}};
        return ranges_subtract(kUniverse, ranges_union(own.ranges, cls.ranges)).empty();
    }
    // T = ~own.r: C subset-of ~own.r  <=>  C intersect own.r is empty.
    if (!cls.negated) { return ranges_intersect(cls.ranges, own.ranges).empty(); }
    // C = ~cls.r: (~cls.r) intersect own.r = own.r minus cls.r
    return ranges_subtract(own.ranges, cls.ranges).empty();
}

std::uint32_t first_accepted(const CharClass& cls) {
    if (!cls.negated) {
        return cls.ranges.empty() ? 0 : cls.ranges.front().first;
    }
    std::uint32_t candidate = 0;
    while (in_ranges(cls.ranges, candidate)) { ++candidate; }
    return candidate;
}

void stk_set(Stk& dst, const std::vector<const Element*>& src) {
    if (src.size() > kMaxStack) {
        throw std::runtime_error("trie producer: stack deeper than kMaxStack");
    }
    dst.n = static_cast<std::uint8_t>(src.size());
    if (!src.empty()) { std::memcpy(dst.e, src.data(), src.size() * sizeof(Element*)); }
}

bool stk_eq(const Stk& a, const Stk& b) {
    return a.n == b.n && std::memcmp(a.e, b.e, a.n * sizeof(Element*)) == 0;
}

void pos_add(PosSet& out, const Stk& stack) {
    for (int i = 0; i < out.n; ++i) {
        if (stk_eq(out.s[i], stack)) { return; }
    }
    if (out.n >= kMaxPos) {
        throw std::runtime_error("trie producer: position set larger than kMaxPos");
    }
    out.s[out.n++] = stack;
}

// Replica of llama_grammar_advance_stack: expands rule refs until every stack
// ends at a char/token element or is empty (tracked via has_empty).
void advance_expand(const llama_grammar_rules& rules, const Stk& start, Succ& out) {
    Stk todo[64];
    int ntodo = 0;
    Stk seen[64];
    int nseen = 0;
    if (start.n > kMaxStack) { throw std::runtime_error("trie producer: stack deeper than kMaxStack"); }
    todo[ntodo++] = start;
    while (ntodo > 0) {
        Stk cur = todo[--ntodo];
        bool already = false;
        for (int i = 0; i < nseen; ++i) {
            if (stk_eq(seen[i], cur)) {
                already = true;
                break;
            }
        }
        if (already) { continue; }
        if (nseen >= 64) { throw std::runtime_error("trie producer: advance seen overflow"); }
        seen[nseen++] = cur;
        if (cur.n == 0) {
            out.has_empty = true;
            continue;
        }
        const Element* pos = cur.e[cur.n - 1];
        if (pos->type == LLAMA_GRETYPE_RULE_REF) {
            const auto& rule = rules[pos->value];
            const Element* subpos = rule.data();
            do {
                Stk next;
                next.n = cur.n - 1;
                std::memcpy(next.e, cur.e, (cur.n - 1) * sizeof(Element*));
                if (!is_end_of_sequence(pos + 1)) { next.e[next.n++] = pos + 1; }
                if (!is_end_of_sequence(subpos)) { next.e[next.n++] = subpos; }
                if (next.n > kMaxStack) {
                    throw std::runtime_error("trie producer: stack deeper than kMaxStack");
                }
                if (ntodo >= 64) { throw std::runtime_error("trie producer: advance todo overflow"); }
                todo[ntodo++] = next;
                while (!is_end_of_sequence(subpos)) { ++subpos; }
                if (subpos->type == LLAMA_GRETYPE_ALT) {
                    ++subpos;
                } else {
                    break;
                }
            } while (true);
        } else if (pos->type == LLAMA_GRETYPE_CHAR || pos->type == LLAMA_GRETYPE_CHAR_NOT ||
                   pos->type == LLAMA_GRETYPE_CHAR_ANY || pos->type == LLAMA_GRETYPE_TOKEN ||
                   pos->type == LLAMA_GRETYPE_TOKEN_NOT) {
            for (int i = 0; i < out.n; ++i) {
                if (stk_eq(out.s[i], cur)) {
                    already = true;
                    break;
                }
            }
            if (!already) {
                if (out.n >= kMaxPos) {
                    throw std::runtime_error("trie producer: expand result larger than kMaxPos");
                }
                out.s[out.n++] = cur;
            }
        } else {
            throw std::runtime_error("trie producer: unexpected stack element");
        }
    }
}

void succ_merge(PosSet& out, const Succ& in) {
    out.has_empty = out.has_empty || in.has_empty;
    for (int i = 0; i < in.n; ++i) { pos_add(out, in.s[i]); }
}

void pos_copy(PosSet& dst, const PosSet& src) {
    dst.n = src.n;
    dst.has_empty = src.has_empty;
    for (int i = 0; i < src.n; ++i) { dst.s[i] = src.s[i]; }
}

}  // namespace

bool CharClass::accepts(std::uint32_t cp) const noexcept {
    return in_ranges(ranges, cp) != negated;
}

bool CharClass::operator==(const CharClass& other) const noexcept {
    return negated == other.negated && ranges == other.ranges;
}

CharClass class_from_element(const Element* pos) {
    CharClass cls;
    if (pos->type == LLAMA_GRETYPE_CHAR_NOT) { cls.negated = true; }
    do {
        if (pos->type == LLAMA_GRETYPE_CHAR_ANY) {
            cls.ranges.push_back({0, 0x10FFFF});
            pos += 1;
        } else if (pos[1].type == LLAMA_GRETYPE_CHAR_RNG_UPPER) {
            cls.ranges.push_back({pos->value, pos[1].value});
            pos += 2;
        } else {
            cls.ranges.push_back({pos->value, pos->value});
            pos += 1;
        }
    } while (pos->type == LLAMA_GRETYPE_CHAR_ALT);
    std::sort(cls.ranges.begin(), cls.ranges.end());
    std::vector<std::pair<std::uint32_t, std::uint32_t>> merged;
    for (const auto& range : cls.ranges) {
        if (!merged.empty() && range.first <= merged.back().second + 1) {
            merged.back().second = std::max(merged.back().second, range.second);
        } else {
            merged.push_back(range);
        }
    }
    cls.ranges = std::move(merged);
    return cls;
}

bool decode_utf8_bytes(const std::uint8_t* bytes, std::size_t len,
                       const llama_partial_utf8& start, std::vector<std::uint32_t>& cps,
                       llama_partial_utf8& end) {
    std::size_t pos = 0;
    std::uint32_t value = start.value;
    int n_remain = start.n_remain;
    while (pos < len && bytes[pos] != 0 && n_remain > 0) {
        const std::uint8_t next_byte = bytes[pos];
        if ((next_byte >> 6) != 2) { return false; }
        value = (value << 6) + (next_byte & 0x3F);
        ++pos;
        --n_remain;
    }
    if (start.n_remain > 0 && n_remain == 0) { cps.push_back(value); }
    while (pos < len && bytes[pos] != 0) {
        const std::uint8_t first_byte = bytes[pos];
        const std::uint8_t highbits = first_byte >> 4;
        n_remain = kLookup[highbits] - 1;
        if (n_remain < 0) { return false; }
        const std::uint8_t mask = (1 << (7 - n_remain)) - 1;
        value = first_byte & mask;
        ++pos;
        while (pos < len && bytes[pos] != 0 && n_remain > 0) {
            value = (value << 6) + (bytes[pos] & 0x3F);
            ++pos;
            --n_remain;
        }
        if (n_remain == 0) { cps.push_back(value); }
    }
    end = llama_partial_utf8{value, n_remain};
    return true;
}

// ---------------------------------------------------------------------------
// Producer
// ---------------------------------------------------------------------------

Producer::Producer() = default;

void Producer::build(const ninfer::models::qwen3_5::frontend::GrammarVocabulary& vocabulary) {
    token_count_ = static_cast<std::uint32_t>(vocabulary.token_count());
    row_words_ = (token_count_ + 31) / 32;

    struct Item {
        int id;
        std::uint32_t off;
        std::uint16_t len;
    };
    std::vector<Item> items;
    path_data_.clear();
    partial_ids_.clear();
    exceptions_.clear();
    exception_bytes_.clear();
    eog_ids_.clear();

    for (int id = 0; id < static_cast<int>(token_count_); ++id) {
        if (vocabulary.is_eog(id)) {
            eog_ids_.push_back(id);
            continue;
        }
        const std::string_view piece = vocabulary.piece(id);
        if (piece.empty() || piece[0] == '\0') { continue; }
        std::size_t eff = 0;
        while (eff < piece.size() && piece[eff] != '\0') { ++eff; }

        std::vector<std::uint32_t> cps;
        llama_partial_utf8 end{};
        const llama_partial_utf8 start{};
        if (!decode_utf8_bytes(reinterpret_cast<const std::uint8_t*>(piece.data()), eff, start,
                               cps, end)) {
            ExceptionId ex;
            ex.id = id;
            ex.off = static_cast<std::uint32_t>(exception_bytes_.size());
            ex.len = static_cast<std::uint16_t>(eff);
            exception_bytes_.insert(exception_bytes_.end(),
                                    reinterpret_cast<const std::uint8_t*>(piece.data()),
                                    reinterpret_cast<const std::uint8_t*>(piece.data()) + eff);
            exceptions_.push_back(ex);
            continue;
        }
        if (end.n_remain > 0) {
            PartialId p;
            p.id = id;
            p.off = static_cast<std::uint32_t>(path_data_.size());
            p.len = static_cast<std::uint16_t>(cps.size());
            p.value = end.value;
            p.remain = end.n_remain;
            path_data_.insert(path_data_.end(), cps.begin(), cps.end());
            partial_ids_.push_back(p);
            continue;
        }
        if (cps.empty()) { continue; }  // unreachable: valid, complete, zero-codepoint piece
        Item item;
        item.id = id;
        item.off = static_cast<std::uint32_t>(path_data_.size());
        item.len = static_cast<std::uint16_t>(cps.size());
        path_data_.insert(path_data_.end(), cps.begin(), cps.end());
        items.push_back(item);
    }

    auto path_less = [&](const Item& a, const Item& b) {
        const std::uint32_t* pa = path_data_.data() + a.off;
        const std::uint32_t* pb = path_data_.data() + b.off;
        const std::size_t n = std::min(a.len, b.len);
        for (std::size_t i = 0; i < n; ++i) {
            if (pa[i] != pb[i]) { return pa[i] < pb[i]; }
        }
        return a.len < b.len;
    };
    std::sort(items.begin(), items.end(), path_less);

    nodes_.clear();
    edges_.clear();
    euler_ids_.clear();
    euler_off_.clear();
    euler_len_.clear();
    euler_node_.clear();
    std::vector<std::uint32_t> last_edge;
    nodes_.push_back(Node{});
    last_edge.push_back(kNoEdge);
    std::vector<std::uint32_t> chain;
    std::vector<std::uint32_t> prev_path;

    for (const Item& item : items) {
        const std::uint32_t* path = path_data_.data() + item.off;
        std::size_t common = 0;
        while (common < prev_path.size() && common < item.len && prev_path[common] == path[common]) {
            ++common;
        }
        chain.resize(common);
        for (std::size_t d = common; d < item.len; ++d) {
            const std::uint32_t parent = (d == 0) ? 0 : chain[d - 1];
            const std::uint32_t child = static_cast<std::uint32_t>(nodes_.size());
            Node node;
            node.depth = static_cast<std::uint16_t>(d + 1);
            // Lexicographic item order: this item is the first id of the new
            // subtree, so the subtree block starts here and is extended below.
            node.sub_begin = static_cast<std::uint32_t>(euler_ids_.size());
            node.sub_end = static_cast<std::uint32_t>(euler_ids_.size());
            nodes_.push_back(node);
            last_edge.push_back(kNoEdge);
            Edge edge;
            edge.cp = path[d];
            edge.child = child;
            edge.parent = parent;
            edge.next = kNoEdge;
            const std::uint32_t edge_index = static_cast<std::uint32_t>(edges_.size());
            if (nodes_[parent].first_edge == kNoEdge) {
                nodes_[parent].first_edge = edge_index;
            } else {
                edges_[last_edge[parent]].next = edge_index;
            }
            last_edge[parent] = edge_index;
            edges_.push_back(edge);
            chain.push_back(child);
        }
        const std::uint32_t end_node = chain.empty() ? 0 : chain.back();
        euler_ids_.push_back(item.id);
        euler_off_.push_back(item.off);
        euler_len_.push_back(item.len);
        euler_node_.push_back(end_node);
        Node& end = nodes_[end_node];
        if (end.own_begin == kNoEdge) { end.own_begin = static_cast<std::uint32_t>(euler_ids_.size()) - 1; }
        end.own_end = static_cast<std::uint32_t>(euler_ids_.size());
        prev_path.assign(path, path + item.len);
    }

    for (std::size_t n = nodes_.size(); n-- > 0;) {
        std::uint16_t best = 0;
        std::uint32_t end = nodes_[n].own_end == kNoEdge ? nodes_[n].sub_begin : nodes_[n].own_end;
        for (std::uint32_t e = nodes_[n].first_edge; e != kNoEdge; e = edges_[e].next) {
            const Node& child = nodes_[edges_[e].child];
            const std::uint16_t candidate = static_cast<std::uint16_t>(1 + child.max_remain);
            if (candidate > best) { best = candidate; }
            if (child.sub_end > end) { end = child.sub_end; }
        }
        nodes_[n].max_remain = best;
        nodes_[n].sub_end = end;
    }
    classes_.clear();
    class_index_.clear();
    build_child_index();
}

std::uint64_t Producer::memory_bytes() const noexcept {
    std::uint64_t bytes = 0;
    bytes += nodes_.capacity() * sizeof(Node);
    bytes += edges_.capacity() * sizeof(Edge);
    bytes += euler_ids_.capacity() * sizeof(int);
    bytes += euler_off_.capacity() * sizeof(std::uint32_t);
    bytes += euler_len_.capacity() * sizeof(std::uint16_t);
    bytes += path_data_.capacity() * sizeof(std::uint32_t);
    bytes += partial_ids_.capacity() * sizeof(PartialId);
    bytes += exceptions_.capacity() * sizeof(ExceptionId);
    bytes += exception_bytes_.capacity();
    bytes += cap_table_.capacity() * sizeof(CapEntry);
    bytes += prep_table_.capacity() * sizeof(PrepEntry);
    return bytes;
}

void Producer::build_child_index() {
    child_off_.assign(nodes_.size() + 1, 0);
    std::vector<std::uint32_t> counts(nodes_.size(), 0);
    for (std::uint32_t i = 0; i < edges_.size(); ++i) { ++counts[edges_[i].parent]; }
    std::uint32_t total = 0;
    for (std::uint32_t n = 0; n < nodes_.size(); ++n) {
        child_off_[n] = total;
        total += counts[n];
    }
    child_off_[nodes_.size()] = total;
    child_edges_.assign(total, 0);
    std::vector<std::uint32_t> fill(child_off_.begin(), child_off_.end() - 1);
    for (std::uint32_t i = 0; i < edges_.size(); ++i) {
        child_edges_[fill[edges_[i].parent]++] = i;
    }
    for (std::uint32_t n = 0; n < nodes_.size(); ++n) {
        std::sort(child_edges_.begin() + child_off_[n], child_edges_.begin() + child_off_[n + 1],
                  [this](std::uint32_t a, std::uint32_t b) { return edges_[a].cp < edges_[b].cp; });
    }
}

std::uint32_t Producer::class_id(const CharClass& cls) {
    std::string key;
    key.push_back(cls.negated ? '\1' : '\0');
    for (const auto& range : cls.ranges) {
        for (int shift = 24; shift >= 0; shift -= 8) {
            key.push_back(static_cast<char>((range.first >> shift) & 0xFF));
        }
        for (int shift = 24; shift >= 0; shift -= 8) {
            key.push_back(static_cast<char>((range.second >> shift) & 0xFF));
        }
    }
    auto it = class_index_.find(key);
    if (it != class_index_.end()) { return it->second; }
    const std::uint32_t id = static_cast<std::uint32_t>(classes_.size());
    classes_.push_back(ClassCache{cls, {}, {}, {}, false, false});
    class_index_.emplace(std::move(key), id);
    return id;
}

std::uint32_t Producer::class_id_for_top(const Element* top) {
    auto it = top_class_.find(top);
    if (it != top_class_.end()) { return it->second; }
    const CharClass cls = class_from_element(top);
    const std::uint32_t cid = class_id(cls);
    top_class_.emplace(top, cid);
    return cid;
}

const Producer::PrepEntry* Producer::prepare_cached(const Stk& stack) {
    if (prep_table_.empty()) { prep_table_.resize(1024); }
    static const auto hash_stack = [](const Stk& s) {
        std::uint64_t h = 1469598103934665603ULL;
        const auto* bytes = reinterpret_cast<const std::uint8_t*>(s.e);
        for (int i = 0; i < s.n * static_cast<int>(sizeof(Element*)); ++i) {
            h ^= bytes[i];
            h *= 1099511628211ULL;
        }
        return h;
    };
    const std::uint64_t h = hash_stack(stack);
    const std::size_t mask = prep_table_.size() - 1;
    std::size_t idx = static_cast<std::size_t>(h) & mask;
    std::size_t probes = 0;
    while (probes++ <= prep_table_.size()) {
        PrepEntry& entry = prep_table_[idx];
        if (entry.gen != prep_gen_) {
            entry.gen = prep_gen_;
            entry.hash = h;
            entry.key = stack;
            entry.is_char = false;
            entry.class_id = 0;
            entry.succ.n = 0;
            entry.succ.has_empty = false;
            const Element* top = stack.e[stack.n - 1];
            if (is_char_type(top)) {
                entry.is_char = true;
                entry.class_id = class_id_for_top(top);
                const std::uint32_t chr = first_accepted(classes_[entry.class_id].cls);
                const auto match = match_char(top, chr);
                Stk next;
                next.n = stack.n - 1;
                std::memcpy(next.e, stack.e, next.n * sizeof(Element*));
                if (!is_end_of_sequence(match.second)) { next.e[next.n++] = match.second; }
                advance_expand(*rules_, next, entry.succ);
            }
            return &entry;
        }
        if (entry.hash == h && entry.key.n == stack.n &&
            std::memcmp(entry.key.e, stack.e, stack.n * sizeof(Element*)) == 0) {
            return &entry;
        }
        idx = (idx + 1) & mask;
    }
}

void Producer::ensure_class_structures(std::uint32_t cid) {
    ClassCache& cache = classes_[cid];
    if (cache.ready) { return; }
    if (cache.cls.negated) {
        cache.has_exclusions = !cache.cls.ranges.empty();
        if (cache.has_exclusions) {
            cache.excl_bitmap.assign(euler_ids_.size(), 0);
            cache.excl_euler.reserve(256);
            for (std::size_t k = 0; k < euler_ids_.size(); ++k) {
                const std::uint32_t* path = path_data_.data() + euler_off_[k];
                for (std::uint16_t i = 0; i < euler_len_[k]; ++i) {
                    if (in_ranges(cache.cls.ranges, path[i])) {
                        cache.excl_bitmap[k] = 1;
                        cache.excl_euler.push_back(static_cast<std::uint32_t>(k));
                        break;
                    }
                }
            }
        }
    } else {
        cache.outside.assign(nodes_.size(), 0);
        for (std::size_t n = nodes_.size(); n-- > 0;) {
            std::uint8_t outside = 0;
            for (std::uint32_t e = nodes_[n].first_edge; e != kNoEdge; e = edges_[e].next) {
                if (!cache.cls.accepts(edges_[e].cp) || cache.outside[edges_[e].child]) {
                    outside = 1;
                    break;
                }
            }
            cache.outside[n] = outside;
        }
    }
    cache.ready = true;
}

const std::vector<std::uint8_t>& Producer::outside_for(std::uint32_t cid) {
    ensure_class_structures(cid);
    return classes_[cid].outside;
}

void Producer::set_bit(int id) {
    if (writes_enabled) {
        (*row_)[static_cast<std::size_t>(id) >> 5U] |= 1U << (static_cast<unsigned>(id) & 31U);
        ++stats.bits_set;
    }
}

int Producer::capacity_of(const Stk& stack_key, const CharClass& cls, int fuel) {
    ++stats.capacity_calls;
    if (fuel <= 0 || stack_key.n == 0) { return 0; }
    int* slot = cap_slot(stack_key);
    if (*slot >= 0) { return *slot; }
    const Element* top = stack_key.e[stack_key.n - 1];
    if (!is_char_type(top) || !accepts_all(top, cls)) {
        *slot = 0;
        return 0;
    }
    const std::uint32_t chr = first_accepted(cls);
    const auto match = match_char(top, chr);
    Stk next;
    next.n = stack_key.n - 1;
    std::memcpy(next.e, stack_key.e, next.n * sizeof(Element*));
    if (!is_end_of_sequence(match.second)) { next.e[next.n++] = match.second; }
    Succ succ;
    advance_expand(*rules_, next, succ);
    *slot = fuel;  // in-progress sentinel: a cycle saturates
    int best = 0;
    for (int i = 0; i < succ.n; ++i) {
        const int cap = capacity_of(succ.s[i], cls, fuel - 1);
        if (cap > best) { best = cap; }
    }
    ++stats.capacity_steps;
    int result = 1 + best;
    if (result > fuel) { result = fuel; }
    *slot = result;
    return result;
}

int* Producer::cap_slot(const Stk& stack) {
    if (cap_table_.empty()) {
        cap_table_.resize(8192);
    }
    static const auto hash_stack = [](const Stk& s) {
        std::uint64_t h = 1469598103934665603ULL;
        const auto* bytes = reinterpret_cast<const std::uint8_t*>(s.e);
        for (int i = 0; i < s.n * static_cast<int>(sizeof(Element*)); ++i) {
            h ^= bytes[i];
            h *= 1099511628211ULL;
        }
        return h;
    };
    const std::uint64_t h = hash_stack(stack);
    const std::size_t mask = cap_table_.size() - 1;
    std::size_t idx = static_cast<std::size_t>(h) & mask;
    std::size_t probes = 0;
    while (probes++ <= cap_table_.size()) {
        CapEntry& entry = cap_table_[idx];
        if (entry.gen != cap_gen_) {
            entry.gen = cap_gen_;
            entry.hash = h;
            entry.stack = stack;
            entry.cap = -1;
            return &entry.cap;
        }
        if (entry.hash == h && entry.stack.n == stack.n &&
            std::memcmp(entry.stack.e, stack.e, stack.n * sizeof(Element*)) == 0) {
            return &entry.cap;
        }
        idx = (idx + 1) & mask;
    }
}

bool Producer::try_bulk(std::uint32_t node, const PosSet& pos, std::uint32_t depth,
                        const PrepEntry* const* prepared) {
    const std::uint32_t max_remain = nodes_[node].max_remain;
    if (std::getenv("TRIE_DEBUG_STK") != nullptr && node == 0 && depth == 0 && stats.bulks < 4) {
        std::fprintf(stderr, "[bulk] fill=%llu node=0 max_remain=%u pos.n=%u",
                     (unsigned long long)stats.fills, max_remain, pos.n);
        for (int i = 0; i < pos.n; ++i) {
            const Element* t = pos.s[i].e[pos.s[i].n - 1];
            const std::uint32_t cid = class_id_for_top(t);
            std::fprintf(stderr, " | pos%d n=%u top=(%d,%u) cid=%u neg=%d cap=%d", i, pos.s[i].n,
                         (int)t->type, t->value, cid,
                         (int)classes_[cid].cls.negated,
                         capacity_of(pos.s[i], classes_[cid].cls, kCapFuel));
        }
        std::fprintf(stderr, "\n");
        std::fflush(stderr);
    }
    if (max_remain == 0) { return true; }  // leaf: own ids already set
    if (pos.n == 0) { return false; }
    for (int i = 0; i < pos.n; ++i) {
        const Element* top = pos.s[i].e[pos.s[i].n - 1];
        if (!is_char_type(top)) { continue; }
        const std::uint32_t cid = prepared != nullptr ? prepared[i]->class_id : class_id_for_top(top);
        if (!classes_[cid].cls.negated && outside_for(cid)[node] != 0) {
            ++stats.declined_class;
            continue;
        }
        if (capacity_of(pos.s[i], classes_[cid].cls, kCapFuel) < static_cast<int>(max_remain)) {
            ++stats.declined_capacity;
            continue;
        }
        ClassCache& cache = classes_[cid];
        ensure_class_structures(cid);
        const bool skip_excluded = cache.has_exclusions;
        const std::uint64_t tset = __rdtsc();
        if (writes_enabled) {
            for (std::uint32_t k = nodes_[node].sub_begin; k < nodes_[node].sub_end; ++k) {
                if (skip_excluded && cache.excl_bitmap[k] != 0) { continue; }
                set_bit(euler_ids_[k]);
            }
        }
        stats.cyc_bulk_set += __rdtsc() - tset;
        stats.bulk_ids += nodes_[node].sub_end - nodes_[node].sub_begin;
        ++stats.bulks;
        if (skip_excluded) {
            // Ids whose remaining path hits an excluded codepoint are judged exactly
            // against this node's position set (other positions may still accept).
            const auto& excluded = cache.excl_euler;
            auto it = std::lower_bound(excluded.begin(), excluded.end(), nodes_[node].sub_begin);
            const auto end = std::lower_bound(excluded.begin(), excluded.end(), nodes_[node].sub_end);
            for (; it != end; ++it) {
                const std::uint32_t k = *it;
                ++stats.excl_judged;
                judge_linear(pos, path_data_.data() + euler_off_[k] + depth,
                             euler_len_[k] - depth, llama_partial_utf8{0, 0}, euler_ids_[k],
                             false);
            }
        }
        return true;
    }
    return false;
}

void Producer::visit(std::uint32_t node, const PosSet& pos, std::uint32_t depth,
                     std::vector<int>& anc_pos, std::vector<int>& anc_neg) {
    ++stats.nodes_visited;
    const Node& nd = nodes_[node];
    if (writes_enabled && nd.own_begin != kNoEdge) {
        for (std::uint32_t k = nd.own_begin; k < nd.own_end; ++k) { set_bit(euler_ids_[k]); }
    }

    if (bulk_enabled && try_bulk(node, pos, depth, nullptr)) { return; }

    const std::size_t ap0 = anc_pos.size();
    const std::size_t an0 = anc_neg.size();
    for (int i = 0; i < pos.n; ++i) {
        const Element* top = pos.s[i].e[pos.s[i].n - 1];
        if (top->type == LLAMA_GRETYPE_TOKEN) {
            anc_pos.push_back(static_cast<int>(top->value));
        } else if (top->type == LLAMA_GRETYPE_TOKEN_NOT) {
            anc_neg.push_back(static_cast<int>(top->value));
        }
    }

    const PrepEntry* prepared[kMaxPos];
    for (int i = 0; i < pos.n; ++i) { prepared[i] = prepare_cached(pos.s[i]); }

    if (std::getenv("TRIE_DEBUG_STK") != nullptr && node == 0 && depth == 0) {
        std::fprintf(stderr, "[visit] fill=%llu node=0 depth=0 pos.n=%u\n",
                     (unsigned long long)stats.fills, pos.n);
        std::fflush(stderr);
    }
    if (std::getenv("TRIE_DEBUG_STK") != nullptr && stats.fills == 15 && node == 0 && depth == 0) {
        for (int i = 0; i < pos.n; ++i) {
            std::fprintf(stderr,
                         "[stk] fill=%llu pos=%d n=%u is_char=%d cid=%u succ_n=%d has_empty=%d:",
                         (unsigned long long)stats.fills, i, pos.s[i].n, (int)prepared[i]->is_char,
                         prepared[i]->class_id, (int)prepared[i]->succ.n,
                         (int)prepared[i]->succ.has_empty);
            for (int j = 0; j < pos.s[i].n; ++j) {
                std::fprintf(stderr, " (%d,%u)", (int)pos.s[i].e[j]->type, pos.s[i].e[j]->value);
            }
            std::fprintf(stderr, "\n");
        }
        for (std::uint32_t cid = 0; cid < classes_.size(); ++cid) {
            const ClassCache& cc = classes_[cid];
            auto fxx = [](const std::vector<std::uint8_t>& v) {
                std::uint64_t h = 1469598103934665603ULL;
                for (std::uint8_t b : v) {
                    h ^= b;
                    h *= 1099511628211ULL;
                }
                return h;
            };
            std::fprintf(stderr,
                         "[cls] cid=%u neg=%d nrange=%zu ready=%d outside_sz=%zu outside_h=%llu "
                         "excl_sz=%zu excl_h=%llu has_excl=%d\n",
                         cid, (int)cc.cls.negated, cc.cls.ranges.size(), (int)cc.ready,
                         cc.outside.size(), (unsigned long long)fxx(cc.outside),
                         cc.excl_bitmap.size(), (unsigned long long)fxx(cc.excl_bitmap),
                         (int)cc.has_exclusions);
            for (const auto& r : cc.cls.ranges) {
                std::fprintf(stderr, "  (%u,%u)\n", r.first, r.second);
            }
        }
        std::fflush(stderr);
    }

    struct SigEntry {
        std::uint64_t sig;
        PosSet set;
    };
    constexpr int kSigCache = 4;
    SigEntry sigs[kSigCache];
    int nsigs = 0;

    for (std::uint32_t e = nd.first_edge; e != kNoEdge; e = edges_[e].next) {
        ++stats.edges_visited;
        const std::uint32_t cp = edges_[e].cp;
        std::uint64_t sig = 0;
        for (int i = 0; i < pos.n; ++i) {
            if (prepared[i]->is_char && classes_[prepared[i]->class_id].cls.accepts(cp)) {
                sig |= 1ULL << i;
            }
        }
        PosSet local;
        PosSet* next = &local;
        if (sig == 0) {
            local.n = 0;
            local.has_empty = false;
        } else {
            bool found = false;
            for (int i = 0; i < nsigs; ++i) {
                if (sigs[i].sig == sig) {
                    next = &sigs[i].set;
                    found = true;
                    break;
                }
            }
            if (!found) {
                local.n = 0;
                local.has_empty = false;
                for (int i = 0; i < pos.n; ++i) {
                    if ((sig >> i) & 1U) { succ_merge(local, prepared[i]->succ); }
                }
                if (nsigs < kSigCache) {
                    sigs[nsigs].sig = sig;
                    pos_copy(sigs[nsigs].set, local);
                    next = &sigs[nsigs].set;
                    ++nsigs;
                }
            }
        }
        if (next->n == 0 && !next->has_empty && anc_pos.empty() && anc_neg.empty()) { continue; }
        visit(edges_[e].child, *next, depth + 1, anc_pos, anc_neg);
    }

    anc_pos.resize(ap0);
    anc_neg.resize(an0);
}

void Producer::judge_linear(const PosSet& start, const std::uint32_t* cps, std::size_t len,
                            llama_partial_utf8 tail, int id, bool count_tail) {
    if (count_tail) { ++stats.tail_judged; }
    PosSet cur = start;
    for (std::size_t d = 0; d < len; ++d) {
        for (int i = 0; i < cur.n; ++i) {
            const Element* top = cur.s[i].e[cur.s[i].n - 1];
            if ((top->type == LLAMA_GRETYPE_TOKEN && static_cast<int>(top->value) == id) ||
                (top->type == LLAMA_GRETYPE_TOKEN_NOT && static_cast<int>(top->value) != id)) {
                set_bit(id);
                return;
            }
        }
        PosSet next;
        for (int i = 0; i < cur.n; ++i) {
            const Element* top = cur.s[i].e[cur.s[i].n - 1];
            if (!is_char_type(top)) { continue; }
            const auto match = match_char(top, cps[d]);
            if (!match.first) { continue; }
            Stk stack;
            stack.n = cur.s[i].n - 1;
            std::memcpy(stack.e, cur.s[i].e, stack.n * sizeof(Element*));
            if (!is_end_of_sequence(match.second)) { stack.e[stack.n++] = match.second; }
            Succ succ;
            advance_expand(*rules_, stack, succ);
            succ_merge(next, succ);
        }
        cur = next;
        ++stats.tail_steps;
        // Empty stacks reject while codepoints remain; a lone empty stack still
        // accepts at the end of the piece (handled by the end check below).
        if (cur.n == 0 && (!cur.has_empty || d + 1 < len)) { return; }
    }
    bool accept = false;
    for (int i = 0; i < cur.n && !accept; ++i) {
        const Element* top = cur.s[i].e[cur.s[i].n - 1];
        if (top->type == LLAMA_GRETYPE_TOKEN || top->type == LLAMA_GRETYPE_TOKEN_NOT) {
            accept = tail.n_remain == 0;
        } else if (is_char_type(top)) {
            accept = tail.n_remain == 0 || match_partial_char(top, tail);
        }
    }
    if (!accept && cur.has_empty && tail.n_remain == 0) { accept = true; }
    if (accept) { set_bit(id); }
}


// ---------------------------------------------------------------------------
// Class-counter fast path
// ---------------------------------------------------------------------------

bool Producer::is_chain_ref(const Element* e) const noexcept {
    return !chain_refs_.empty() &&
           std::binary_search(chain_refs_.begin(), chain_refs_.end(), e);
}

// Expands `stack` until char/token tops, tracking
//   k      = number of counter rounds the path has consumed (thr semantics),
//   region = position still inside a round body (vs handed off to the continuation).
// A chain frame ref is traversed exactly at a round boundary: its body alternative
// takes k+1, its empty alternative (the continuation) takes k.
void Producer::expand_threshold(const Stk& stack, std::uint8_t k, bool region, KSet& out) {
    if (fast_abort_) { return; }
    if (stack.n == 0) {
        out.has_empty = true;
        if (k < out.empty_k) { out.empty_k = k; }
        return;
    }
    if (stack.n > kMaxStack || k > 200) {
        fast_abort_ = true;
        return;
    }
    const Element* top = stack.e[stack.n - 1];
    if (top->type == LLAMA_GRETYPE_RULE_REF) {
        const bool chain = is_chain_ref(top);
        const auto& rule = fast_rules_->at(top->value);
        const Element* subpos = rule.data();
        do {
            Stk next;
            next.n = static_cast<std::uint8_t>(stack.n - 1);
            std::memcpy(next.e, stack.e, next.n * sizeof(Element*));
            if (!is_end_of_sequence(top + 1)) { next.e[next.n++] = top + 1; }
            if (!is_end_of_sequence(subpos)) { next.e[next.n++] = subpos; }
            if (next.n > kMaxStack) {
                fast_abort_ = true;
                return;
            }
            const bool body_alt = subpos == rule.data();
            const std::uint8_t child_k =
                chain ? static_cast<std::uint8_t>(body_alt ? k + 1 : k) : k;
            if (fast_body_only_ && chain && !body_alt) {
                // Body-only mode: the empty alternative marks a round boundary.
                out.has_empty = true;
                if (child_k < out.empty_k) { out.empty_k = child_k; }
            } else {
                expand_threshold(next, child_k, chain ? body_alt : region, out);
            }
            while (!is_end_of_sequence(subpos)) { ++subpos; }
            if (subpos->type == LLAMA_GRETYPE_ALT) {
                ++subpos;
            } else {
                break;
            }
        } while (true);
        return;
    }
    if (top->type == LLAMA_GRETYPE_TOKEN || top->type == LLAMA_GRETYPE_TOKEN_NOT) {
        fast_abort_ = true;  // token terminals are not modelled by the counter table
        return;
    }
    if (!is_char_type(top) || out.n >= kFastPos || stack.n > kFastStack) {
        fast_abort_ = true;
        return;
    }
    KPos& pos = out.p[out.n++];
    pos.stack.n = static_cast<std::uint8_t>(stack.n);
    std::memcpy(pos.stack.e, stack.e, stack.n * sizeof(Element*));
    pos.k = k;
    pos.region = region;
}

const Producer::FragEntry* Producer::frag_cached(const KPos& q) {
    if (frag_table_.empty()) { frag_table_.resize(16384); }
    static const auto hash_key = [](const FragEntry& e) {
        std::uint64_t h = 1469598103934665603ULL;
        const auto* bytes = reinterpret_cast<const std::uint8_t*>(e.key.e);
        for (int i = 0; i < e.key.n * static_cast<int>(sizeof(Element*)); ++i) {
            h ^= bytes[i];
            h *= 1099511628211ULL;
        }
        h ^= e.k;
        h *= 1099511628211ULL;
        h ^= e.region ? 1 : 0;
        return h;
    };
    FragEntry probe;
    probe.key.n = q.stack.n;
    std::memcpy(probe.key.e, q.stack.e, q.stack.n * sizeof(Element*));
    probe.k = q.k;
    probe.region = q.region;
    const std::uint64_t h = hash_key(probe);
    const std::size_t mask = frag_table_.size() - 1;
    std::size_t idx = static_cast<std::size_t>(h) & mask;
    std::size_t probes = 0;
    while (probes++ <= frag_table_.size()) {
        FragEntry& entry = frag_table_[idx];
        if (entry.gen != frag_gen_) {
            entry.gen = frag_gen_;
            entry.hash = h;
            entry.key = probe.key;
            entry.k = q.k;
            entry.region = q.region;
            entry.is_char = false;
            entry.class_id = 0;
            entry.succ = KSet{};
            const Element* top = q.stack.e[q.stack.n - 1];
            if (is_char_type(top)) {
                entry.is_char = true;
                entry.class_id = class_id_for_top(top);
                const std::uint32_t chr = first_accepted(classes_[entry.class_id].cls);
                const auto match = match_char(top, chr);
                Stk next;
                next.n = static_cast<std::uint8_t>(q.stack.n - 1);
                std::memcpy(next.e, q.stack.e, next.n * sizeof(Element*));
                if (!is_end_of_sequence(match.second)) { next.e[next.n++] = match.second; }
                expand_threshold(next, q.k, q.region, entry.succ);
            }
            return &entry;
        }
        if (entry.hash == h && entry.key.n == q.stack.n && entry.k == q.k &&
            entry.region == q.region &&
            std::memcmp(entry.key.e, q.stack.e, q.stack.n * sizeof(Element*)) == 0) {
            return &entry;
        }
        idx = (idx + 1) & mask;
    }
}

void Producer::threshold_visit(std::uint32_t node, const KSet& cur) {
    if (fast_abort_) { return; }

    for (int i = 0; i < cur.n; ++i) {
        const std::uint8_t k = cur.p[i].k;
        if (k < min_k_all_[node]) { min_k_all_[node] = k; }
        if (cur.p[i].region && k < min_k_region_[node]) { min_k_region_[node] = k; }
    }
    if (cur.has_empty) {
        if (cur.empty_k < min_k_all_[node]) { min_k_all_[node] = cur.empty_k; }
        if (cur.empty_k < min_k_boundary_[node]) { min_k_boundary_[node] = cur.empty_k; }
    }

    // Per-position successor fragments (memoized per (stack, k, region)).
    const FragEntry* frags[kFastPos];
    for (int i = 0; i < cur.n; ++i) {
        frags[i] = frag_cached(cur.p[i]);
        if (fast_abort_) { return; }
    }

    for (std::uint32_t e = nodes_[node].first_edge; e != kNoEdge; e = edges_[e].next) {
        const std::uint32_t cp = edges_[e].cp;
        KSet next;
        for (int i = 0; i < cur.n; ++i) {
            if (!frags[i]->is_char) { continue; }
            if (!classes_[frags[i]->class_id].cls.accepts(cp)) { continue; }
            const KSet& frag = frags[i]->succ;
            if (frag.has_empty) {
                next.has_empty = true;
                if (frag.empty_k < next.empty_k) { next.empty_k = frag.empty_k; }
            }
            for (int q = 0; q < frag.n; ++q) {
                int found = -1;
                for (int r = 0; r < next.n; ++r) {
                    if (next.p[r].stack.n == frag.p[q].stack.n &&
                        std::memcmp(next.p[r].stack.e, frag.p[q].stack.e,
                                    frag.p[q].stack.n * sizeof(Element*)) == 0) {
                        found = r;
                        break;
                    }
                }
                if (found >= 0) {
                    if (frag.p[q].k < next.p[found].k) { next.p[found].k = frag.p[q].k; }
                    next.p[found].region = next.p[found].region || frag.p[q].region;
                } else {
                    if (next.n >= kFastPos) {
                        fast_abort_ = true;
                        return;
                    }
                    next.p[next.n++] = frag.p[q];
                }
            }
        }
        if (next.n == 0 && !next.has_empty) { continue; }
        threshold_visit(edges_[e].child, next);
        if (fast_abort_) { return; }
    }
}

bool Producer::recognize_chain(const llama_grammar_rules& rules, const PosSet& pos, std::string& key,
                               int& p, std::vector<Stk>& c_set) {
    ++stats.recog_attempts;
    if (pos.n == 0) { return false; }
    // No token terminals anywhere in the state positions.
    for (int i = 0; i < pos.n; ++i) {
        for (int j = 0; j < pos.s[i].n; ++j) {
            const Element* el = pos.s[i].e[j];
            if (el->type == LLAMA_GRETYPE_TOKEN || el->type == LLAMA_GRETYPE_TOKEN_NOT) {
                return false;
            }
        }
    }

    const bool dbg = std::getenv("TRIE_DEBUG_REC") != nullptr;
    static int dbg_n = 0;
    if (dbg && stats.recog_attempts <= 3) {
        std::fprintf(stderr, "[rec] attempt=%llu pos.n=%u has_empty=%d:",
                     (unsigned long long)stats.recog_attempts, pos.n, (int)pos.has_empty);
        for (int i = 0; i < pos.n; ++i) {
            std::fprintf(stderr, " [%u:", pos.s[i].n);
            for (int j = 0; j < pos.s[i].n; ++j) {
                std::fprintf(stderr, " (%d,%u)", (int)pos.s[i].e[j]->type, pos.s[i].e[j]->value);
            }
        }
        std::fprintf(stderr, "]\n");
        std::fflush(stderr);
    }
    for (int i = 0; i < pos.n; ++i) {
        const Stk& s = pos.s[i];
        if (s.n < 2) {
            if (dbg && dbg_n < 8) { ++dbg_n; std::fprintf(stderr, "rec: pos too short n=%u\n", s.n); }
            continue;
        }
        const Element* ref = s.e[s.n - 2];
        if (ref->type != LLAMA_GRETYPE_RULE_REF) {
            if (dbg && dbg_n < 8) {
                ++dbg_n;
                std::fprintf(stderr, "rec: e[n-2] not a ref type=%d value=%u\n", (int)ref->type,
                             ref->value);
            }
            continue;
        }
        // Region stacks share this exact ref below a char top; every other stack
        // must be exactly the expansion of the continuation below the ref.
        int region_n = 0;
        for (int b = 0; b < pos.n; ++b) {
            const Stk& t = pos.s[b];
            if (t.n >= 2 && t.e[t.n - 2] == ref && is_char_type(t.e[t.n - 1])) { ++region_n; }
        }
        if (region_n == 0) {
            if (dbg && dbg_n < 8) {
                ++dbg_n;
                std::fprintf(stderr, "rec: region_n=0 ref=%u\n", ref->value);
            }
            continue;
        }
        Stk c_raw;
        c_raw.n = static_cast<std::uint8_t>(s.n - 2);
        std::memcpy(c_raw.e, s.e, c_raw.n * sizeof(Element*));
        Succ c_expanded;
        advance_expand(rules, c_raw, c_expanded);
        if (c_expanded.n != pos.n - region_n) {
            if (dbg && dbg_n < 8) { ++dbg_n; std::fprintf(stderr, "rec: c-size mismatch cexp=%d pos_nonreg=%d ref=%u\n", (int)c_expanded.n, pos.n - region_n, ref->value); }
            continue;
        }
        if (c_expanded.has_empty != pos.has_empty) {
            if (dbg && dbg_n < 8) { ++dbg_n; std::fprintf(stderr, "rec: c-empty mismatch\n"); }
            continue;
        }
        bool c_ok = true;
        for (int a = 0; a < c_expanded.n && c_ok; ++a) {
            bool found = false;
            for (int b = 0; b < pos.n; ++b) {
                const Stk& t = pos.s[b];
                if (t.n >= 2 && t.e[t.n - 2] == ref) { continue; }
                if (t.n == c_expanded.s[a].n &&
                    std::memcmp(t.e, c_expanded.s[a].e, t.n * sizeof(Element*)) == 0) {
                    found = true;
                    break;
                }
            }
            c_ok = found;
        }
        if (!c_ok) {
            if (dbg && dbg_n < 8) { ++dbg_n; std::fprintf(stderr, "rec: c-content mismatch ref=%u\n", ref->value); }
            continue;
        }

        c_set.clear();
        c_set.reserve(c_expanded.n);
        for (int a = 0; a < c_expanded.n; ++a) {
            Stk st;
            st.n = c_expanded.s[a].n;
            std::memcpy(st.e, c_expanded.s[a].e, st.n * sizeof(Element*));
            c_set.push_back(std::move(st));
        }
        if (c_expanded.has_empty) { c_set.push_back(Stk{}); }

        // Frame chain validation from ref. A frame is `BODY (REF link)? | ε`; the
        // last body element is a link only when the referenced rule is a frame whose
        // body matches this frame's body (the module's normalization rule).
        struct Resolver {
            const llama_grammar_rules& rules;
            std::vector<std::uint8_t> state;  // 0 unknown, 1 busy, 2 ok, 3 bad
            std::vector<std::uint32_t> blen;
            std::vector<std::uint8_t> link;
            static bool elements_equal(const llama_grammar_rule& a, std::size_t ao,
                                       const llama_grammar_rule& b, std::size_t bo, std::size_t n) {
                for (std::size_t j = 0; j < n; ++j) {
                    if (a[ao + j].type != b[bo + j].type || a[ao + j].value != b[bo + j].value) {
                        return false;
                    }
                }
                return true;
            }
            bool resolve(std::uint32_t rid) {
                if (state[rid] == 2) { return true; }
                if (state[rid] != 0) { return false; }
                state[rid] = 1;
                const llama_grammar_rule& rule = rules[rid];
                std::size_t alt = 0;
                int nalts = 0;
                if (rule.size() >= 3 && rule.back().type == LLAMA_GRETYPE_END) {
                    for (std::size_t j = 0; j + 1 < rule.size(); ++j) {
                        if (rule[j].type == LLAMA_GRETYPE_ALT) { alt = j; ++nalts; }
                    }
                }
                bool ok = nalts == 1 && alt >= 1;
                if (ok) {
                    blen[rid] = static_cast<std::uint32_t>(alt);
                    link[rid] = 0;
                    const Element& last = rule[alt - 1];
                    if (last.type == LLAMA_GRETYPE_RULE_REF) {
                        const std::uint32_t t = last.value;
                        if (t < rules.size() && t == rid) {
                            link[rid] = 1;
                            blen[rid] = static_cast<std::uint32_t>(alt - 1);
                        } else if (t < rules.size() && resolve(t) && blen[t] == alt - 1 &&
                                   elements_equal(rule, 0, rules[t], 0, blen[t])) {
                            link[rid] = 1;
                            blen[rid] = static_cast<std::uint32_t>(alt - 1);
                        }
                    }
                }
                state[rid] = ok ? 2 : 3;
                return ok;
            }
        };
        Resolver resolver{rules, std::vector<std::uint8_t>(rules.size(), 0),
                          std::vector<std::uint32_t>(rules.size(), 0),
                          std::vector<std::uint8_t>(rules.size(), 0)};

        chain_refs_.clear();
        const Element* cur = ref;
        std::size_t frames = 0;
        const llama_grammar_rule* body0 = nullptr;
        std::size_t body0_len = 0;
        bool unbounded = false;
        bool walk_ok = true;
        while (true) {
            bool seen = false;
            for (const Element* r : chain_refs_) {
                if (r == cur) { seen = true; break; }
            }
            if (seen) { unbounded = true; break; }
            if (cur->value >= rules.size() || !resolver.resolve(cur->value)) { walk_ok = false; break; }
            chain_refs_.push_back(cur);
            const llama_grammar_rule& rule = rules[cur->value];
            const std::size_t blen = resolver.blen[cur->value];
            if (body0 == nullptr) {
                body0 = &rule;
                body0_len = blen;
            } else {
                if (body0_len != blen ||
                    !Resolver::elements_equal(*body0, 0, rule, 0, blen)) { walk_ok = false; break; }
            }
            ++frames;
            if (resolver.link[cur->value]) {
                cur = &rule[blen];
            } else {
                break;
            }
        }
        if (!walk_ok || frames == 0) {
            if (dbg && dbg_n < 8) { ++dbg_n; std::fprintf(stderr, "rec: walk fail frames=%zu ref=%u\n", frames, ref->value); }
            continue;
        }
        std::sort(chain_refs_.begin(), chain_refs_.end());
        p = unbounded ? (1 << 20) : static_cast<int>(frames + 1);

        // C must not contain any chain ref (would corrupt the round bookkeeping).
        bool ref_in_c = false;
        for (const Stk& st : c_set) {
            for (int j = 0; j < st.n && !ref_in_c; ++j) {
                if (is_chain_ref(st.e[j])) { ref_in_c = true; }
            }
        }
        if (ref_in_c) { continue; }

        // Key: C_raw located by (rule, index) + body elements by (type, value).
        // Rule/index location uses a linear scan over rules (rules are few hundred to
        // few thousand; done once per recognized state family candidate).
        auto locate = [&](const Element* el) -> std::pair<std::uint32_t, std::uint32_t> {
            for (std::uint32_t r = 0; r < rules.size(); ++r) {
                const auto& rr = rules[r];
                if (!rr.empty() && el >= rr.data() && el < rr.data() + rr.size()) {
                    return {r, static_cast<std::uint32_t>(el - rr.data())};
                }
            }
            return {0xFFFFFFFFU, 0};
        };
        key.clear();
        for (int j = 0; j < c_raw.n; ++j) {
            const auto [r, idx] = locate(c_raw.e[j]);
            if (r == 0xFFFFFFFFU) { break; }
            for (int sh = 24; sh >= 0; sh -= 8) { key.push_back(static_cast<char>((r >> sh) & 0xFF)); }
            for (int sh = 24; sh >= 0; sh -= 8) { key.push_back(static_cast<char>((idx >> sh) & 0xFF)); }
        }
        key.push_back('|');
        fast_body_key_.clear();
        for (std::size_t j = 0; j < body0_len; ++j) {
            for (int sh = 24; sh >= 0; sh -= 8) {
                key.push_back(static_cast<char>(((*body0)[j].type >> sh) & 0xFF));
                fast_body_key_.push_back(static_cast<char>(((*body0)[j].type >> sh) & 0xFF));
            }
            for (int sh = 24; sh >= 0; sh -= 8) {
                key.push_back(static_cast<char>(((*body0)[j].value >> sh) & 0xFF));
                fast_body_key_.push_back(static_cast<char>(((*body0)[j].value >> sh) & 0xFF));
            }
        }
        ++stats.recog_ok;
        return true;
    }
    return false;
}

bool Producer::build_body_table(const std::string& body_key, int p) {
    auto it = bodies_.find(body_key);
    if (it != bodies_.end() && it->second.max_p >= 0 &&
        it->second.max_p >= std::min(p, 128)) {
        return true;
    }
    if (std::getenv("TRIE_DEBUG_TAB")) { std::fprintf(stderr, "[body] start key=%zu p=%d\n", body_key.size(), p); std::fflush(stderr); }
    const auto t0 = std::chrono::steady_clock::now();
    fast_rules_ = rules_;
    fast_abort_ = false;
    fast_body_only_ = true;
    ++frag_gen_;
    min_k_all_.assign(nodes_.size(), 0xFF);
    min_k_region_.assign(nodes_.size(), 0xFF);
    min_k_boundary_.assign(nodes_.size(), 0xFF);

    KSet init;
    for (int i = 0; i < root_.n; ++i) {
        bool in_c = false;
        for (const Stk& st : fast_c_set_) {
            if (st.n == root_.s[i].n &&
                std::memcmp(st.e, root_.s[i].e, st.n * sizeof(Element*)) == 0) {
                in_c = true;
                break;
            }
        }
        if (in_c) { continue; }
        if (init.n >= kFastPos || root_.s[i].n > kFastStack) {
            fast_abort_ = true;
            break;
        }
        KPos& kp = init.p[init.n++];
        kp.stack.n = root_.s[i].n;
        std::memcpy(kp.stack.e, root_.s[i].e, kp.stack.n * sizeof(Element*));
        kp.k = 1;
        kp.region = true;
    }
    if (std::getenv("TRIE_DEBUG_TAB")) { std::fprintf(stderr, "[body] dfs start n=%d\n", (int)init.n); std::fflush(stderr); }
    if (!fast_abort_) { threshold_visit(0, init); }
    fast_body_only_ = false;
    if (std::getenv("TRIE_DEBUG_TAB")) { std::fprintf(stderr, "[body] dfs done abort=%d\n", (int)fast_abort_); std::fflush(stderr); }

    BodyEntry& body = bodies_[body_key];
    if (fast_abort_) {
        body.max_p = -1;
        body.buckets.clear();
        body.node_boundary.clear();
        body.thr.clear();
        ++stats.table_build_fail;
        return false;
    }
    const int table_max = std::min(p, 128);
    body.max_p = table_max;
    body.node_boundary = min_k_boundary_;
    body.buckets.assign(static_cast<std::size_t>(table_max + 1) * row_words_, 0);
    body.thr.assign(euler_ids_.size(), 0xFF);
    for (std::size_t k = 0; k < euler_ids_.size(); ++k) {
        const std::uint32_t e = euler_node_[k];
        std::uint8_t m = min_k_region_[e];
        if (min_k_boundary_[e] < m) { m = min_k_boundary_[e]; }
        if (m == 0xFF || m > table_max) { continue; }  // not region-acceptable in the table range
        body.thr[k] = m;
        const int id = euler_ids_[k];
        body.buckets[static_cast<std::size_t>(m) * row_words_ + (id >> 5)] |= 1U << (id & 31);
    }
    ++stats.body_builds;
    const auto t1 = std::chrono::steady_clock::now();
    stats.body_build_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    return true;
}

std::uint32_t Producer::trie_child(std::uint32_t node, std::uint32_t cp) const noexcept {
    if (child_off_.empty()) { return kNoEdge; }
    auto begin = child_edges_.begin() + child_off_[node];
    auto end = child_edges_.begin() + child_off_[node + 1];
    auto it = std::lower_bound(begin, end, cp,
                               [this](std::uint32_t e, std::uint32_t v) { return edges_[e].cp < v; });
    if (it == end || edges_[*it].cp != cp) { return kNoEdge; }
    return edges_[*it].child;
}

bool Producer::c_accepts_suffix(const std::uint32_t* cps, std::size_t len) {
    std::vector<Stk> cur = fast_c_set_;
    bool has_empty = false;
    for (const Stk& st : cur) {
        if (st.n == 0) { has_empty = true; }
    }
    for (std::size_t d = 0; d < len; ++d) {
        Succ next;
        for (const Stk& st : cur) {
            if (st.n == 0) { continue; }
            const Element* top = st.e[st.n - 1];
            if (!is_char_type(top)) { return false; }
            const auto match = match_char(top, cps[d]);
            if (!match.first) { continue; }
            Stk s2;
            s2.n = static_cast<std::uint8_t>(st.n - 1);
            std::memcpy(s2.e, st.e, s2.n * sizeof(Element*));
            if (!is_end_of_sequence(match.second)) { s2.e[s2.n++] = match.second; }
            advance_expand(*fast_rules_, s2, next);
        }
        cur.assign(next.s, next.s + next.n);
        has_empty = next.has_empty;
        if (cur.empty()) { return false; }
    }
    return has_empty || !cur.empty();
}

Producer::KSet Producer::initial_kset() {
    KSet init;
    for (int i = 0; i < root_.n; ++i) {
        bool in_c = false;
        for (const Stk& st : fast_c_set_) {
            if (st.n == root_.s[i].n &&
                std::memcmp(st.e, root_.s[i].e, st.n * sizeof(Element*)) == 0) {
                in_c = true;
                break;
            }
        }
        if (init.n >= kFastPos || root_.s[i].n > kFastStack) {
            fast_abort_ = true;
            KSet bad;
            return bad;
        }
        KPos& kp = init.p[init.n++];
        kp.stack.n = root_.s[i].n;
        std::memcpy(kp.stack.e, root_.s[i].e, kp.stack.n * sizeof(Element*));
        kp.k = in_c ? 0 : 1;
        kp.region = !in_c;
    }
    if (root_.has_empty) {
        init.has_empty = true;
        init.empty_k = 0;
    }
    return init;
}

std::uint8_t Producer::partial_threshold(const std::uint32_t* cps, std::size_t len,
                                         llama_partial_utf8 tail) {
    KSet cur = initial_kset();
    if (fast_abort_ || cur.n != root_.n) { return 0xFF; }
    for (std::size_t d = 0; d < len; ++d) {
        KSet next;
        for (int i = 0; i < cur.n; ++i) {
            const Element* top = cur.p[i].stack.e[cur.p[i].stack.n - 1];
            if (!is_char_type(top)) { continue; }
            const std::uint32_t cid = class_id_for_top(top);
            if (!classes_[cid].cls.accepts(cps[d])) { continue; }
            const auto match = match_char(top, first_accepted(classes_[cid].cls));
            Stk next_stack;
            next_stack.n = static_cast<std::uint8_t>(cur.p[i].stack.n - 1);
            std::memcpy(next_stack.e, cur.p[i].stack.e, next_stack.n * sizeof(Element*));
            if (!is_end_of_sequence(match.second)) { next_stack.e[next_stack.n++] = match.second; }
            expand_threshold(next_stack, cur.p[i].k, cur.p[i].region, next);
            if (fast_abort_) { return 0xFF; }
        }
        cur = next;
        if (cur.n == 0 && (!cur.has_empty || d + 1 < len)) { return 0xFF; }
    }
    std::uint8_t best = 0xFF;
    for (int i = 0; i < cur.n; ++i) {
        const Element* top = cur.p[i].stack.e[cur.p[i].stack.n - 1];
        if (!is_char_type(top)) { continue; }
        if (tail.n_remain == 0 || match_partial_char(top, tail)) {
            if (cur.p[i].k < best) { best = cur.p[i].k; }
        }
    }
    if (cur.has_empty && tail.n_remain == 0 && cur.empty_k < best) { best = cur.empty_k; }
    return best;
}

bool Producer::build_counter_table(const std::string& key, int p) {
    auto existing = tables_.find(key);
    if (existing != tables_.end() && existing->second.table_max >= 0 &&
        existing->second.table_max >= std::min(p, 128)) {
        return true;
    }
    const auto t0 = std::chrono::steady_clock::now();
    if (!build_body_table(fast_body_key_, p)) { return false; }
    if (std::getenv("TRIE_DEBUG_TAB")) { std::fprintf(stderr, "[key] body ok, corrections scan...\n"); std::fflush(stderr); }
    const BodyEntry& body = bodies_[fast_body_key_];
    const int table_max = std::min(p, std::min(128, body.max_p));

    // Correction scan: tokens whose suffix after a boundary is accepted by the
    // continuation (handoff), with fewer rounds than the region-only threshold.
    std::vector<const CharClass*> c_classes;
    for (const Stk& st : fast_c_set_) {
        if (st.n == 0) { continue; }
        const Element* top = st.e[st.n - 1];
        if (is_char_type(top)) { c_classes.push_back(&classes_[class_id_for_top(top)].cls); }
    }
    std::vector<std::pair<std::uint32_t, std::uint8_t>> corrections;  // euler idx, threshold
    for (std::size_t k = 0; k < euler_ids_.size(); ++k) {
        if (body.thr[k] == 0) { continue; }  // already accepted with zero rounds
        const std::uint32_t* path = path_data_.data() + euler_off_[k];
        const std::size_t len = euler_len_[k];
        // whole-token handoff (zero rounds consumed)
        std::uint8_t handoff = 0xFF;
        bool maybe = false;
        {
            const CharClass* first = nullptr;
            for (const CharClass* cls : c_classes) {
                if (cls->accepts(path[0])) { first = cls; break; }
            }
            if (first != nullptr && c_accepts_suffix(path, len)) { handoff = 0; }
        }
        if (handoff == 0xFF) {
            std::uint32_t node = 0;
            for (std::size_t i = 0; i < len; ++i) {
                node = trie_child(node, path[i]);
                if (node == kNoEdge) { break; }
                const std::uint8_t bk = body.node_boundary[node];
                if (bk == 0xFF || bk >= handoff) { continue; }
                if (i + 1 < len) {
                    bool first_ok = false;
                    for (const CharClass* cls : c_classes) {
                        if (cls->accepts(path[i + 1])) { first_ok = true; break; }
                    }
                    if (!first_ok) { continue; }
                }
                if (c_accepts_suffix(path + i + 1, len - (i + 1))) { handoff = bk; }
            }
        }
        (void)maybe;
        if (std::getenv("TRIE_DEBUG_TAB") != nullptr &&
            (euler_ids_[k] == 328 || euler_ids_[k] == 11611)) {
            std::fprintf(stderr, "[corr] id=328 len=%zu body_thr=%u handoff=%u path0=%u path1=%u\n",
                         len, (unsigned)body.thr[k], (unsigned)handoff,
                         len > 0 ? path[0] : 0, len > 1 ? path[1] : 0);
            std::uint32_t nd = 0;
            for (std::size_t i = 0; i < len; ++i) {
                nd = trie_child(nd, path[i]);
                std::fprintf(stderr, "   depth=%zu node=%u bk=%u\n", i + 1, nd,
                             nd == kNoEdge ? 255u : (unsigned)body.node_boundary[nd]);
                if (nd == kNoEdge) { break; }
            }
        }
        if (handoff != 0xFF && (body.thr[k] == 0xFF || handoff < body.thr[k])) {
            corrections.emplace_back(static_cast<std::uint32_t>(k), handoff);
        }
    }

    if (std::getenv("TRIE_DEBUG_TAB")) { std::fprintf(stderr, "[key] corrections=%zu, masks...\n", corrections.size()); std::fflush(stderr); }
    TableEntry& entry = tables_[key];
    entry.table_max = table_max;
    entry.masks = body.buckets;
    std::uint64_t correction_bits = 0;
    for (const auto& [k, thr] : corrections) {
        if (thr > table_max) { continue; }  // handoff outside the table range: not allowed in it
        const int id = euler_ids_[k];
        if (body.thr[k] != 0xFF && body.thr[k] <= table_max) {
            entry.masks[static_cast<std::size_t>(body.thr[k]) * row_words_ + (id >> 5)] &=
                ~(1U << (id & 31));
        }
        entry.masks[static_cast<std::size_t>(thr) * row_words_ + (id >> 5)] |= 1U << (id & 31);
        ++correction_bits;
    }
    for (int b = 1; b <= table_max; ++b) {
        std::uint32_t* dst = entry.masks.data() + static_cast<std::size_t>(b) * row_words_;
        const std::uint32_t* src = entry.masks.data() + static_cast<std::size_t>(b - 1) * row_words_;
        for (std::uint32_t w = 0; w < row_words_; ++w) { dst[w] |= src[w]; }
    }

    // Partial-end tokens: threshold each against the region + continuation exactly.
    if (std::getenv("TRIE_DEBUG_TAB")) { std::fprintf(stderr, "[key] masks done, partials...\n"); std::fflush(stderr); }
    entry.partial_thr.assign(partial_ids_.size(), 0xFF);
    for (std::size_t i = 0; i < partial_ids_.size(); ++i) {
        const PartialId& pp = partial_ids_[i];
        entry.partial_thr[i] = partial_threshold(path_data_.data() + pp.off, pp.len,
                                                 llama_partial_utf8{pp.value, pp.remain});
    }

    if (fast_abort_) {
        entry.table_max = -1;
        entry.masks.clear();
        ++stats.table_build_fail;
        return false;
    }
    stats.bucket_tokens += corrections.size();
    stats.correction_tokens += corrections.size();
    ++stats.table_builds;
    const auto t1 = std::chrono::steady_clock::now();
    stats.build_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    stats.table_masks_words += entry.masks.size();
    return true;
}

void Producer::shape_signature(const llama_partial_utf8& partial, std::uint64_t& h1,
                               std::uint64_t& h2) const {
    h1 = 1469598103934665603ULL;
    h2 = 1099511628211ULL;
    auto mix = [&h1, &h2](std::uint32_t v) {
        h1 ^= v;
        h1 *= 1099511628211ULL;
        h2 = (h2 ^ (v + 0x9e3779b9U)) * 1469598103934665603ULL;
    };
    for (int i = 0; i < root_.n; ++i) {
        const Stk& s = root_.s[i];
        mix(s.n);
        for (int j = 0; j < s.n; ++j) {
            mix(static_cast<std::uint32_t>(s.e[j]->type));
            mix(s.e[j]->value);
        }
    }
    mix(root_.has_empty ? 1U : 0U);
    mix(static_cast<std::uint32_t>(partial.n_remain));
    mix(static_cast<std::uint32_t>(partial.value));
    // Ablation switches change the produced row; fold them in.
    mix((bulk_enabled ? 1U : 0U) | ((tail_enabled ? 2U : 0U) << 8) |
        ((writes_enabled ? 4U : 0U) << 8) | ((fast_path_enabled ? 8U : 0U) << 8));
}

const Producer::ShapeEntry* Producer::shape_lookup(std::uint64_t h1, std::uint64_t h2) const {
    for (const ShapeEntry& e : shape_cache_) {
        if (e.h1 == h1 && e.h2 == h2) { return &e; }
    }
    return nullptr;
}

void Producer::shape_store(std::uint64_t h1, std::uint64_t h2) {
    if (shape_cache_.size() >= kShapeCacheMax) {
        shape_cache_.clear();
        ++stats.shape_evictions;
    }
    ShapeEntry e;
    e.h1 = h1;
    e.h2 = h2;
    e.row = *row_;
    shape_cache_.push_back(std::move(e));
}

void Producer::produce(const llama_grammar_rules& rules, const llama_grammar_stacks& stacks,
                       const llama_partial_utf8& partial, std::vector<std::uint32_t>& row) {
    row.assign(row_words_, 0);
    rules_ = &rules;
    row_ = &row;
    ++cap_gen_;
    ++prep_gen_;
    top_class_.clear();
    ++stats.fills;

    bool can_end = false;
    for (const auto& stack : stacks) {
        if (stack.empty()) { can_end = true; }
    }

    root_.n = 0;
    root_.has_empty = can_end;
    for (const auto& stack : stacks) {
        if (stack.empty()) { continue; }
        stk_set(root_.s[root_.n++], stack);
    }
    if (root_.n > kMaxPos) {
        ++stats.overflow_fills;
        throw std::runtime_error("trie producer: root position set larger than kMaxPos");
    }

    if (partial.n_remain != 0) {
        if (can_end && writes_enabled) {
            for (const int id : eog_ids_) { set_bit(id); }
        }
        ++stats.partial_states;
        // Only pieces whose first bytes are continuation bytes can decode validly
        // with a pending prefix; they are exactly the exceptions (standalone-invalid).
        for (const ExceptionId& ex : exceptions_) {
            std::vector<std::uint32_t> cps;
            llama_partial_utf8 tail{};
            if (!decode_utf8_bytes(exception_bytes_.data() + ex.off, ex.len, partial, cps, tail)) {
                continue;
            }
            ++stats.exception_judged;
            judge_linear(root_, cps.data(), cps.size(), tail, ex.id, false);
        }
        return;
    }

    // Producer-owned compiled-mask cache: the row is a pure function of the state
    // shape (root-position content + can_end + pending partial + ablation switches),
    // so a repeated shape serves its row by memcpy.
    shape_miss_store_ = false;
    if (shape_cache_enabled_ && std::getenv("TRIE_SHAPE_CACHE_OFF") == nullptr) {
        std::uint64_t h1 = 0, h2 = 0;
        shape_signature(partial, h1, h2);
        if (const ShapeEntry* hit = shape_lookup(h1, h2)) {
            ++stats.shape_hits;
            if (writes_enabled) {
                std::memcpy(row_->data(), hit->row.data(), row_words_ * sizeof(std::uint32_t));
            }
            return;
        }
        ++stats.shape_misses;
        shape_miss_h1_ = h1;
        shape_miss_h2_ = h2;
        // Store the row only for a repeated shape; first occurrences record the
        // signature only (no 31 KB copy for one-shot shapes).
        bool repeat = false;
        for (const auto& s : shape_seen_) {
            if (s.first == h1 && s.second == h2) { repeat = true; break; }
        }
        if (shape_seen_.size() < kShapeSeenMax) { shape_seen_.emplace_back(h1, h2); }
        else { shape_seen_.clear(); }
        shape_miss_store_ = repeat;
    }

    if (fast_path_enabled && writes_enabled) {
        std::string key;
        int p = 0;
        if (recognize_chain(rules, root_, key, p, fast_c_set_) &&
            build_counter_table(key, p)) {
            const TableEntry& entry = tables_[key];
            const int idx = std::min(p, entry.table_max);
            std::memcpy(row.data(), entry.masks.data() + static_cast<std::size_t>(idx) * row_words_,
                        static_cast<std::size_t>(row_words_) * sizeof(std::uint32_t));
            if (can_end) {
                for (const int id : eog_ids_) { set_bit(id); }
            }
            ++stats.fast_fills;
            if (std::getenv("TRIE_DEBUG_FILL") != nullptr) {
                const int dbg_id = 11611;
                std::uint32_t dbg_k = 0xFFFFFFFF;
                for (std::size_t k = 0; k < euler_ids_.size(); ++k) {
                    if (euler_ids_[k] == dbg_id) { dbg_k = static_cast<std::uint32_t>(k); break; }
                }
                const bool tbl_bit =
                    (entry.masks[static_cast<std::size_t>(idx) * row_words_ + (dbg_id >> 5)] >>
                     (dbg_id & 31)) & 1U;
                const bool row_bit = (row[dbg_id >> 5] >> (dbg_id & 31)) & 1U;
                const BodyEntry& db = bodies_[fast_body_key_];
                std::fprintf(stderr,
                             "[fastfill] id=%d p=%d idx=%d table_max=%d tbl_bit=%d row_bit=%d "
                             "k=%u body_thr=%u body_max_p=%d\n",
                             dbg_id, p, idx, entry.table_max, tbl_bit, row_bit, dbg_k,
                             dbg_k != 0xFFFFFFFF ? (unsigned)db.thr[dbg_k] : 255u, db.max_p);
                std::fflush(stderr);
            }
            if (tail_enabled) {
                for (std::size_t i = 0; i < partial_ids_.size(); ++i) {
                    const std::uint8_t thr = entry.partial_thr[i];
                    if (thr != 0xFF && static_cast<int>(thr) <= p) { set_bit(partial_ids_[i].id); }
                }
            }
            if (std::getenv("TRIE_DEBUG_ROW") != nullptr && (stats.fills == 15 || stats.fills == 16)) {
                for (int id = 0; id < token_count_; ++id) {
                    if ((row[static_cast<std::size_t>(id) >> 5U] >> (id & 31)) & 1U) {
                        std::fprintf(stderr, "%d ", id);
                    }
                }
                std::fprintf(stderr, "\n");
                std::fflush(stderr);
            }
            if (shape_miss_store_) { shape_store(shape_miss_h1_, shape_miss_h2_); }
            return;
        }
    }

    ++stats.fallback_fills;
    if (can_end && writes_enabled) {
        for (const int id : eog_ids_) { set_bit(id); }
    }

    std::vector<int> anc_pos;
    std::vector<int> anc_neg;
    visit(0, root_, 0, anc_pos, anc_neg);

    if (tail_enabled) {
        const std::uint64_t ttail = __rdtsc();
        for (const PartialId& p : partial_ids_) {
            judge_linear(root_, path_data_.data() + p.off, p.len,
                         llama_partial_utf8{p.value, p.remain}, p.id, true);
        }
        stats.cyc_tail += __rdtsc() - ttail;
    }
    if (std::getenv("TRIE_DEBUG_ROW") != nullptr && (stats.fills == 15 || stats.fills == 16)) {
        for (int id = 0; id < token_count_; ++id) {
            if ((row[static_cast<std::size_t>(id) >> 5U] >> (id & 31)) & 1U) {
                std::fprintf(stderr, "%d ", id);
            }
        }
        std::fprintf(stderr, "\n");
        std::fflush(stderr);
    }
    if (shape_miss_store_) { shape_store(shape_miss_h1_, shape_miss_h2_); }
}

}  // namespace trie_probe

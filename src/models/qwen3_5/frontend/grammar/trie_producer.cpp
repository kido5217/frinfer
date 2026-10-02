// Compiled token-trie mask producer implementation (wayfinder map #45, ticket #79).
//
// Faithful port of the verified prototype (tools/spike/compiled-mask/producer on the
// research/compiled-mask-producer branch): same algorithms, same caps, same error
// semantics. Deltas from the prototype, all covered by the D4 differential oracle:
//   - the counter-table key includes the frontier (the body positions the parallel
//     engine candidates sit at), so a fill at a deeper frontier (e.g. mid
//     escape-sequence in the body) gets its own table built from the fill's actual
//     position set instead of reusing a stale one;
//   - prepare() prebuilds the class structures and the fresh-round-frontier counter
//     tables at grammar compile time (D2a); on-demand builds remain as the backstop;
//   - produce() returns a span into producer-owned storage (scratch row or shape
//     store) instead of writing a caller vector.
#include "trie_producer.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace ninfer::models::qwen3_5::frontend::detail {

// ---------------------------------------------------------------------------
// Rule addressing and counter normalization
// ---------------------------------------------------------------------------

// Maps a stack element pointer back to (rule id, element index). Rule element vectors are
// separately allocated and never move after construction, so ranges are stable.
class RuleAddressIndex {
public:
    struct Range {
        const llama_grammar_element* begin;
        const llama_grammar_element* end;
        std::uint32_t rule;
    };

    void build(const llama_grammar_rules& rules) {
        ranges_.clear();
        ranges_.reserve(rules.size());
        for (std::size_t rule = 0; rule < rules.size(); ++rule) {
            ranges_.push_back({rules[rule].data(), rules[rule].data() + rules[rule].size(),
                               static_cast<std::uint32_t>(rule)});
        }
        std::sort(ranges_.begin(), ranges_.end(), [](const Range& left, const Range& right) {
            return std::less<const llama_grammar_element*>{}(left.begin, right.begin);
        });
    }

    [[nodiscard]] std::pair<std::uint32_t, std::uint32_t>
    locate(const llama_grammar_element* element) const {
        const auto found = std::upper_bound(
            ranges_.begin(), ranges_.end(), element,
            [](const llama_grammar_element* value, const Range& range) {
                return std::less<const llama_grammar_element*>{}(value, range.begin);
            });
        if (found == ranges_.begin()) {
            throw std::logic_error("grammar state references an element outside the rule set");
        }
        const Range& range = *(found - 1);
        if (element >= range.end) {
            throw std::logic_error("grammar state references an element outside the rule set");
        }
        return {range.rule, static_cast<std::uint32_t>(element - range.begin)};
    }

private:
    std::vector<Range> ranges_;
};

// Counter-normalized rule identity.
//
// A repetition expansion (`{m,n}`, `*`, `+`, `?`) generates a chain of exactly-one-round
// rules `f_p = body · f_{p-1}`, `f_1 = body`. Two chain positions p >= fold_limit have the
// same mask: a single accepted token consumes at most (max piece bytes) rounds, so beyond
// that limit every continuation is reachable from both. Positions below the limit (and the
// chain tail after the body) keep exact identities, because near the bound the mask is what
// enforces the repetition count — collapsing those would let a token overshoot `maxLength`.
//
// Frames are recognized structurally from the generated rules: first alternative
// `body [link]`, second alternative empty. A candidate link counts only when its target is
// itself a frame with the identical body (or the frame itself, the unbounded `S*` shape),
// which keeps hand-written look-alikes on the exact path unless they are genuine counters.
constexpr std::uint64_t kInfinitePosition = std::numeric_limits<std::uint64_t>::max();

RepetitionNormalization normalize_repetitions(const llama_grammar_rules& rules,
                                              std::uint32_t fold_limit) {
    const std::size_t count = rules.size();
    RepetitionNormalization result;
    result.fold_limit = fold_limit;
    result.canonical_rule.resize(count);
    for (std::size_t rule = 0; rule < count; ++rule) {
        result.canonical_rule[rule] = static_cast<std::uint32_t>(rule);
    }

    struct Frame {
        std::size_t alternative = 0; // first-alternative element count
        std::uint32_t link      = 0;
        bool link_candidate     = false;
        bool linked             = false;
    };

    std::vector<Frame> frames(count);
    std::vector<bool> is_frame(count, false);
    for (std::size_t rule = 0; rule < count; ++rule) {
        const llama_grammar_rule& elements = rules[rule];
        if (elements.size() < 3) { continue; }
        if (elements.back().type != LLAMA_GRETYPE_END) { continue; }
        if (elements[elements.size() - 2].type != LLAMA_GRETYPE_ALT) { continue; }
        std::size_t alternatives = 0;
        for (const llama_grammar_element& element : elements) {
            if (element.type == LLAMA_GRETYPE_ALT) { ++alternatives; }
        }
        if (alternatives != 1) { continue; }
        const std::size_t alternative = elements.size() - 2;
        if (alternative == 0) { continue; }
        Frame frame;
        frame.alternative = alternative;
        if (alternative >= 2 && elements[alternative - 1].type == LLAMA_GRETYPE_RULE_REF) {
            frame.link_candidate = true;
            frame.link           = elements[alternative - 1].value;
        }
        frame.linked   = frame.link_candidate;
        frames[rule]   = frame;
        is_frame[rule] = true;
    }

    auto body_size = [&](std::size_t rule) {
        const Frame& frame = frames[rule];
        return frame.alternative - (frame.linked ? 1 : 0);
    };
    auto body_equal = [&](std::size_t left, std::size_t right) {
        const std::size_t size = body_size(left);
        if (size != body_size(right)) { return false; }
        for (std::size_t index = 0; index < size; ++index) {
            const llama_grammar_element& a = rules[left][index];
            const llama_grammar_element& b = rules[right][index];
            if (a.type != b.type || a.value != b.value) { return false; }
        }
        return true;
    };

    // A linked reading is only kept when the link target is a frame with an identical body;
    // the check is monotone (linked only ever turns off), so it converges.
    bool changed = true;
    while (changed) {
        changed = false;
        for (std::size_t rule = 0; rule < count; ++rule) {
            Frame& frame = frames[rule];
            if (!frame.linked) { continue; }
            const std::size_t target = frame.link;
            if (target == rule) { continue; } // the unbounded S* self-recursion
            if (target >= count || !is_frame[target]) {
                frame.linked = false;
                changed      = true;
                continue;
            }
            if (!body_equal(rule, target)) {
                frame.linked = false;
                changed      = true;
            }
        }
    }

    // Chain positions: base frames are 1, every link adds one round; a link cycle means the
    // chain can stop at any count, i.e. the saturated position.
    std::vector<std::uint64_t> position(count, 0);
    std::vector<std::uint32_t> on_path(count, 0); // 1-based index in the current walk path
    for (std::size_t start = 0; start < count; ++start) {
        if (!is_frame[start] || position[start] != 0) { continue; }
        std::vector<std::size_t> path;
        std::size_t current = start;
        bool cycle          = false;
        bool base           = false;
        while (true) {
            if (position[current] != 0) { break; } // known below: back-fill the path
            if (on_path[current] != 0) {
                cycle = true;
                break;
            }
            on_path[current] = static_cast<std::uint32_t>(path.size()) + 1;
            path.push_back(current);
            if (!frames[current].linked) {
                base = true;
                break;
            }
            current = frames[current].link;
        }
        if (cycle) {
            // The cycle loops forever, and everything leading into it inherits that.
            for (const std::size_t frame : path) { position[frame] = kInfinitePosition; }
        } else if (base) {
            // path.back() is the base (position 1); the frames above it count up.
            for (std::size_t index = path.size(); index-- > 0;) {
                position[path[index]] = static_cast<std::uint64_t>(path.size() - index);
            }
        } else {
            std::uint64_t next = position[current] + 1;
            for (std::size_t index = path.size(); index-- > 0;) { position[path[index]] = next++; }
        }
        for (const std::size_t frame : path) { on_path[frame] = 0; }
    }

    // Canonical ids: frames at the same clamped position with an identical body share one
    // id (`fold_limit` absorbs every saturated position, including infinite self-recursion).
    std::unordered_map<std::string, std::uint32_t> representatives;
    std::uint32_t next_id = static_cast<std::uint32_t>(count);
    for (std::size_t rule = 0; rule < count; ++rule) {
        if (!is_frame[rule]) { continue; }
        const std::uint64_t raw_position = position[rule] == 0 ? 1 : position[rule];
        const std::uint32_t clamped = raw_position == kInfinitePosition || raw_position > fold_limit
                                          ? fold_limit
                                          : static_cast<std::uint32_t>(raw_position);
        std::string key;
        key.reserve(body_size(rule) * 8 + 4);
        for (std::size_t index = 0; index < body_size(rule); ++index) {
            const llama_grammar_element& element = rules[rule][index];
            for (int shift = 24; shift >= 0; shift -= 8) {
                key.push_back(static_cast<char>((element.type >> shift) & 0xff));
            }
            for (int shift = 24; shift >= 0; shift -= 8) {
                key.push_back(static_cast<char>((element.value >> shift) & 0xff));
            }
        }
        for (int shift = 24; shift >= 0; shift -= 8) {
            key.push_back(static_cast<char>((clamped >> shift) & 0xff));
        }
        const auto [found, inserted] = representatives.try_emplace(std::move(key), next_id);
        if (inserted) { ++next_id; }
        result.canonical_rule[rule] = found->second;
    }
    return result;
}

namespace {

// UTF-8 and character-class helpers shared across this translation unit only.
[[nodiscard]] bool decode_utf8_bytes(const std::uint8_t* bytes, std::size_t len,
                                     const llama_partial_utf8& start,
                                     std::vector<std::uint32_t>& cps, llama_partial_utf8& end);
[[nodiscard]] CharClass class_from_element(const Element* pos);

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

} // namespace

bool CharClass::accepts(std::uint32_t cp) const noexcept {
    return in_ranges(ranges, cp) != negated;
}

bool CharClass::operator==(const CharClass& other) const noexcept {
    return negated == other.negated && ranges == other.ranges;
}

namespace {

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
        value                   = first_byte & mask;
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

} // namespace

// ---------------------------------------------------------------------------
// Producer
// ---------------------------------------------------------------------------

Producer::Producer(bool shape_cache) : shape_cache_enabled(shape_cache) {}

void Producer::build(const GrammarVocabulary& vocabulary) {
    const auto started = std::chrono::steady_clock::now();
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
    trie_depth_ = 0;
    std::size_t max_piece_bytes = 0;

    for (int id = 0; id < static_cast<int>(token_count_); ++id) {
        if (vocabulary.is_eog(id)) {
            eog_ids_.push_back(id);
            continue;
        }
        const std::string_view piece = vocabulary.piece(id);
        max_piece_bytes = std::max(max_piece_bytes, piece.size());
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
            if (node.depth > trie_depth_) { trie_depth_ = node.depth; }
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

    // One walk scratch per codepoint depth: the deepest matching path is the longest
    // piece, and pre-sizing keeps references stable across the recursive descent.
    visit_scratch_.resize(static_cast<std::size_t>(trie_depth_) + 1);

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
    // The fold limit for counter normalization: a single token consumes at most
    // (max piece bytes) counter rounds, so beyond this every continuation is reachable.
    fold_limit_ = static_cast<std::uint32_t>(max_piece_bytes) + 2;
    const auto finished = std::chrono::steady_clock::now();
    stats.build_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(finished - started).count());
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
    bytes += visit_scratch_.capacity() * sizeof(VisitScratch);
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
    row_[static_cast<std::size_t>(id) >> 5U] |= 1U << (static_cast<unsigned>(id) & 31U);
    ++stats.bits_set;
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
        for (std::uint32_t k = nodes_[node].sub_begin; k < nodes_[node].sub_end; ++k) {
            if (skip_excluded && cache.excl_bitmap[k] != 0) { continue; }
            set_bit(euler_ids_[k]);
        }
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
    if (depth >= visit_scratch_.size()) {
        throw std::logic_error("grammar trie walk exceeded its recorded depth");
    }
    VisitScratch& scratch = visit_scratch_[depth];
    scratch.nsigs = 0;
    PosSet& local = scratch.local;

    const Node& nd = nodes_[node];
    if (nd.own_begin != kNoEdge) {
        for (std::uint32_t k = nd.own_begin; k < nd.own_end; ++k) { set_bit(euler_ids_[k]); }
    }

    if (try_bulk(node, pos, depth, nullptr)) { return; }

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

    for (std::uint32_t e = nd.first_edge; e != kNoEdge; e = edges_[e].next) {
        ++stats.edges_visited;
        const std::uint32_t cp = edges_[e].cp;
        std::uint64_t sig = 0;
        for (int i = 0; i < pos.n; ++i) {
            if (prepared[i]->is_char && classes_[prepared[i]->class_id].cls.accepts(cp)) {
                sig |= 1ULL << i;
            }
        }
        PosSet* next = &local;
        if (sig == 0) {
            local.n = 0;
            local.has_empty = false;
        } else {
            bool found = false;
            for (int i = 0; i < scratch.nsigs; ++i) {
                if (scratch.sigs[i] == sig) {
                    next = &scratch.sets[i];
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
                if (scratch.nsigs < kVisitSigCache) {
                    scratch.sigs[scratch.nsigs] = sig;
                    pos_copy(scratch.sets[scratch.nsigs], local);
                    next = &scratch.sets[scratch.nsigs];
                    ++scratch.nsigs;
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
    for (int i = 0; i < pos.n; ++i) {
        const Stk& s = pos.s[i];
        if (s.n < 2) { continue; }
        const Element* ref = s.e[s.n - 2];
        if (ref->type != LLAMA_GRETYPE_RULE_REF) { continue; }
        // Region stacks share this exact ref below a char top; every other stack
        // must be exactly the expansion of the continuation below the ref.
        int region_n = 0;
        int region_idx[32] = {};
        for (int b = 0; b < pos.n; ++b) {
            const Stk& t = pos.s[b];
            if (t.n >= 2 && t.e[t.n - 2] == ref && is_char_type(t.e[t.n - 1])) {
                if (region_n < 32) { region_idx[region_n] = b; }
                ++region_n;
            }
        }
        if (region_n == 0) { continue; }
        Stk c_raw;
        c_raw.n = static_cast<std::uint8_t>(s.n - 2);
        std::memcpy(c_raw.e, s.e, c_raw.n * sizeof(Element*));
        Succ c_expanded;
        advance_expand(rules, c_raw, c_expanded);
        if (c_expanded.n != pos.n - region_n) { continue; }
        if (c_expanded.has_empty != pos.has_empty) { continue; }
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
        if (!c_ok) { continue; }

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
        if (!walk_ok || frames == 0) { continue; }
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

        // Key: C_raw located by (rule, index) + body elements by (type, value) +
        // the frontier (the region tops, by (rule, index)). The frontier is part
        // of the key because the table's round thresholds are only valid from the
        // positions it was built at: a fill at a deeper frontier (mid-escape in
        // the body, a different set of parallel candidates) must not reuse the
        // fresh-round-frontier table.
        auto locate = [&](const Element* el) -> std::pair<std::uint32_t, std::uint32_t> {
            for (std::uint32_t r = 0; r < rules.size(); ++r) {
                const auto& rr = rules[r];
                if (!rr.empty() && el >= rr.data() && el < rr.data() + rr.size()) {
                    return {r, static_cast<std::uint32_t>(el - rr.data())};
                }
            }
            return {0xFFFFFFFFU, 0};
        };
        auto append_word = [&key](std::uint32_t v) {
            for (int sh = 24; sh >= 0; sh -= 8) { key.push_back(static_cast<char>((v >> sh) & 0xFF)); }
        };
        key.clear();
        for (int j = 0; j < c_raw.n; ++j) {
            const auto [r, idx] = locate(c_raw.e[j]);
            if (r == 0xFFFFFFFFU) { break; }
            append_word(r);
            append_word(idx);
        }
        key.push_back('|');
        for (std::size_t j = 0; j < body0_len; ++j) {
            append_word(static_cast<std::uint32_t>((*body0)[j].type));
            append_word((*body0)[j].value);
        }
        key.push_back('#');
        std::vector<std::pair<std::uint32_t, std::uint32_t>> tops;
        for (int b = 0; b < region_n && b < 32; ++b) {
            const Stk& t = pos.s[region_idx[b]];
            const auto [r, idx] = locate(t.e[t.n - 1]);
            if (r == 0xFFFFFFFFU) { return false; }  // cannot key the frontier: fall back
            tops.emplace_back(r, idx);
        }
        std::sort(tops.begin(), tops.end());
        tops.erase(std::unique(tops.begin(), tops.end()), tops.end());
        for (const auto& t : tops) {
            append_word(t.first);
            append_word(t.second);
        }
        // The body (region-only) table is keyed by the FULL state shape (c_raw + body +
        // frontier), not the body alone: its round thresholds depend on the current
        // region position, and reusing a body table built at one frontier for another
        // (e.g. a fresh pair boundary vs a mid-value position) yields stale thresholds.
        fast_body_key_ = key;
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
    if (!fast_abort_) { threshold_visit(0, init); }
    fast_body_only_ = false;

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
        body.buckets[static_cast<std::size_t>(m) * row_words_ + (static_cast<std::size_t>(id) >> 5)] |=
            1U << (static_cast<unsigned>(id) & 31);
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
        // Empty with has_empty is the continuation having consumed the suffix to its
        // end (legitimate accept); empty without it is a dead end.
        if (cur.empty() && !has_empty) { return false; }
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

bool Producer::build_counter_table(const std::string& key, int p, bool on_demand) {
    auto existing = tables_.find(key);
    if (existing != tables_.end() && existing->second.table_max >= 0 &&
        existing->second.table_max >= std::min(p, 128)) {
        return true;
    }
    const auto t0 = std::chrono::steady_clock::now();
    if (!build_body_table(fast_body_key_, p)) { return false; }
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
        if (handoff != 0xFF && (body.thr[k] == 0xFF || handoff < body.thr[k])) {
            corrections.emplace_back(static_cast<std::uint32_t>(k), handoff);
        }
    }

    TableEntry& entry = tables_[key];
    entry.table_max = table_max;
    entry.masks = body.buckets;
    for (const auto& [k, thr] : corrections) {
        if (thr > table_max) { continue; }  // handoff outside the table range: not allowed in it
        const int id = euler_ids_[k];
        if (body.thr[k] != 0xFF && body.thr[k] <= table_max) {
            entry.masks[static_cast<std::size_t>(body.thr[k]) * row_words_ +
                        (static_cast<std::size_t>(id) >> 5)] &= ~(1U << (static_cast<unsigned>(id) & 31));
        }
        entry.masks[static_cast<std::size_t>(thr) * row_words_ +
                    (static_cast<std::size_t>(id) >> 5)] |= 1U << (static_cast<unsigned>(id) & 31);
    }
    for (int b = 1; b <= table_max; ++b) {
        std::uint32_t* dst = entry.masks.data() + static_cast<std::size_t>(b) * row_words_;
        const std::uint32_t* src = entry.masks.data() + static_cast<std::size_t>(b - 1) * row_words_;
        for (std::uint32_t w = 0; w < row_words_; ++w) { dst[w] |= src[w]; }
    }

    // Partial-end tokens: threshold each against the region + continuation exactly.
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
    if (on_demand) { ++stats.table_builds_on_demand; } else { ++stats.table_builds; }
    const auto t1 = std::chrono::steady_clock::now();
    stats.prepare_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    stats.table_masks_words += entry.masks.size();
    return true;
}

void Producer::prepare_classes(const llama_grammar_rules& rules) {
    for (const llama_grammar_rule& rule : rules) {
        for (std::size_t j = 0; j < rule.size(); ++j) {
            const Element& element = rule[j];
            if (element.type != LLAMA_GRETYPE_CHAR && element.type != LLAMA_GRETYPE_CHAR_NOT &&
                element.type != LLAMA_GRETYPE_CHAR_ANY) {
                continue;
            }
            const std::uint32_t cid = class_id(class_from_element(&element));
            ensure_class_structures(cid);
        }
    }
}

// The counter-shape enumeration (D2a). The engine's raw stack is a sequence of
// (rule, index) positions - one current element per nested rule context - so a
// state space walk over positions mirrors the engine exactly: expanding the top
// position (r, i) when rules[r][i] is a RULE_REF replaces it with the
// continuation (r, i+1) and the target rule's start (r, i+1 below (g, 0)); ALT
// jumps to the next alternative; END pops the position. A leaf position (a
// char/token top) whose second-from-top position is a RULE_REF to a frame is a
// fast-path ref occurrence: the ref is the pending frame link, the top is the
// current body position, and the positions below are the continuation. All
// leaves of one occurrence (its parallel body-frontier candidates) share one
// counter table, keyed exactly as recognize_chain keys it - (C placement, body,
// frontier) - so the prebuilt tables are the ones the serving walk will use.
void Producer::prepare_counter_shapes(const llama_grammar_rules& rules, std::uint32_t root_rule) {
    if (rules.empty() || root_rule >= rules.size()) { return; }

    struct Pos {
        std::uint32_t rule;
        std::uint32_t index;
    };
    using Context = std::vector<Pos>;

    struct Occurrence {
        const Element* ref = nullptr;
        Context c_raw;  // the context below the ref
        std::vector<std::vector<Pos>> leaves;  // the parallel leaf contexts (full contexts)
    };

    auto encode = [](const Context& context) {
        std::string key;
        for (const Pos& pos : context) {
            for (int sh = 24; sh >= 0; sh -= 8) {
                key.push_back(static_cast<char>((pos.rule >> sh) & 0xFF));
                key.push_back(static_cast<char>((pos.index >> sh) & 0xFF));
            }
        }
        return key;
    };
    // Collect the leaves, grouped by (ref element, c_raw context). A context is
    // visited once (recursive rule graphs cycle without the dedupe). The walk models
    // the engine's full stack transition system - REF expansion, ALT/END continuation,
    // and char/token CONSUMPTION (advance past the char group, popping the frame when
    // the alternative ends) - so it reaches the counter fresh-round frontiers that sit
    // several literals deep in the rule graph.
    std::set<std::string> seen;
    std::map<const Element*, std::map<std::string, Occurrence>> occurrences;
    RuleAddressIndex address_index;
    address_index.build(rules);
    std::function<void(const Context&)> walk =
        [&walk, &rules, &occurrences, &seen, &encode, &address_index](const Context& context) {
            const std::string encoded = encode(context);
            if (!seen.insert(encoded).second) { return; }
            if (context.empty()) { return; }
        if (context.size() > static_cast<std::size_t>(kMaxStack)) { return; }
        const Pos& top = context.back();
        if (top.index >= rules[top.rule].size()) { return; }
        const Element& element = rules[top.rule][top.index];
        switch (element.type) {
            case LLAMA_GRETYPE_CHAR:
            case LLAMA_GRETYPE_CHAR_NOT:
            case LLAMA_GRETYPE_CHAR_ANY:
            case LLAMA_GRETYPE_TOKEN:
            case LLAMA_GRETYPE_TOKEN_NOT: {
                if (context.size() >= 3) {
                    const Pos& ref_pos = context[context.size() - 2];
                    const Element& ref_element = rules[ref_pos.rule][ref_pos.index];
                    if (ref_element.type == LLAMA_GRETYPE_RULE_REF) {
                        Context c_raw(context.begin(), context.end() - 2);
                        std::string c_key = encode(c_raw);
                        Occurrence& occ =
                            occurrences[&ref_element][std::move(c_key)];
                        occ.ref = &ref_element;
                        occ.c_raw = c_raw;
                        occ.leaves.push_back(context);
                    }
                }
                // Consumption transition: the engine advances past the char group (or the
                // token element); when the alternative ends (END/ALT) the frame pops to the
                // continuation below, otherwise the new element becomes the top.
                const Element* after = nullptr;
                if (element.type == LLAMA_GRETYPE_TOKEN ||
                    element.type == LLAMA_GRETYPE_TOKEN_NOT) {
                    if (top.index + 1 < rules[top.rule].size()) {
                        after = &rules[top.rule][top.index + 1];
                    }
                } else {
                    after = match_char(&element, 0).second;
                }
                if (after != nullptr && !is_end_of_sequence(after)) {
                    const auto [nr, ni] = address_index.locate(after);
                    Context next = context;
                    next.back().rule = nr;
                    next.back().index = ni;
                    walk(next);
                } else {
                    Context next = context;
                    next.pop_back();
                    if (!next.empty()) { walk(next); }
                }
                return;
            }
            case LLAMA_GRETYPE_CHAR_RNG_UPPER:
            case LLAMA_GRETYPE_CHAR_ALT:
                return;
            case LLAMA_GRETYPE_END: {
                Context next = context;
                next.pop_back();
                walk(next);
                return;
            }
            case LLAMA_GRETYPE_ALT: {
                Context next = context;
                next.back().index += 1;
                walk(next);
                return;
            }
            case LLAMA_GRETYPE_RULE_REF: {
                Context next = context;
                next.pop_back();
                if (top.index + 1 < rules[top.rule].size()) {
                    next.push_back({top.rule, top.index + 1});
                }
                next.push_back({element.value, 0});
                walk(next);
                return;
            }
        }
    };

    walk({{root_rule, 0}});

    // Resolve frames once (memoized across all occurrences).
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

    auto append_word = [](std::string& key, std::uint32_t v) {
        for (int sh = 24; sh >= 0; sh -= 8) { key.push_back(static_cast<char>((v >> sh) & 0xFF)); }
    };
    std::set<std::string> built;  // keys prebuilt (deduped across occurrences)

    for (auto& [ref_element, by_c_raw] : occurrences) {
        for (auto& [c_key, occ] : by_c_raw) {
            (void)c_key;
            // The occurrence is only a fast-path shape when every leaf top is a
            // char (token terminals anywhere reject the state) and the body
            // frontier tops are locate-able.
            bool tokens = false;
            for (const Context& leaf : occ.leaves) {
                const Element& e = rules[leaf.back().rule][leaf.back().index];
                if (e.type == LLAMA_GRETYPE_TOKEN || e.type == LLAMA_GRETYPE_TOKEN_NOT) {
                    tokens = true;
                    break;
                }
            }
            if (tokens) { continue; }
            // Token terminals anywhere in the positions reject the state.
            for (const Context& leaf : occ.leaves) {
                for (const Pos& p : leaf) {
                    const Element& e = rules[p.rule][p.index];
                    if (e.type == LLAMA_GRETYPE_TOKEN || e.type == LLAMA_GRETYPE_TOKEN_NOT) {
                        tokens = true;
                        break;
                    }
                }
                if (tokens) { break; }
            }
            if (tokens) { continue; }

            // Chain validation from the ref (same walk as recognition).
            const Element* cur = occ.ref;
            std::size_t frames = 0;
            const llama_grammar_rule* body0 = nullptr;
            std::size_t body0_len = 0;
            bool unbounded = false;
            bool walk_ok = true;
            std::vector<const Element*> chain;
            while (true) {
                bool seen_ref = false;
                for (const Element* r : chain) {
                    if (r == cur) { seen_ref = true; break; }
                }
                if (seen_ref) { unbounded = true; break; }
                if (cur->value >= rules.size() || !resolver.resolve(cur->value)) { walk_ok = false; break; }
                chain.push_back(cur);
                const llama_grammar_rule& rule = rules[cur->value];
                const std::size_t blen = resolver.blen[cur->value];
                if (body0 == nullptr) {
                    body0 = &rule;
                    body0_len = blen;
                } else if (body0_len != blen ||
                           !Resolver::elements_equal(*body0, 0, rule, 0, blen)) {
                    walk_ok = false;
                    break;
                }
                ++frames;
                if (resolver.link[cur->value]) {
                    cur = &rule[blen];
                } else {
                    break;
                }
            }
            if (!walk_ok || frames == 0) { continue; }
            std::sort(chain.begin(), chain.end());
            const int p = unbounded ? (1 << 20) : static_cast<int>(frames + 1);

            // C must not contain any chain ref.
            std::vector<const Element*> c_raw_elements;
            for (const Pos& p : occ.c_raw) {
                c_raw_elements.push_back(&rules[p.rule][p.index]);
            }
            Stk c_stk;
            stk_set(c_stk, c_raw_elements);
            Succ c_expanded;
            try {
                advance_expand(rules, c_stk, c_expanded);
            } catch (const std::runtime_error&) {
                continue;  // deeper than the producer caps: unreachable at runtime
            }
            bool ref_in_c = false;
            for (int a = 0; a < c_expanded.n && !ref_in_c; ++a) {
                for (int j = 0; j < c_expanded.s[a].n && !ref_in_c; ++j) {
                    if (std::binary_search(chain.begin(), chain.end(), c_expanded.s[a].e[j])) {
                        ref_in_c = true;
                    }
                }
            }
            if (ref_in_c) { continue; }

            // Key: (C placement, body, frontier) - byte-identical to recognize_chain's.
            std::string key;
            for (const Pos& p : occ.c_raw) {
                append_word(key, p.rule);
                append_word(key, p.index);
            }
            key.push_back('|');
            for (std::size_t j = 0; j < body0_len; ++j) {
                append_word(key, static_cast<std::uint32_t>((*body0)[j].type));
                append_word(key, (*body0)[j].value);
            }
            key.push_back('#');
            std::vector<std::pair<std::uint32_t, std::uint32_t>> tops;
            for (const Context& leaf : occ.leaves) {
                const Pos& t = leaf.back();
                tops.emplace_back(t.rule, t.index);
            }
            std::sort(tops.begin(), tops.end());
            tops.erase(std::unique(tops.begin(), tops.end()), tops.end());
            for (const auto& t : tops) {
                append_word(key, t.first);
                append_word(key, t.second);
            }
            if (!built.insert(key).second) { continue; }

            // Reconstruct the fill's root position set: the continuation positions
            // plus the occurrence's region stacks (the parallel body-frontier
            // candidates), exactly as the engine's raw stacks look at the ref.
            std::vector<Stk> c_set;
            c_set.reserve(c_expanded.n + 1);
            for (int a = 0; a < c_expanded.n; ++a) {
                Stk st;
                st.n = c_expanded.s[a].n;
                std::memcpy(st.e, c_expanded.s[a].e, st.n * sizeof(Element*));
                c_set.push_back(std::move(st));
            }
            if (c_expanded.has_empty) { c_set.push_back(Stk{}); }

            PosSet root_poset;
            root_poset.has_empty = c_expanded.has_empty;
            try {
                for (const Stk& st : c_set) {
                    if (st.n > 0) { pos_add(root_poset, st); }
                }
                for (const Context& leaf : occ.leaves) {
                    std::vector<const Element*> elements;
                    for (const Pos& p : leaf) {
                        elements.push_back(&rules[p.rule][p.index]);
                    }
                    Stk st;
                    stk_set(st, elements);
                    pos_add(root_poset, st);
                }
                if (root_poset.n == 0) { continue; }
            } catch (const std::runtime_error&) {
                continue;  // position-set cap: unreachable at runtime
            }

            // Build the table with the fill-state members set, exactly as an
            // on-demand build would for a fill at this state. The body table is
            // keyed by the full state shape (same as recognize_chain).
            rules_ = &rules;
            root_ = root_poset;
            fast_c_set_ = c_set;
            fast_body_key_ = key;
            if (build_counter_table(key, p, /*on_demand=*/false)) {
                ++stats.prebuilt_shapes;
            }
        }
    }
}

void Producer::prepare(const llama_grammar_rules& rules, std::uint32_t root_rule) {
    // The canonical rule identity for the shape signature: distinct per counter
    // position below the fold limit, shared above it (where masks are provably
    // equal). Without this the signature is a raw (type, value) mix that collides
    // across the invariant single-element stacks of character-class counters.
    canonical_rule_ = normalize_repetitions(rules, fold_limit_).canonical_rule;
    prepare_classes(rules);
    prepare_counter_shapes(rules, root_rule);
}

void Producer::shape_signature(const llama_partial_utf8& partial, std::uint64_t& h1,
                               std::uint64_t& h2) const {
    // The signature is the canonical state content - (canonical rule, element index)
    // per stack element, stacks order-insensitive, partial canonicalized out of the
    // dead value - exactly the abstraction the row is a pure function of. A raw
    // (type, value) mix is NOT sound: character-class counters keep an invariant
    // single-element (type, value) stack across positions while the mask changes at
    // the bound (only the element's rule index moves).
    constexpr std::uint32_t kStackSeparator = 0xFFFFFFFFU;
    h1 = 1469598103934665603ULL;
    h2 = 0xCBF29CE484222325ULL;
    auto mix = [&h1, &h2](std::uint32_t v) {
        h1 ^= v;
        h1 *= 1099511628211ULL;
        h2 = (h2 ^ (v + 0x9E3779B97F4A7C15ULL)) * 0xC2B2AE3D27D4EB4FULL;
    };
    mix(partial.n_remain == 0 ? 0U : partial.value);
    mix(static_cast<std::uint32_t>(partial.n_remain));
    RuleAddressIndex index;
    index.build(*rules_);
    std::vector<std::vector<std::uint32_t>> encoded;
    encoded.reserve(root_.n);
    for (int i = 0; i < root_.n; ++i) {
        std::vector<std::uint32_t> words;
        words.reserve(static_cast<std::size_t>(root_.s[i].n) * 2 + 1);
        for (int j = 0; j < root_.s[i].n; ++j) {
            const auto [rule, element_index] = index.locate(root_.s[i].e[j]);
            words.push_back(canonical_rule_[rule]);
            words.push_back(element_index);
        }
        words.push_back(kStackSeparator);
        encoded.push_back(std::move(words));
    }
    std::sort(encoded.begin(), encoded.end());
    for (const auto& words : encoded) {
        for (const std::uint32_t w : words) { mix(w); }
    }
    mix(root_.has_empty ? 1U : 0U);
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
    e.row = row_;
    shape_cache_.push_back(std::move(e));
    ++stats.shape_rows_stored;
}

std::span<const std::uint32_t> Producer::produce(const llama_grammar_rules& rules,
                                                 const llama_grammar_stacks& stacks,
                                                 const llama_partial_utf8& partial) {
    row_.assign(row_words_, 0);
    rules_ = &rules;
    ++cap_gen_;
    ++prep_gen_;
    top_class_.clear();
    const auto started = std::chrono::steady_clock::now();
    ++stats.fills;

    bool can_end = false;
    for (const auto& stack : stacks) {
        if (stack.empty()) { can_end = true; }
    }

    root_.n = 0;
    root_.has_empty = can_end;
    for (const auto& stack : stacks) {
        if (stack.empty()) { continue; }
        if (root_.n >= kMaxPos) {
            // Checked before the write: the position set is a fixed array, so a width
            // beyond kMaxPos must reject (fail-closed) rather than overflow it.
            ++stats.overflow_fills;
            throw std::runtime_error("trie producer: more than kMaxPos parallel stacks");
        }
        stk_set(root_.s[root_.n++], stack);
    }

    if (partial.n_remain != 0) {
        if (can_end) {
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
        const auto finished = std::chrono::steady_clock::now();
        stats.fill_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(finished - started).count());
        return std::span<const std::uint32_t>(row_);
    }

    // Producer-owned compiled-mask cache: the row is a pure function of the state
    // shape (root-position content + can_end + pending partial), so a repeated
    // shape serves its row by pointer.
    shape_miss_store_ = false;
    if (shape_cache_enabled) {
        std::uint64_t h1 = 0, h2 = 0;
        shape_signature(partial, h1, h2);
        if (const ShapeEntry* hit = shape_lookup(h1, h2)) {
            ++stats.shape_hits;
            return std::span<const std::uint32_t>(hit->row);
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

    {
        std::string key;
        int p = 0;
        if (recognize_chain(rules, root_, key, p, fast_c_set_) &&
            build_counter_table(key, p, /*on_demand=*/true)) {
            const TableEntry& entry = tables_[key];
            const int idx = std::min(p, entry.table_max);
            std::memcpy(row_.data(), entry.masks.data() + static_cast<std::size_t>(idx) * row_words_,
                        static_cast<std::size_t>(row_words_) * sizeof(std::uint32_t));
            if (can_end) {
                for (const int id : eog_ids_) { set_bit(id); }
            }
            ++stats.fast_fills;
            for (std::size_t i = 0; i < partial_ids_.size(); ++i) {
                const std::uint8_t thr = entry.partial_thr[i];
                if (thr != 0xFF && static_cast<int>(thr) <= p) { set_bit(partial_ids_[i].id); }
            }
            const auto finished = std::chrono::steady_clock::now();
            stats.fill_ns += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(finished - started).count());
            if (shape_miss_store_) { shape_store(shape_miss_h1_, shape_miss_h2_); }
            if (shape_miss_store_) {
                // The stored copy is the last one pushed; point into it.
                return std::span<const std::uint32_t>(shape_cache_.back().row);
            }
            return std::span<const std::uint32_t>(row_);
        }
    }

    ++stats.fallback_fills;
    if (can_end) {
        for (const int id : eog_ids_) { set_bit(id); }
    }

    std::vector<int> anc_pos;
    std::vector<int> anc_neg;
    visit(0, root_, 0, anc_pos, anc_neg);

    for (const PartialId& p : partial_ids_) {
        judge_linear(root_, path_data_.data() + p.off, p.len,
                     llama_partial_utf8{p.value, p.remain}, p.id, true);
    }
    const auto finished = std::chrono::steady_clock::now();
    stats.fill_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(finished - started).count());
    if (shape_miss_store_) { shape_store(shape_miss_h1_, shape_miss_h2_); }
    if (shape_miss_store_) {
        return std::span<const std::uint32_t>(shape_cache_.back().row);
    }
    return std::span<const std::uint32_t>(row_);
}

}  // namespace ninfer::models::qwen3_5::frontend::detail

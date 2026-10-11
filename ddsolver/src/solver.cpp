// A double-dummy solver for contract bridge, written from scratch.
//
// Double dummy = every hand is visible and both sides play perfectly. The
// question is: from this position, how many more tricks does North-South
// take? That is a two-player, zero-sum, perfect-information game, so it can
// be solved exactly by game-tree search. A full deal has an astronomically
// large tree; the techniques below are what make it tractable:
//
//   1. Bitmask state      each hand is four 13-bit suit masks; playing a card
//                         is one AND, following suit is one test.
//   2. Yes/no search      instead of "how many tricks?", ask "can N-S take
//                         at least t tricks?" (a null-window search) and find
//                         the exact count by stepping t from a guess (or by
//                         binary search when there is no guess).
//   3. Equivalent cards   Q and J in one hand with the K already gone are the
//                         same card for play purposes: search only one.
//   4. Transposition      positions at the start of a trick are cached with a
//      table (relative    lower and an upper bound. The key uses RELATIVE
//      ranks)             ranks: once the A, K and Q of a suit are gone the J
//                         plays exactly like an ace.
//   5. Partition search   a cached result also records WHICH cards decided it.
//                         Low cards that never won a trick by rank become
//                         interchangeable "small cards", so one entry answers
//                         every position that differs only in those cards.
//   6. Cheap bounds       quick tricks (top cards the side on lead can cash) and
//                         top trumps (each trump above all the opponents' trumps
//                         is a sure trick) often settle a position without
//                         searching it.
//   7. Move ordering      try the likely-best card first (cash winners, win
//                         cheaply, play low when partner is winning, ...),
//                         so alpha-beta cutoffs happen early.
//
// Each technique can be switched off with a flag so its effect can be measured
// (benchmark/dds_scratch_benchmark.py). Plain minimax alpha-beta (F_MINIMAX)
// is kept as the baseline.
//
// Conventions: players 0=N 1=E 2=S 3=W (N-S are the even players); suits
// 0=S 1=H 2=D 3=C; trump 0..3, or 4 = no trump; rank 0=2 ... 12=A.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#define DDS_API extern "C" __declspec(dllexport)
#else
#define DDS_API extern "C" __attribute__((visibility("default")))
#endif

typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

enum {
    F_TT = 1,          // transposition table
    F_RELRANK = 2,     // relative-rank keys (needs F_TT)
    F_EQUIV = 4,       // equivalent-card pruning
    F_QUICK = 8,       // quick-trick and top-trump bounds
    F_ORDER = 16,      // move-ordering heuristics
    F_MINIMAX = 32,    // baseline: plain minimax alpha-beta on trick counts
    F_PARTITION = 64,  // table entries ignore low cards that never mattered (needs F_TT)
    F_ALL = F_TT | F_RELRANK | F_EQUIV | F_QUICK | F_ORDER | F_PARTITION,
};

static inline int popc(unsigned x) { return __builtin_popcount(x); }
static inline int top_bit(unsigned x) { return 31 - __builtin_clz(x); }  // x != 0

static inline u64 mix(u64 x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33; return x;
}

static uint32_t g_gen = 1;  // bumping it empties every table at once

// ---------------------------------------------------------------- simple TT
// Used without F_PARTITION: one key fixes every remaining card (exact ranks, or
// relative ranks with F_RELRANK).
struct TTEntry {
    u64 k0, k1, k2;
    int8_t lb, ub;      // bounds on N-S tricks from this trick-start position
    uint8_t best_suit;  // suit of the best lead found (255 = none)
    uint8_t depth;      // tricks left (bigger subtrees are kept in preference)
    u32 gen;
};
static const int TT_BITS = 21;  // 2M entries x 32 bytes = 64 MB
static TTEntry* g_tt = nullptr;

// ---------------------------------------------------------------- partition TT
// An entry fixes the suit lengths of every hand, plus the owners of only the top
// k cards of each suit (k chosen per suit when the entry is stored). Entries are
// found in two steps: positions with the same lengths, leader and trump share a
// bucket that lists which k-combinations have been stored there; for each one,
// the position's own top-k pattern is looked up directly in a hash table.
struct PEntry {
    u64 lens;          // 16 x 4 bits: length of each suit in each hand
    u32 pat[4];        // per suit: owners of the top k cards, 2 bits each, top card first
    u16 ks;            // k of each suit, 4 bits each
    uint8_t lt;        // leader | trump << 2
    int8_t lb, ub;     // bounds on N-S tricks from here
    uint8_t best_suit;
    uint8_t depth;
    u32 gen;
};
// Each k-combination carries a 64-bit filter of the patterns stored under it, so most
// combinations that cannot match are skipped without touching the entry table.
static const int INLINE_KS = 12;
struct KNode { u16 ks; int32_t next; u64 filt; };
struct PBucket {
    u64 lens; u32 gen; uint8_t lt; uint8_t n_inline; int32_t head; int32_t count;
    u16 ks[INLINE_KS]; u64 filt[INLINE_KS];   // the first k-combinations, stored inline
};

static const int PE_BITS = 21;        // 2M entries x 40 bytes = 80 MB
static const int PB_BITS = 17;        // 128K buckets x 128 bytes
static const int KPOOL_MAX = 1 << 21;
static const int MAX_KS_PER_BUCKET = 512;
static PEntry* g_pe = nullptr;
static PBucket* g_pb = nullptr;
static KNode* g_kpool = nullptr;
static int g_kpool_n = 0;

static void tables_new_generation() {
    if (!g_pe) {
        g_pe = (PEntry*)calloc((size_t)1 << PE_BITS, sizeof(PEntry));
        g_pb = (PBucket*)calloc((size_t)1 << PB_BITS, sizeof(PBucket));
        g_kpool = (KNode*)malloc(sizeof(KNode) * (size_t)KPOOL_MAX);
    }
    g_gen++;
    g_kpool_n = 0;
}

// ---------------------------------------------------------------- solver
// What a search result depended on, per suit:
//   r: ranks that won a trick by rank, or that a bound or table entry relied on;
//   g: rank intervals of equivalent-card groups that were searched through one
//      member only (their members must stay on the same side of the cut-off
//      between "exact" and "small" cards, see pt_store).
struct Rel { u16 r[4], g[4]; };
struct Move { int8_t suit, rank; u16 group; int score; };  // group: the equivalent cards it stands for

struct Solver {
    u16 hand[4][4];
    int trump;      // 0..3, 4 = NT
    int flags;
    int leader;     // player who led the current trick
    int trick_pos;  // cards already played to the current trick (0..3)
    int t_suit[4], t_rank[4];
    int ns_won;     // tricks N-S won since the root
    int tricks_left;  // tricks not yet completed (counts the current one)
    u64 nodes, node_limit;
    bool aborted;

    int to_play() const { return (leader + trick_pos) & 3; }

    // index (0..3, position in trick) of the card currently winning the trick
    int winning_pos(int n) const {
        int w = 0;
        for (int i = 1; i < n; i++) {
            if (t_suit[i] == t_suit[w]) { if (t_rank[i] > t_rank[w]) w = i; }
            else if (t_suit[i] == trump) w = i;
        }
        return w;
    }

    // a completed trick is saved in the undo record: the next trick reuses t_suit/t_rank
    struct Undo { int prev_leader; int8_t completed, ns_gain; int8_t ts[4], tr[4]; int8_t win_suit, win_rank, by_rank; };

    Undo play(int s, int r) {
        Undo u{leader, 0, 0, {0, 0, 0, 0}, {0, 0, 0, 0}, 0, 0, 0};
        int p = to_play();
        hand[p][s] &= (u16)~(1u << r);
        t_suit[trick_pos] = s; t_rank[trick_pos] = r;
        trick_pos++;
        if (trick_pos == 4) {
            int w = winning_pos(4);
            int winner = (leader + w) & 3;
            u.completed = 1;
            // the winner's rank mattered only if it beat another card of its own suit
            u.win_suit = (int8_t)t_suit[w]; u.win_rank = (int8_t)t_rank[w];
            for (int i = 0; i < 4; i++) if (i != w && t_suit[i] == t_suit[w]) u.by_rank = 1;
            for (int i = 0; i < 4; i++) { u.ts[i] = (int8_t)t_suit[i]; u.tr[i] = (int8_t)t_rank[i]; }
            u.ns_gain = (winner & 1) == 0;
            ns_won += u.ns_gain;
            leader = winner;
            trick_pos = 0;
            tricks_left--;
        }
        return u;
    }

    void unplay(const Undo& u) {
        if (u.completed) {
            ns_won -= u.ns_gain;
            leader = u.prev_leader;
            trick_pos = 4;
            tricks_left++;
            for (int i = 0; i < 4; i++) { t_suit[i] = u.ts[i]; t_rank[i] = u.tr[i]; }
        }
        trick_pos--;
        int p = to_play();
        hand[p][t_suit[trick_pos]] |= (u16)(1u << t_rank[trick_pos]);
    }

    u16 suit_all(int s) const { return hand[0][s] | hand[1][s] | hand[2][s] | hand[3][s]; }

    // cards of suit s still "in play" for comparisons: in hands or on the table
    u16 suit_present(int s) const {
        u16 m = suit_all(s);
        for (int i = 0; i < trick_pos; i++) if (t_suit[i] == s) m |= (u16)(1u << t_rank[i]);
        return m;
    }

    // ---------------------------------------------------- move generation
    int gen_moves(Move* out) const {
        int p = to_play();
        int n = 0;
        int lo_s = 0, hi_s = 3;
        if (trick_pos > 0 && hand[p][t_suit[0]]) lo_s = hi_s = t_suit[0];
        for (int s = lo_s; s <= hi_s; s++) {
            unsigned m = hand[p][s];
            if (!m) continue;
            unsigned present = suit_present(s);
            int prev = -1;
            while (m) {
                int r = top_bit(m);
                m &= ~(1u << r);
                if ((flags & F_EQUIV) && prev >= 0) {
                    // a card joins the group above it when no card held elsewhere
                    // (or lying on the table) sits between them
                    unsigned between = ((1u << prev) - 1) & ~((2u << r) - 1);
                    if ((present & between) == 0) {
                        out[n - 1].rank = (int8_t)r; out[n - 1].group |= (u16)(1u << r); prev = r; continue;
                    }
                }
                out[n].suit = (int8_t)s; out[n].rank = (int8_t)r; out[n].group = (u16)(1u << r); out[n].score = 0; n++;
                prev = r;
            }
        }
        return n;
    }

    // a hand that can ruff suit s: void in s and holding trumps
    bool can_ruff(int h, int s) const {
        return trump < 4 && s != trump && hand[h][s] == 0 && hand[h][trump] != 0;
    }

    void order_moves(Move* mv, int n, int tt_suit) const {
        int p = to_play();
        int part = (p + 2) & 3, lho = (p + 1) & 3, rho = (p + 3) & 3;
        for (int i = 0; i < n; i++) {
            int s = mv[i].suit, r = mv[i].rank;
            int sc = 0;
            if (trick_pos == 0) {
                unsigned all = suit_all(s);
                int top = top_bit(all);
                bool opp_ruff = can_ruff(lho, s) || can_ruff(rho, s);
                int len = popc(hand[p][s]);
                if (r == top && !opp_ruff) sc = 100 + r;                        // cash a winner
                else if ((hand[part][s] >> top & 1) && !opp_ruff) sc = 80 - r;  // low to partner's winner
                else if (can_ruff(part, s) && !opp_ruff) sc = 70 - r;           // give partner a ruff
                else sc = 20 + 2 * len - r;                                     // low from length
                if (s == trump && trump < 4) sc += 5;
                if (s == tt_suit) sc += 1000;
            } else {
                int w = winning_pos(trick_pos);
                bool partner_winning = ((trick_pos - w) & 1) == 0;  // partner played 2 positions back
                int led = t_suit[0];
                bool beats;
                if (s == t_suit[w]) beats = r > t_rank[w];
                else beats = (s == trump && trump < 4);
                bool last = trick_pos == 3;
                if (s != led && !(s == trump && trump < 4)) {
                    sc = 10 + popc(hand[p][s]) - r;                       // discard low from length
                } else if (partner_winning) {
                    sc = 50 - r;                                          // don't waste a high card
                } else if (beats) {
                    if (s != led) sc = 75 - r;                            // ruff, as cheaply as possible
                    else if (last || trick_pos == 2) sc = 80 - r;         // win as cheaply as possible
                    else {
                        // second hand: winning is good if it can't be beaten later
                        unsigned higher_elsewhere = suit_all(s) & ~hand[p][s] & ~((2u << r) - 1);
                        sc = higher_elsewhere ? 30 - r : 70 - r;
                    }
                } else {
                    sc = 50 - r;                                          // can't win: play low
                }
            }
            mv[i].score = sc;
        }
        // insertion sort, highest score first (n <= 13)
        for (int i = 1; i < n; i++) {
            Move m = mv[i]; int j = i - 1;
            while (j >= 0 && mv[j].score < m.score) { mv[j + 1] = mv[j]; j--; }
            mv[j + 1] = m;
        }
    }

    // ---------------------------------------------------- cheap bounds
    // Quick tricks: tricks the player on lead (at a trick start) can surely take by
    // cashing top cards. Conservative, so a valid lower bound:
    //  - only cards above every other card of the suit count (nobody can beat them),
    //  - in a side suit, cashing stops before any hand holding trumps runs out of the
    //    suit (it could ruff; for partner that would also move the lead).
    // rel receives the cards the bound relies on.
    int cashable(int p, u16* rel) const {
        int qt = 0;
        for (int s = 0; s < 4; s++) {
            unsigned m = hand[p][s];
            if (!m) continue;
            unsigned others = hand[(p + 1) & 3][s] | hand[(p + 2) & 3][s] | hand[(p + 3) & 3][s];
            int k = 0;
            for (int r = 12; r >= 0; r--) {
                if (m >> r & 1) k++;
                else if (others >> r & 1) break;
            }
            if (!k) continue;
            if (trump < 4 && s != trump) {
                for (int d = 1; d <= 3; d++) {
                    int h = (p + d) & 3;
                    if (hand[h][trump]) k = std::min(k, popc(hand[h][s]));
                }
            }
            for (int r = 12, c = 0; r >= 0 && c < k; r--) if (m >> r & 1) { rel[s] |= (u16)(1u << r); c++; }
            qt += k;
        }
        return qt;
    }

    // The leader cashes its own winners, or leads a low card to partner's winners in
    // a suit where partner holds the top card, after which partner cashes all of its own.
    int quick_tricks(u16* rel) const {
        int p = leader, q = (leader + 2) & 3;
        u16 a[4] = {0, 0, 0, 0};
        int best = cashable(p, a);
        int best_entry = -1;
        u16 b[4] = {0, 0, 0, 0};
        int part = 0;
        bool part_done = false;
        for (int s = 0; s < 4; s++) {
            unsigned all = suit_all(s);
            if (!hand[p][s] || !(hand[q][s] >> top_bit(all) & 1)) continue;   // need a card to lead, partner on top
            if (trump < 4 && s != trump && (can_ruff((p + 1) & 3, s) || can_ruff((p + 3) & 3, s))) continue;
            if (!part_done) { part = cashable(q, b); part_done = true; }
            if (part > best) { best = part; best_entry = s; }
        }
        if (best_entry >= 0) memcpy(a, b, sizeof(b));
        for (int s = 0; s < 4; s++) rel[s] |= a[s];
        return std::min(best, tricks_left);
    }

    // Top trumps: a trump higher than every trump the opponents hold wins the trick
    // whenever it is played (and each card is played on a different trick), so a
    // side takes at least as many tricks as either of its hands has such trumps.
    int top_trumps(int side, u16* rel) const {
        if (trump == 4) return 0;
        unsigned opp = hand[side ^ 1][trump] | hand[(side ^ 1) + 2][trump];
        int above = opp ? top_bit(opp) + 1 : 0;
        int best = std::max(popc((unsigned)hand[side][trump] >> above), popc((unsigned)hand[side + 2][trump] >> above));
        if (best && opp) rel[trump] |= (u16)(suit_all(trump) & ~((1u << (above - 1)) - 1));  // the opponents' top trump and up
        return best;
    }

    // ---------------------------------------------------- simple TT keys
    void make_key(u64* k) const {
        u64 w[3] = {0, 0, 0};
        int bit = 0;
        auto put = [&](u64 v, int nb) {
            int word = bit >> 6, off = bit & 63;
            w[word] |= v << off;
            if (off + nb > 64) w[word + 1] |= v >> (64 - off);
            bit += nb;
        };
        put((u64)leader, 2);
        put((u64)trump, 3);
        for (int s = 0; s < 4; s++) {
            if (flags & F_RELRANK) {   // owners of the remaining cards, top down
                put((u64)popc(suit_all(s)), 4);
                for (int r = 12; r >= 0; r--)
                    for (int h = 0; h < 4; h++)
                        if (hand[h][s] >> r & 1) { put((u64)h, 2); break; }
            } else {                   // owner (or "gone") of each of the 13 ranks
                for (int r = 12; r >= 0; r--) {
                    int o = 0;
                    for (int h = 0; h < 4; h++) if (hand[h][s] >> r & 1) { o = h + 1; break; }
                    put((u64)o, 3);
                }
            }
        }
        k[0] = w[0]; k[1] = w[1]; k[2] = w[2];
    }

    TTEntry* tt_probe(const u64* k, bool create) const {
        u64 h = mix(k[0] ^ mix(k[1] ^ mix(k[2])));
        size_t i = (size_t)(h & (((u64)1 << TT_BITS) - 1)) & ~(size_t)1;  // 2-way bucket
        TTEntry* a = &g_tt[i];
        TTEntry* b = &g_tt[i + 1];
        for (TTEntry* e : {a, b})
            if (e->gen == g_gen && e->k0 == k[0] && e->k1 == k[1] && e->k2 == k[2]) return e;
        if (!create) return nullptr;
        TTEntry* v;
        if (a->gen != g_gen) v = a;
        else if (b->gen != g_gen) v = b;
        else v = (a->depth <= b->depth) ? a : b;
        v->k0 = k[0]; v->k1 = k[1]; v->k2 = k[2];
        v->lb = 0; v->ub = (int8_t)tricks_left; v->best_suit = 255;
        v->depth = (uint8_t)tricks_left; v->gen = g_gen;
        return v;
    }

    // ---------------------------------------------------- partition TT
    // lengths of every suit in every hand, and per suit the owners of all remaining
    // cards from the top down (2 bits each)
    void signature(u64& lens, u32* pat) const {
        lens = 0;
        for (int h = 0; h < 4; h++)
            for (int s = 0; s < 4; s++) lens |= (u64)popc(hand[h][s]) << ((h * 4 + s) * 4);
        for (int s = 0; s < 4; s++) {
            u32 v = 0; int i = 0;
            for (int r = 12; r >= 0; r--) {
                unsigned bit = 1u << r;
                if (!(suit_all(s) & bit)) continue;
                int h = (hand[0][s] & bit) ? 0 : (hand[1][s] & bit) ? 1 : (hand[2][s] & bit) ? 2 : 3;
                v |= (u32)h << (2 * i); i++;
            }
            pat[s] = v;
        }
    }

    static inline u32 low_bits(int k) { return k >= 16 ? 0xffffffffu : ((1u << (2 * k)) - 1); }

    static u64 entry_hash(u64 lens, uint8_t lt, u16 ks, const u32* pat) {
        u64 h = mix(lens ^ ((u64)lt << 50) ^ ((u64)ks << 32));
        h = mix(h ^ pat[0] ^ ((u64)pat[1] << 32));
        return mix(h ^ pat[2] ^ ((u64)pat[3] << 32));
    }

    PBucket* find_bucket(u64 lens, uint8_t lt, bool create) const {
        u64 h = mix(lens ^ ((u64)lt << 58) ^ 0x9e3779b97f4a7c15ULL);
        size_t mask = ((size_t)1 << PB_BITS) - 1;
        for (int i = 0; i < 8; i++) {
            PBucket* b = &g_pb[(h + i) & mask];
            if (b->gen == g_gen && b->lens == lens && b->lt == lt) return b;
            if (b->gen != g_gen) {
                if (!create) return nullptr;
                b->gen = g_gen; b->lens = lens; b->lt = lt; b->head = -1; b->count = 0; b->n_inline = 0;
                return b;
            }
        }
        return nullptr;  // neighbourhood full: don't store
    }

    PEntry* find_entry(u64 lens, uint8_t lt, u16 ks, const u32* pat, bool create, u64 h) const {
        size_t mask = ((size_t)1 << PE_BITS) - 1;
        PEntry* victim = nullptr;
        for (int i = 0; i < 4; i++) {
            PEntry* e = &g_pe[(h + i) & mask];
            if (e->gen == g_gen) {
                if (e->lens == lens && e->lt == lt && e->ks == ks && !memcmp(e->pat, pat, 16)) return e;
                if (!victim || (victim->gen == g_gen && e->depth < victim->depth)) victim = e;
            } else if (!victim || victim->gen == g_gen) {
                victim = e;
            }
        }
        if (!create) return nullptr;
        victim->lens = lens; victim->lt = lt; victim->ks = ks; memcpy(victim->pat, pat, 16);
        victim->lb = 0; victim->ub = (int8_t)tricks_left; victim->best_suit = 255;
        victim->depth = (uint8_t)tricks_left; victim->gen = g_gen;
        return victim;
    }

    // the top k cards of each suit, as rank bits: what a matched entry depended on
    void top_cards(u16 ks, u16* rel) const {
        for (int s = 0; s < 4; s++) {
            int k = ks >> (4 * s) & 15;
            unsigned all = suit_all(s), m = 0;
            for (int c = 0; c < k && all; c++) { int r = top_bit(all); all &= ~(1u << r); m |= 1u << r; }
            rel[s] = (u16)m;
        }
    }

    // 1 = N-S surely make `need` more tricks, 0 = surely not, -1 = unknown
    int pt_lookup(int need, u64 lens, const u32* pat, u16* rel, int* hint) const {
        uint8_t lt = (uint8_t)(leader | trump << 2);
        const PBucket* b = find_bucket(lens, lt, false);
        if (!b) return -1;
        int ni = b->n_inline;
        for (int i = 0, j = b->head; i < ni || j >= 0; i++) {
            u16 ks; u64 filt;
            if (i < ni) { ks = b->ks[i]; filt = b->filt[i]; }
            else { ks = g_kpool[j].ks; filt = g_kpool[j].filt; j = g_kpool[j].next; }
            u32 mp[4];
            for (int s = 0; s < 4; s++) mp[s] = pat[s] & low_bits(ks >> (4 * s) & 15);
            u64 h = entry_hash(lens, lt, ks, mp);
            if (!(filt >> (h >> 58) & 1)) continue;
            const PEntry* e = find_entry(lens, lt, ks, mp, false, h);
            if (!e) continue;
            if (e->lb >= need) { top_cards(ks, rel); return 1; }
            if (e->ub < need) { top_cards(ks, rel); return 0; }
            if (e->best_suit != 255) *hint = e->best_suit;
        }
        return -1;
    }

    // The entry fixes every card from the lowest relevant one upwards ("exact" cards);
    // the rest are interchangeable small cards. If that cut-off falls inside a group of
    // equivalent cards that was searched through one member, the whole group must be
    // exact (otherwise a position with the same exact cards could split the group and
    // offer a move that was never searched), so the cut-off moves below the group.
    void pt_store(int need, bool result, const Rel& rel, int best_suit, u64 lens, const u32* pat_in) {
        uint8_t lt = (uint8_t)(leader | trump << 2);
        u32 pat[4];
        u16 ks = 0;
        for (int s = 0; s < 4; s++) {
            int k = 0;
            if (rel.r[s]) {
                int t = __builtin_ctz(rel.r[s]);
                unsigned g = rel.g[s];
                if (t > 0 && (g >> t & 1) && (g >> (t - 1) & 1)) {
                    t--;
                    while (t > 0 && (g >> (t - 1) & 1)) t--;
                }
                k = popc((unsigned)suit_all(s) >> t);
            }
            ks |= (u16)(k << (4 * s));
            pat[s] = pat_in[s] & low_bits(k);
        }
        PBucket* b = find_bucket(lens, lt, true);
        if (!b) return;
        u64 h = entry_hash(lens, lt, ks, pat);
        u64 bit = (u64)1 << (h >> 58);
        u64* filt = nullptr;
        for (int i = 0; i < b->n_inline; i++) if (b->ks[i] == ks) { filt = &b->filt[i]; break; }
        if (!filt)
            for (int j = b->head; j >= 0; j = g_kpool[j].next) if (g_kpool[j].ks == ks) { filt = &g_kpool[j].filt; break; }
        if (!filt) {
            if (b->count >= MAX_KS_PER_BUCKET) return;
            if (b->n_inline < INLINE_KS) {
                b->ks[b->n_inline] = ks; b->filt[b->n_inline] = 0; filt = &b->filt[b->n_inline]; b->n_inline++;
            } else {
                if (g_kpool_n >= KPOOL_MAX) {   // index full: empty all tables (only costs speed)
                    g_gen++; g_kpool_n = 0;
                    b = find_bucket(lens, lt, true);
                    b->ks[0] = ks; b->filt[0] = 0; filt = &b->filt[0]; b->n_inline = 1;
                } else {
                    KNode& kn = g_kpool[g_kpool_n];
                    kn.ks = ks; kn.filt = 0; kn.next = b->head;
                    b->head = g_kpool_n++;
                    filt = &kn.filt;
                }
            }
            b->count++;
        }
        *filt |= bit;
        PEntry* e = find_entry(lens, lt, ks, pat, true, h);
        if (result) { if (e->lb < need) e->lb = (int8_t)need; }
        else { if (e->ub > need - 1) e->ub = (int8_t)(need - 1); }
        if (best_suit != 255) e->best_suit = (uint8_t)best_suit;
    }

    // ---------------------------------------------------- yes/no search
    // Can N-S reach `target` tricks in total (counting ns_won so far)?
    // rel receives what the answer depended on (for the partition table).
    bool search(int target, Rel& rel) {
        memset(&rel, 0, sizeof(rel));
        if (aborted) return false;
        if (++nodes > node_limit) { aborted = true; return false; }

        const bool use_tt = flags & F_TT, use_pt = use_tt && (flags & F_PARTITION);
        u64 key[3], lens = 0;
        u32 pat[4];
        int tt_suit = 255;
        int need = target - ns_won;
        if (trick_pos == 0) {
            if (need <= 0) return true;
            if (need > tricks_left) return false;
            if (flags & F_QUICK) {
                u16 b[4] = {0, 0, 0, 0};
                int lead_side = leader & 1;
                int lead_sure = std::max(quick_tricks(b), top_trumps(lead_side, b));
                if (lead_side == 0 ? lead_sure >= need : tricks_left - lead_sure < need) {
                    memcpy(rel.r, b, sizeof(b)); return lead_side == 0;
                }
                u16 c[4] = {0, 0, 0, 0};
                int other_sure = top_trumps(lead_side ^ 1, c);
                if (lead_side == 1 ? other_sure >= need : tricks_left - other_sure < need) {
                    memcpy(rel.r, c, sizeof(c)); return lead_side == 1;
                }
            }
            if (use_pt) {
                signature(lens, pat);
                int d = pt_lookup(need, lens, pat, rel.r, &tt_suit);
                if (d >= 0) return d == 1;
            } else if (use_tt) {
                make_key(key);
                TTEntry* e = tt_probe(key, false);
                if (e) {
                    if (e->lb >= need) return true;
                    if (e->ub < need) return false;
                    tt_suit = e->best_suit;
                }
            }
        }

        Move mv[13];
        int n = gen_moves(mv);
        if (flags & F_ORDER) order_moves(mv, n, tt_suit);
        bool ns_move = (to_play() & 1) == 0;
        bool result = !ns_move;  // N-S: fails unless a move works; E-W: succeeds unless a move refutes
        int best_suit = 255;
        Rel c;
        for (int i = 0; i < n; i++) {
            Undo u = play(mv[i].suit, mv[i].rank);
            bool r = search(target, c);
            if (u.completed && u.by_rank) c.r[u.win_suit] |= (u16)(1u << u.win_rank);
            unplay(u);
            if (aborted) return false;
            if (r == ns_move) {  // this move decides the node: only its line matters
                result = r; best_suit = mv[i].suit;
                rel = c;
                break;
            }
            // every move was needed
            for (int s = 0; s < 4; s++) { rel.r[s] |= c.r[s]; rel.g[s] |= c.g[s]; }
            u16 grp = mv[i].group;
            if (grp & (grp - 1)) {  // this move stood for several equivalent cards
                int lo = __builtin_ctz(grp), hi = top_bit(grp);
                rel.g[mv[i].suit] |= (u16)(((2u << hi) - 1) & ~((1u << lo) - 1));
            }
        }

        if (trick_pos == 0) {
            if (use_pt) pt_store(need, result, rel, best_suit, lens, pat);
            else if (use_tt) {
                TTEntry* e = tt_probe(key, true);
                if (result) { if (e->lb < need) e->lb = (int8_t)need; }
                else { if (e->ub > need - 1) e->ub = (int8_t)(need - 1); }
                if (best_suit != 255) e->best_suit = (uint8_t)best_suit;
            }
        }
        return result;
    }

    // ---------------------------------------------------- baseline minimax
    int minimax(int alpha, int beta) {
        if (aborted) return 0;
        if (++nodes > node_limit) { aborted = true; return 0; }
        if (tricks_left == 0) return ns_won;
        Move mv[13];
        int n = gen_moves(mv);
        if (flags & F_ORDER) order_moves(mv, n, 255);
        bool ns_move = (to_play() & 1) == 0;
        int best = ns_move ? -1 : 99;
        for (int i = 0; i < n; i++) {
            Undo u = play(mv[i].suit, mv[i].rank);
            int v = minimax(alpha, beta);
            unplay(u);
            if (aborted) return 0;
            if (ns_move) { if (v > best) best = v; if (best > alpha) alpha = best; }
            else { if (v < best) best = v; if (best < beta) beta = best; }
            if (alpha >= beta) break;
        }
        return best;
    }

    // N-S tricks from the current position on (the current trick included)
    int value(int guess = -1) {
        int base = ns_won;
        if (flags & F_MINIMAX) {
            int v = minimax(-1, 99);
            return aborted ? -1 : v - base;
        }
        int lo = 0, hi = tricks_left;
        int t = guess < 0 ? (lo + hi + 1) / 2 : std::min(std::max(guess, 1), hi);
        while (lo < hi) {
            Rel rel;
            bool ok = search(base + t, rel);
            if (aborted) return -1;
            if (ok) lo = t; else hi = t - 1;
            if (guess < 0) t = (lo + hi + 1) / 2;
            else t = ok ? lo + 1 : hi;
        }
        return lo;
    }
};

// ---------------------------------------------------------------- C API
// hands: 16 values, hands[player*4 + suit] = 13-bit rank mask.
// trick: the cards already played to the current trick, in order, each as
// suit*16 + rank; trick_len of them (0..3), led by `leader`.
static bool setup(Solver& S, const uint16_t* hands, int trump, int leader, int trick_len,
                  const int* trick, int flags, uint64_t node_limit) {
    memset(&S, 0, sizeof(S));
    if (trump < 0 || trump > 4 || leader < 0 || leader > 3 || trick_len < 0 || trick_len > 3) return false;
    for (int p = 0; p < 4; p++)
        for (int s = 0; s < 4; s++) S.hand[p][s] = hands[p * 4 + s] & 0x1fff;
    S.trump = trump;
    S.flags = flags;
    if (!(flags & F_TT)) S.flags &= ~(F_RELRANK | F_PARTITION);
    if ((S.flags & F_TT) && !(S.flags & F_PARTITION) && !g_tt)   // the simple table is only for comparisons
        g_tt = (TTEntry*)calloc((size_t)1 << TT_BITS, sizeof(TTEntry));
    S.leader = leader;
    S.trick_pos = trick_len;
    for (int i = 0; i < trick_len; i++) { S.t_suit[i] = trick[i] >> 4; S.t_rank[i] = trick[i] & 15; }
    int p = S.to_play();
    int cards_to_play = popc(S.hand[p][0]) + popc(S.hand[p][1]) + popc(S.hand[p][2]) + popc(S.hand[p][3]);
    S.tricks_left = cards_to_play;  // the player to act holds one card per unfinished trick
    S.node_limit = node_limit ? node_limit : ~(u64)0;
    return cards_to_play > 0;
}

// N-S tricks from this position on (current trick included); -1 = bad input or node limit hit.
DDS_API int dds_solve(const uint16_t* hands, int trump, int leader, int trick_len, const int* trick,
                      int flags, uint64_t node_limit, uint64_t* nodes_out) {
    tables_new_generation();
    Solver S;
    if (!setup(S, hands, trump, leader, trick_len, trick, flags, node_limit)) return -1;
    int v = S.value();
    if (nodes_out) *nodes_out = S.nodes;
    return v;
}

// Score every legal card for the player to act: out_cards[i] = suit*16+rank,
// out_ns[i] = N-S tricks from here if that card is played (then perfect play).
// Returns the number of legal cards, or -1.
DDS_API int dds_score_cards(const uint16_t* hands, int trump, int leader, int trick_len, const int* trick,
                            int flags, uint64_t node_limit, int* out_cards, int* out_ns, uint64_t* nodes_out) {
    tables_new_generation();
    Solver S;
    if (!setup(S, hands, trump, leader, trick_len, trick, flags, node_limit)) return -1;
    int saved_flags = S.flags;
    S.flags &= ~F_EQUIV;  // list every card, not one per equivalence group
    Move mv[13];
    int n = S.gen_moves(mv);
    S.flags = saved_flags;
    u64 total = 0;
    for (int i = 0; i < n; i++) {
        Solver T = S;
        T.play(mv[i].suit, mv[i].rank);
        int gained = T.ns_won;  // 1 if this card completed a trick that N-S won
        T.ns_won = 0;
        int v = (T.tricks_left == 0) ? 0 : T.value();
        total += T.nodes;
        if (v < 0) { if (nodes_out) *nodes_out = total; return -1; }
        out_cards[i] = mv[i].suit * 16 + mv[i].rank;
        out_ns[i] = v + gained;
    }
    if (nodes_out) *nodes_out = total;
    return n;
}

// Full double-dummy table for a whole deal (13 cards per hand).
// out[strain*4 + declarer] = tricks taken by `declarer`'s side when the
// player on declarer's left leads; strain 0=S 1=H 2=D 3=C 4=NT.
DDS_API int dds_table(const uint16_t* hands, int flags, uint64_t node_limit, int* out, uint64_t* nodes_out) {
    u64 total = 0;
    for (int strain = 0; strain < 5; strain++) {
        tables_new_generation();  // entries stay valid across the four leads of one strain
        int guess = -1;
        for (int decl = 0; decl < 4; decl++) {
            Solver S;
            int leader = (decl + 1) & 3;
            if (!setup(S, hands, strain, leader, 0, nullptr, flags, node_limit)) return -1;
            int ns = S.value(guess);
            guess = ns;
            total += S.nodes;
            if (ns < 0) { if (nodes_out) *nodes_out = total; return -1; }
            int tricks = S.tricks_left;
            out[strain * 4 + decl] = (decl & 1) == 0 ? ns : tricks - ns;
        }
    }
    if (nodes_out) *nodes_out = total;
    return 0;
}

DDS_API void dds_clear_tt() { tables_new_generation(); }

DDS_API int dds_version() { return 2; }

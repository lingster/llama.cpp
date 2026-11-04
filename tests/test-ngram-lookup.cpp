#include "ngram-map.h"

#undef NDEBUG
#include <cassert>
#include <cstdio>

static common_params_speculative_lookup make_params() {
    common_params_speculative_lookup params;
    params.enabled  = true;
    params.n_min    = 3;
    params.n_max    = 6;
    params.n_strong = 5;
    params.n_agree  = 2;
    params.n_draft  = 8;
    return params;
}

static void test_find() {
    const auto params = make_params();
    llama_tokens out;

    // no match
    assert(common_ngram_lookup_find(params, { 1, 2, 3, 4, 5, 6 }, 7, out) == 0);
    assert(out.empty());

    // match of length 3 (2 3 4), followed by 5 6 7
    assert(common_ngram_lookup_find(params, { 1, 2, 3, 4, 5, 6, 7, 9, 2, 3 }, 4, out) == 3);
    assert((out == llama_tokens { 5, 6, 7, 9, 2, 3, 4 }));

    // the longest match wins over a more recent shorter one
    {
        const llama_tokens hist = { 8, 1, 2, 3, 50, 51, 9, 1, 2, 3, 60, 61, 8, 1, 2 };
        assert(common_ngram_lookup_find(params, hist, 3, out) == 4);
        assert(out[0] == 50 && out[1] == 51);
    }

    // equal length: the most recent wins
    {
        const llama_tokens hist = { 1, 2, 3, 50, 0, 1, 2, 3, 60, 5, 1, 2 };
        assert(common_ngram_lookup_find(params, hist, 3, out) == 3);
        assert(out[0] == 60);
    }

    // a match can overlap the suffix: period 2 repeats
    {
        const llama_tokens hist = { 7, 8, 7, 8, 7, 8, 7 };
        assert(common_ngram_lookup_find(params, hist, 8, out) == 6);
        assert((out == llama_tokens { 7, 8 }));
    }

    // the continuation is capped at n_draft
    {
        auto p = params;
        p.n_draft = 2;
        assert(common_ngram_lookup_find(p, { 1, 2, 3, 4, 5, 6, 0, 1, 2 }, 3, out) == 3);
        assert((out == llama_tokens { 4, 5 }));
    }
}

static void test_fuse() {
    const auto params = make_params();

    // strong match replaces the draft
    {
        llama_tokens draft = { 10, 11, 12 };
        const auto res = common_ngram_lookup_fuse(params, { 20, 21, 22, 23 }, 5, draft, nullptr);
        assert(res.n_replaced == 3 && res.has_tail);
        assert((draft == llama_tokens { 20, 21, 22 }));
    }

    // weak match without agreement is ignored, and offers no tail
    {
        llama_tokens draft = { 10, 11, 12 };
        const auto res = common_ngram_lookup_fuse(params, { 10, 21, 22, 23 }, 3, draft, nullptr);
        assert(res.n_replaced == 0 && !res.has_tail);
        assert((draft == llama_tokens { 10, 11, 12 }));
    }

    // weak match with agreement on the first n_agree tokens replaces the rest
    {
        llama_tokens draft = { 10, 11, 12 };
        const auto res = common_ngram_lookup_fuse(params, { 10, 11, 22, 23 }, 3, draft, nullptr);
        assert(res.n_replaced == 1 && res.has_tail);
        assert((draft == llama_tokens { 10, 11, 22 }));
    }

    // a short lookup keeps the tokens of the model draft after it
    {
        llama_tokens draft = { 10, 11, 12, 13 };
        const auto res = common_ngram_lookup_fuse(params, { 20, 21 }, 5, draft, nullptr);
        assert(res.n_replaced == 2 && !res.has_tail);
        assert((draft == llama_tokens { 20, 21, 12, 13 }));
    }

    // replaced and appended positions get a point mass
    {
        llama_tokens draft = { 10, 11 };
        std::vector<std::vector<llama_token_data>> q = {
            { { 10, 0.0f, 0.6f }, { 30, 0.0f, 0.4f } },
            { { 11, 0.0f, 0.9f }, { 31, 0.0f, 0.1f } },
        };
        const llama_tokens lookup = { 10, 21, 22 };
        const auto res = common_ngram_lookup_fuse(params, lookup, 5, draft, &q);
        assert(res.n_replaced == 1 && res.has_tail);
        assert(q[0].size() == 2);
        assert(q[1].size() == 1 && q[1][0].id == 21 && q[1][0].p == 1.0f);

        assert(common_ngram_lookup_extend(lookup, draft, &q) == 1);
        assert((draft == llama_tokens { 10, 21, 22 }));
        assert(q.size() == 3 && q[2].size() == 1 && q[2][0].id == 22 && q[2][0].p == 1.0f);
    }
}

int main() {
    test_find();
    test_fuse();

    printf("OK\n");
    return 0;
}
